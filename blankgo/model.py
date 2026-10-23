"""Shared parts of blankgo: the shape, the blank draw, the step-size schedule, the text, and the float64 model.

THE SHAPE: GPT-2's own shape scaled down. Width d = 64 or 128, L = 2 or 4 layers, heads of 16 lanes (d/16 heads, so
the score's 1/sqrt(16) = 1/4 is a shift of 2), 128 positions. Separate input and output word tables (8,193 rows each),
learned position rows. Each layer: water level (layer norm, gain and offset) -> q k v plate (with bias) -> the arrival
(causal, softmax) -> out plate (with bias) -> back into the stream; water level -> up plate (4d, bias) -> GPT-2's own
bend (the tanh form) -> down plate (bias) -> back into the stream. A last water level, then the output word table
(no bias). Plates are [in][out], as GPT-2 keeps them.

THE BLANK: a bell-curve draw from a fixed seed with GPT-2's own spreads: every plate, both word tables and the position
rows 0.02; the two plates that write back into the stream (out, down) 0.02/sqrt(2L). Gains start at exactly 1,
offsets and biases at exactly 0 (these are not drawn). Every value is a whole number at 2^-30 (int64).

THE GRID: every weight is a whole number n standing for n x 2^-30.
"""
import os, math, numpy as np, torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = 30
ONE = 1 << GRID
V, T, HS = 8193, 128, 16
SEED = 20261006
torch.set_default_dtype(torch.float64)


def layout(d, L):
    """(name, shape, spread) in a fixed order; spread is a number (drawn), 'one' (gain) or 'zero' (offset, bias)"""
    s2 = 0.02 / math.sqrt(2 * L)
    out = [("wte", (V, d), 0.02), ("wpe", (T, d), 0.02)]
    for l in range(L):
        out += [(f"ln1.{l}.g", (d,), "one"), (f"ln1.{l}.b", (d,), "zero"),
                (f"qkv.{l}.w", (d, 3 * d), 0.02), (f"qkv.{l}.b", (3 * d,), "zero"),
                (f"proj.{l}.w", (d, d), s2), (f"proj.{l}.b", (d,), "zero"),
                (f"ln2.{l}.g", (d,), "one"), (f"ln2.{l}.b", (d,), "zero"),
                (f"fc.{l}.w", (d, 4 * d), 0.02), (f"fc.{l}.b", (4 * d,), "zero"),
                (f"mproj.{l}.w", (4 * d, d), s2), (f"mproj.{l}.b", (d,), "zero")]
    out += [("lnf.g", (d,), "one"), ("lnf.b", (d,), "zero"), ("head", (d, V), 0.02)]
    return out


def drawn(d, L):
    return [n for n, _, s in layout(d, L) if not isinstance(s, str)]


def tag(d, L): return f"w{d}_L{L}"


def blank(d, L, seed=SEED):
    """the blank: drawn once from the fixed seed, saved, and read back on every later call (checked against a fresh draw)"""
    path = os.path.join(HERE, f"blank_{tag(d, L)}.npz")
    rng = np.random.default_rng(seed)
    fresh = {}
    for n, shape, s in layout(d, L):
        if s == "one": fresh[n] = np.full(shape, ONE, dtype=np.int64)
        elif s == "zero": fresh[n] = np.zeros(shape, dtype=np.int64)
        else: fresh[n] = np.rint(rng.standard_normal(shape) * s * ONE).astype(np.int64)
    if os.path.exists(path):
        old = np.load(path)
        assert all(np.array_equal(old[n], fresh[n]) for n in fresh), "saved blank differs from the seed's draw"
    else:
        np.savez_compressed(path, **fresh)
    return fresh


STEPS, WARM = 3000, 30


def step_size(t, peak):
    """the step size at push t (1..STEPS): rises evenly over the first 30 pushes, then a half cosine down to 10%"""
    if t <= WARM: return peak * t / WARM
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (t - WARM) / (STEPS - WARM))))


def shave_chain(B0, t_end, peak, wd):
    """the blank after t_end shaves (recipe C): B <- B - rint(B x step x wd), the same whole-number step the build takes"""
    B = B0.copy()
    for t in range(1, t_end + 1):
        B -= np.rint(B * (step_size(t, peak) * wd)).astype(np.int64)
    return B


def data():
    z = np.load(os.path.join(HERE, "data.npz"))
    return z["build"].astype(np.int64), z["held"].astype(np.int64)


def held_windows(held, n=128):
    """n fixed held-out windows of T+1 words, evenly spaced over the held-out part"""
    starts = np.linspace(0, len(held) - T - 2, n).astype(np.int64)
    return np.stack([held[s:s + T + 1] for s in starts])


def word_texts(held, n=30, lead=8, read=16):
    """the 30 held-out texts the machine is checked on: lead + read words each, offset from the loss windows"""
    starts = np.linspace(1000, len(held) - 2000, n).astype(np.int64) + 37
    return np.stack([held[s:s + lead + read] for s in starts])


def to_float(Wint):
    """whole numbers at 2^-30 -> float64 tensors; exact (every |n| is far below 2^53)"""
    return {n: torch.from_numpy(w.astype(np.float64) * (2.0 ** -GRID)) for n, w in Wint.items()}


def forward(p, ids, d, L):
    Bn, Tn = ids.shape; nh = d // HS
    x = p["wte"][ids] + p["wpe"][:Tn]
    live = torch.ones(Tn, Tn, dtype=torch.bool).tril()
    for l in range(L):
        h = F.layer_norm(x, (d,), p[f"ln1.{l}.g"], p[f"ln1.{l}.b"], 1e-5)
        q, k, v = (h @ p[f"qkv.{l}.w"] + p[f"qkv.{l}.b"]).split(d, -1)
        q, k, v = (z.view(Bn, Tn, nh, HS).transpose(1, 2) for z in (q, k, v))
        s = (q @ k.transpose(-1, -2)) * 0.25
        a = torch.softmax(s.masked_fill(~live, float("-inf")), -1)
        o = (a @ v).transpose(1, 2).reshape(Bn, Tn, d)
        x = x + o @ p[f"proj.{l}.w"] + p[f"proj.{l}.b"]
        h = F.layer_norm(x, (d,), p[f"ln2.{l}.g"], p[f"ln2.{l}.b"], 1e-5)
        x = x + F.gelu(h @ p[f"fc.{l}.w"] + p[f"fc.{l}.b"], approximate="tanh") @ p[f"mproj.{l}.w"] + p[f"mproj.{l}.b"]
    x = F.layer_norm(x, (d,), p["lnf.g"], p["lnf.b"], 1e-5)
    return x @ p["head"]


def held_loss(p, win, d, L, chunk=16):
    """mean held-out loss (nats a word) over the fixed windows, and its standard error over windows"""
    per = []
    with torch.no_grad():
        for i in range(0, len(win), chunk):
            w = torch.from_numpy(win[i:i + chunk])
            lg = forward(p, w[:, :-1], d, L)
            per.append(F.cross_entropy(lg.reshape(-1, V), w[:, 1:].reshape(-1), reduction="none").view(len(w), -1).mean(1))
    per = torch.cat(per).numpy()
    return float(per.mean()), float(per.std(ddof=1) / math.sqrt(len(per)))
