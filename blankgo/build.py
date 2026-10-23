"""Build a tiny GPT-2-shaped model on a known blank, as an exact thing.
    python3 build.py width=64 layers=2 recipe=A [steps=3000] [snaps=0,10,30,100,300,1000,2000,3000]

EXACT BOOKKEEPING. Every weight is a whole number at 2^-30 held in int64. Each step:
  - the model is run in float64 from the current whole numbers (n x 2^-30, exact), on 16 windows of 128 words of the
    building part (the same windows, in the same order, for every recipe);
  - the answer's slope for every weight comes back in float64; the push is worked out from it in float64, rounded to
    the grid, and added as a whole number;
  - the shave (recipe C only) is also a whole number: rint(B x step x wd) taken from the blank's part and
    rint(P x step x wd) taken from the pushes' part, separately, so each part is shaved on its own.
  Four whole-number running totals per weight: W (the weight, changed only by its own adds), R (every push added,
  never shaved), S_B (all shave taken from the blank), S_P (all shave taken from the pushes). Asserted EVERY step for
  every weight:   W == blank - S_B + R - S_P     (the shaved blank plus the pushes as they stand, nothing lost)
  and the float64 numbers the model runs on are exactly W x 2^-30.

THREE RECIPES, the SAME blank, the same windows, the same step-size shape (rise over 30 pushes, half cosine to 10%):
  A  shave off, plain push: push = -(step size) x slope,                    peak step size 0.5
  B  shave off, each weight's push scaled by its own running size of past slopes (beta 0.9 / 0.95, 1e-8 floor),
     peak step size 1e-3
  C  as B, with the shave on: 0.1 x step size per push, on the drawn parts only (plates, both word tables, position
     rows; gains, offsets and biases are never shaved), as Pythia's recipe does
Snapshots: P = W - B_t (the pushes as they stand) as int64, plus S_P for C; the shaved blank B_t is not stored, it is
replayed exactly by model.shave_chain and its hash checked against the one logged here. Held-out loss at each
snapshot on 128 fixed windows of the held-out part (16,384 words).

PREDICTIONS (the session's, written before any run of this script):
  P1. The bookkeeping assert holds at every step of every run; 0 failures. The model run is exactly W x 2^-30.
  P2. Step 0 (the blank itself) held-out loss = ln 8193 + d x 0.02^2 / 2: 9.024 (width 64), 9.037 (width 128), within
      0.01 each (the blank's word scores are a bell curve of spread 0.02 sqrt(d) around 0).
  P3. Recipe C, the blank's drawn part after t pushes: sum|B_t| / sum|B_0| = product of (1 - 0.1 x step size):
      0.99982 (t=10), 0.99149 (100), 0.97201 (300), 0.91276 (1000), 0.84801 (3000), within 1e-5. In A and B the blank
      stays exactly as drawn (S_B = 0 at every step).
  P4. Held-out loss (nats a word):
        step      10        100       300       1000      3000
        B, C      8.0-8.9   6.2-7.0   5.6-6.5   5.0-6.0   4.6-5.6
        A         7.5-8.9   6.5-7.5   6.0-7.2   5.6-6.8   5.5-6.8
      at 3000: wider and deeper is lower (w128 L4 lowest, w64 L2 highest) in every recipe; C within 0.05 of B at every
      snapshot; A above B at every shape.
  P5. Time: about 1 s a step for w128 L4, 0.3-0.5 s for w64 L2, under 2.5 GB.
"""
import os, sys, json, time, hashlib, resource, numpy as np, torch, torch.nn.functional as F
from model import (HERE, GRID, ONE, V, T, SEED, STEPS, layout, drawn, tag, blank, step_size, data, held_windows,
                   to_float, forward, held_loss)

A = dict(a.split("=") for a in sys.argv[1:])
d, L, RECIPE = int(A.get("width", 64)), int(A.get("layers", 2)), A.get("recipe", "A")
NSTEPS = int(A.get("steps", STEPS))
SNAPS = [int(x) for x in A.get("snaps", "0,10,30,100,300,1000,2000,3000").split(",")]
PEAK = {"A": 0.5, "B": 1e-3, "C": 1e-3}[RECIPE]
WD = 0.1 if RECIPE == "C" else 0.0
B1, B2, EPS = 0.9, 0.95, 1e-8
BATCH = 16
name = f"{tag(d, L)}_{RECIPE}"
OUT = os.path.join(HERE, "snaps"); os.makedirs(OUT, exist_ok=True)

blank0 = blank(d, L)
NAMES = [n for n, _, _ in layout(d, L)]
DRAWN = set(drawn(d, L))
build_ids, held_ids = data()
win = held_windows(held_ids)
starts = np.random.default_rng(SEED + 1).integers(0, len(build_ids) - T - 1, size=(STEPS, BATCH))

W = {n: blank0[n].copy() for n in NAMES}
R = {n: np.zeros_like(blank0[n]) for n in NAMES}
SB = {n: np.zeros_like(blank0[n]) for n in NAMES}
SP = {n: np.zeros_like(blank0[n]) for n in NAMES}
m = {n: np.zeros(blank0[n].shape) for n in NAMES}
v = {n: np.zeros(blank0[n].shape) for n in NAMES}
blank_abs = sum(np.abs(blank0[n]).sum() for n in DRAWN)

def h(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
def hall(D): return hashlib.sha256(b"".join(h(D[n]).encode() for n in NAMES)).hexdigest()[:16]

log = dict(width=d, layers=L, recipe=RECIPE, peak=PEAK, wd=WD, steps=NSTEPS, batch=BATCH, snaps={}, curve=[])
t0 = time.time(); fails = 0; zero_push = 0; drawn_count = sum(blank0[n].size for n in DRAWN)

def snapshot(t):
    p = to_float(W)
    for n in NAMES: assert np.array_equal(p[n].numpy() * (2.0 ** GRID), W[n].astype(np.float64))
    lo, se = held_loss(p, win, d, L)
    Bt = {n: blank0[n] - SB[n] for n in NAMES}
    P = {n: W[n] - Bt[n] for n in NAMES}
    rms = lambda D, ks: float(np.sqrt(sum((D[k].astype(np.float64) ** 2).sum() for k in ks) / sum(D[k].size for k in ks)))
    kinds = {"wte": ["wte"], "wpe": ["wpe"], "head": ["head"]}
    for k in ("qkv", "proj", "fc", "mproj"): kinds[k] = [f"{k}.{l}.w" for l in range(L)]
    sizes = {k: dict(blank_rms=rms(Bt, ks) / ONE, pushes_rms=rms(P, ks) / ONE) for k, ks in kinds.items()}
    rec = dict(step=t, held_loss=lo, held_se=se, seconds=time.time() - t0,
               blank_left=float(sum(np.abs(Bt[n]).sum() for n in DRAWN) / blank_abs),
               zero_pushes_this_step=zero_push / drawn_count if t else None,
               hash_W=hall(W), hash_Bt=hall(Bt), hash_R=hall(R), hash_SP=hall(SP), sizes=sizes,
               peak_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20)
    log["snaps"][t] = rec
    save = dict(P={n: P[n] for n in NAMES})
    if RECIPE == "C": save["SP"] = SP
    np.savez_compressed(os.path.join(OUT, f"{name}_step{t}.npz"), **{f"{k}.{n}": a for k, D in save.items() for n, a in D.items()})
    print(f"  step {t:5d}  held-out loss {lo:.4f} (se {se:.4f})  blank left {rec['blank_left']:.6f}  "
          f"{rec['seconds']:.0f}s  {rec['peak_mb']:.0f} MB", flush=True)
    json.dump(log, open(os.path.join(HERE, f"build_{name}.json"), "w"), indent=1)

print(f"build {name}: width {d}, layers {L}, recipe {RECIPE} (peak step {PEAK}, shave {WD}), {NSTEPS} steps", flush=True)
if 0 in SNAPS: snapshot(0)
for t in range(1, NSTEPS + 1):
    p = {n: x.requires_grad_() for n, x in to_float(W).items()}
    ids = torch.from_numpy(np.stack([build_ids[s:s + T + 1] for s in starts[t - 1]]))
    loss = F.cross_entropy(forward(p, ids[:, :-1], d, L).reshape(-1, V), ids[:, 1:].reshape(-1))
    loss.backward()
    lr = step_size(t, PEAK); zero_push = 0
    for n in NAMES:
        g = p[n].grad.numpy()
        if RECIPE == "A":
            upd = lr * g
        else:
            m[n] = B1 * m[n] + (1 - B1) * g; v[n] = B2 * v[n] + (1 - B2) * g * g
            upd = lr * (m[n] / (1 - B1 ** t)) / (np.sqrt(v[n] / (1 - B2 ** t)) + EPS)
        push = -np.rint(upd * (2.0 ** GRID)).astype(np.int64)
        if n in DRAWN:
            zero_push += int((push == 0).sum())
        sb = sp = 0
        if WD and n in DRAWN:
            Bt = blank0[n] - SB[n]; Pt = R[n] - SP[n]
            sb = np.rint(Bt * (lr * WD)).astype(np.int64); sp = np.rint(Pt * (lr * WD)).astype(np.int64)
        W[n] += push - sb - sp
        R[n] += push; SB[n] += sb; SP[n] += sp
        if not np.array_equal(W[n], blank0[n] - SB[n] + R[n] - SP[n]): fails += 1
    assert fails == 0, f"bookkeeping broke at step {t}"
    if t % 10 == 0: log["curve"].append((t, float(loss)))
    if t in SNAPS: snapshot(t)
log["bookkeeping_failures"] = fails
log["bookkeeping_checks"] = NSTEPS * len(NAMES)
json.dump(log, open(os.path.join(HERE, f"build_{name}.json"), "w"), indent=1)
print(f"done: bookkeeping W == blank - S_B + R - S_P held at all {NSTEPS} steps x {len(NAMES)} parts; failures {fails}")
