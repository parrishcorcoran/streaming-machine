"""Independent rebuild of foundation_share.py's numbers, by a separate path. 2026-10-06.

It does not read the whole-number npz files and shares no code with foundation_share.py. It reads Pythia's own fp16
plates (safetensors, one tensor at a time), computes in float64, and writes its own schedule.

  1. s = (W . W0)/(W0 . W0) for every plate and both word tables, from the fp16 values; compared with
     foundation_share.json's exact whole-number s.
  2. Never-built rows at fp16's OWN grain, which is finer than the shared 16-place grid for small values and is the
     only grain at which the 70m's steps 1000 and 10000 can be judged (there a row's values fill 16 places and fp16
     keeps 11): a row passes if every lane has |w - c w0| <= ulp16(w) + c ulp16(w0), c the row's own
     (w . w0)/(w0 . w0). Input and output tables, every size, and the 70m at every step. Control: each row against
     another row's start.
  3. The schedule's product as exp(sum log1p(-0.1 lr_t)), lr_t vectorised (warmup 1,430 steps linear, cosine
     lr -> min_lr over the rest), and the never-built rows' common c against it: the shave the settings leave out,
     ln(c / product), at each step.

PREDICTIONS (written before this script first ran):
  - s agrees with foundation_share.json to 1e-4 (relative) on every plate and table of every size and step.
  - never-built input rows: the same 241 ids at 143000 (every size) and at 50000 (70m); at 1000 and 10000 at least
    241 and containing all 241 (words that first came up after those steps add to the count); the 27 padding rows
    pass at every step. Output table: 0 rows at every size and step. Shuffled control: 0 rows.
  - the never-built rows' c, one value per size and step (spread across rows under 0.5%), equal to what
    foundation_share.log printed for the padding rows: 70m 0.9657, 0.3967, 0.01126, 6.1e-4; 14m 6.1e-4;
    160m 0.01180. The schedule's product recomputed here equals foundation_share.py's to 1e-9.
"""
import os, json, math
import numpy as np
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__)); MD = os.path.join(HERE, "models")
REF = json.load(open(os.path.join(HERE, "foundation_share.json")))
LRS = {"14m": (1e-3, 1e-4), "70m": (1e-3, 1e-4), "160m": (6e-4, 6e-5)}
rng = np.random.default_rng(7)

def product(size, upto, T=143000, warm=1430, wd=0.1):
    lr, mn = LRS[size]; t = np.arange(1, upto + 1, dtype=np.float64)
    l = np.where(t <= warm, lr * t / warm, mn + (lr - mn) * 0.5 * (1 + np.cos(np.pi * (t - warm) / (T - warm))))
    return float(np.exp(np.log1p(-wd * l).sum()))

def tensor(f, key): return f.get_tensor(key)

def plate_keys(L):
    out = []
    for l in range(L):
        p = f"gpt_neox.layers.{l}."
        out += [(f"qkv.{l}", p + "attention.query_key_value.weight"), (f"proj.{l}", p + "attention.dense.weight"),
                (f"fc.{l}", p + "mlp.dense_h_to_4h.weight"), (f"mproj.{l}", p + "mlp.dense_4h_to_h.weight")]
    return out + [("head", "embed_out.weight"), ("embed", "gpt_neox.embed_in.weight")]

def never_built(w16, w016, shuffle=False):
    """row blocks, so the 160m tables stay well under 3 GB"""
    R = w16.shape[0]; perm = rng.permutation(R) if shuffle else np.arange(R); ids = []; c_all = np.zeros(R)
    for a in range(0, R, 4096):
        b = min(R, a + 4096); x16, y16 = w16[a:b], w016[perm[a:b]]
        w, w0 = x16.astype(np.float64), y16.astype(np.float64)
        c = (w * w0).sum(1) / (w0 * w0).sum(1); c_all[a:b] = c
        tol = np.spacing(np.abs(x16)).astype(np.float64) + np.abs(c)[:, None] * np.spacing(np.abs(y16)).astype(np.float64)
        ids += (a + np.flatnonzero((np.abs(w - c[:, None] * w0) <= tol).all(1))).tolist()
    return np.array(ids, dtype=np.int64), c_all

def dot_chunks(x16, y16):
    x, y = x16.reshape(-1), y16.reshape(-1); n = 0.0; d = 0.0
    for a in range(0, x.size, 1 << 22):
        u, v = x[a:a + (1 << 22)].astype(np.float64), y[a:a + (1 << 22)].astype(np.float64); n += float(np.dot(u, v)); d += float(np.dot(v, v))
    return n / d

def run(size, step, ref_s):
    f1 = safe_open(os.path.join(MD, f"pythia-{size}-step{step}", "model.safetensors"), "np")
    f0 = safe_open(os.path.join(MD, f"pythia-{size}-step0", "model.safetensors"), "np")
    L = json.load(open(os.path.join(MD, f"pythia-{size}-step{step}", "config.json")))["num_hidden_layers"]
    worst = 0.0; res = dict(product=product(size, step))
    for name, key in plate_keys(L):
        w16, w016 = tensor(f1, key), tensor(f0, key)
        s = dot_chunks(w16, w016)
        if ref_s is not None:
            worst = max(worst, abs(s / ref_s[name] - 1))
        if name in ("head", "embed"):
            ids, c = never_built(w16, w016); sh, _ = never_built(w16, w016, shuffle=True)
            res[name] = dict(s=s, never_built=len(ids), ids=ids.tolist(), shuffled=len(sh),
                             pad_pass=int(np.isin(np.arange(50277, 50304), ids).sum()),
                             c_med=float(np.median(c[ids])) if len(ids) else None,
                             c_spread=float((c[ids].max() - c[ids].min()) / np.median(c[ids])) if len(ids) else None)
        del w16, w016
    res["worst_rel_s_vs_share"] = worst
    return res

OUT = {}
ref_sizes = REF["sizes"]
for size in ("14m", "70m", "160m"):
    ref = {k: v["s"] for k, v in ref_sizes[size]["plates"].items()}
    r = run(size, 143000, ref); OUT[f"{size}@143000"] = r
for step in (1000, 10000, 50000):
    ref = {k: v["s"] for k, v in REF["steps70"][str(step)]["plates"].items()}
    OUT[f"70m@{step}"] = run("70m", step, ref)

ids241 = set(ref_sizes["70m"]["tables"]["embed"]["idsB"])          # only the first 200 were stored; check containment
full = set(OUT["70m@143000"]["embed"]["ids"])
print("schedule product here vs foundation_share.py:", {k: f"{OUT[k]['product']:.6e}" for k in OUT},
      " share's 143000:", {s: f"{ref_sizes[s]['s_schedule']:.6e}" for s in ref_sizes})
for k, r in OUT.items():
    e, h = r["embed"], r["head"]
    print(f"{k:12s} s vs share worst rel {r['worst_rel_s_vs_share']:.2e} | input never built {e['never_built']} (padding {e['pad_pass']}/27, "
          f"shuffled {e['shuffled']}) c {e['c_med']:.6g} spread {e['c_spread']:.2e}  ln(c/product) {math.log(e['c_med'] / r['product']):+.4f} | "
          f"output never built {h['never_built']} (shuffled {h['shuffled']})")
    print(f"{'':12s} contains the 241: {full <= set(e['ids'])}; same as 143000 set: {set(e['ids']) == full}; share's first 200 inside: {ids241 <= set(e['ids'])}")
json.dump(OUT, open(os.path.join(HERE, "foundation_check.json"), "w"), indent=1)
