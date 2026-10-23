"""Independent rebuild of zerosum_count.py's main numbers by a separate road.  2026-10-06.

It does not read the whole-number npz files and shares no code with zerosum_count.py:
  - Pythia's own fp16 plates read with torch (safetensors' torch loader), in Pythia's own [out][in] layout
  - every number in float64 at fp16's own grain (no shared grid, no shifting)
  - D = W - W0 and D' = W - c W0 with c the float c_med of foundation_check.json (no rounding of c W0)
  - write lines: attention.dense and mlp.dense_4h_to_h one COLUMN over the lanes (dim 0), embed_in one ROW (dim 1);
    read neurons: query_key_value and mlp.dense_h_to_4h one ROW over the lanes (dim 1), embed_out one ROW (dim 1)
  - z of a line = its sum / sqrt(its sum of squares); beyond 3 decided as sum^2 > 9 x sum of squares
Compares with zerosum_<size>.json and zerosum_spans70.json: the root mean square of z over lines / neurons (D, D',
W, W0, every span raw and pushes-only), the share beyond 3, each plate's net z, and item 1's shares and medians.

PREDICTION (written before this checker first ran; the first run peaked at 3.65 GB, so item 1 is now
built one change at a time in preallocated arrays, the numbers unchanged): the two roads differ only by the grids' rounding (fp16 keeps 11
places, the npz 15 places per neuron), so: RMS z within 2% (relative) or 0.01 on every plate and span; share beyond
3 within 0.5 points; plate net z within 0.1 or 2%; item 1's shares within 1 point (ties |D| = |W0| fall
differently on the two grids), medians within 2%.

    python3 zerosum_checker.py sizes=14m,70m,160m [spans=1] [tag=all]   (one size per process keeps each under 3 GB)
"""
import os, sys, json, math
import torch
from safetensors import safe_open

A = dict(a.split("=") for a in sys.argv[1:])
SIZES = [x for x in A.get("sizes", "14m,70m,160m").split(",") if x]; SPANS = A.get("spans", "1") == "1"; TAG = A.get("tag", "all")
HERE = os.path.dirname(os.path.abspath(__file__)); F0 = os.path.join(os.path.dirname(HERE), "foundation0"); MD = os.path.join(F0, "models")
CHK = json.load(open(os.path.join(F0, "foundation_check.json")))
torch.set_grad_enabled(False)
f64 = torch.float64


def cval(size, step): return 1.0 if step == 0 else CHK[f"{size}@{step}"]["embed"]["c_med"]


def plate_list(L):
    out = []
    for l in range(L):
        p = f"gpt_neox.layers.{l}."
        out += [(f"qkv.{l}", p + "attention.query_key_value.weight", "r", 1), (f"proj.{l}", p + "attention.dense.weight", "w", 0),
                (f"fc.{l}", p + "mlp.dense_h_to_4h.weight", "r", 1), (f"mproj.{l}", p + "mlp.dense_4h_to_h.weight", "w", 0)]
    return out + [("head", "embed_out.weight", "r", 1), ("embed", "gpt_neox.embed_in.weight", "w", 1)]


class Acc:
    """unit sums over the lanes, accumulated over row blocks (the unit dim is fixed per plate)"""
    def __init__(self): self.n, self.q = [], []
    def add(self, x, dim): self.n.append(x.sum(dim)); self.q.append((x * x).sum(dim))
    def stats(self, dim):
        n = torch.cat(self.n) if dim == 1 else torch.stack(self.n).sum(0)
        q = torch.cat(self.q) if dim == 1 else torch.stack(self.q).sum(0)
        ok = q > 0; z = n[ok] / q[ok].sqrt()
        return dict(rms=float(z.pow(2).mean().sqrt()), over3=float((n[ok] ** 2 > 9 * q[ok]).double().mean()),
                    plate_z=float(n.sum() / q.sum().sqrt()))


def tensors(f, key, rows=None):
    sl = f.get_slice(key)
    if rows is None: return sl[:].to(f64)
    return sl[rows[0]:rows[1]].to(f64)


def blocks(shape, dim):
    """the plate in row blocks when the unit is a row (dim 1) and the plate is a word table; else whole"""
    R = shape[0]
    if dim == 1 and R > 8192: return [(a, min(R, a + 8192)) for a in range(0, R, 8192)]
    return [None]


def size_item1(w, w0, d, acc):
    aD, a0, aW = d.abs(), w0.abs(), w.abs(); o, m = acc["at"], d.numel()
    acc["n"] += m; acc["d_lt_blank"] += int((aD < a0).sum()); acc["within20"] += int((5 * aD <= a0).sum())
    acc["d_lt_fin"] += int((aD < aW).sum()); acc["sign_kept"] += int(((w * w0) > 0).sum())
    acc["r0"][o:o + m] = (aD / a0).float().flatten(); acc["rW"][o:o + m] = (aD / aW).float().flatten(); acc["at"] += m


def finish_item1(acc):
    n = acc["n"]; r0, rW = acc["r0"], acc["rW"]
    res = dict(d_lt_blank=acc["d_lt_blank"] / n, within20=acc["within20"] / n, d_lt_fin=acc["d_lt_fin"] / n, sign_kept=acc["sign_kept"] / n,
               med_d_over_blank=float(r0[~r0.isnan()].median()), med_d_over_fin=float(rW[~rW.isnan()].median()))   # 0/0 left out
    del acc["r0"], acc["rW"]
    return res


def run_size(size):
    """one change at a time for item 1, so the 160m word tables stay well under 3 GB"""
    f1 = safe_open(os.path.join(MD, f"pythia-{size}-step143000", "model.safetensors"), "pt")
    f0 = safe_open(os.path.join(MD, f"pythia-{size}-step0", "model.safetensors"), "pt")
    L = json.load(open(os.path.join(MD, f"pythia-{size}-step143000", "config.json")))["num_hidden_layers"]
    c = cval(size, 143000); out = {}
    for name, key, kind, dim in plate_list(L):
        shape = f1.get_slice(key).get_shape(); numel = shape[0] * shape[1]
        accs = {k: Acc() for k in ("D", "Dp", "W", "W0")}; item1 = {}
        for ch in ("D", "Dp"):
            it = dict(n=0, d_lt_blank=0, within20=0, d_lt_fin=0, sign_kept=0, at=0,
                      r0=torch.empty(numel, dtype=torch.float32), rW=torch.empty(numel, dtype=torch.float32))
            for rows in blocks(shape, dim):
                w, w0 = tensors(f1, key, rows), tensors(f0, key, rows)
                d = w - w0 if ch == "D" else w - c * w0
                size_item1(w, w0, d, it)
                if ch == "D":
                    for k, x in (("D", d), ("Dp", w - c * w0), ("W", w), ("W0", w0)): accs[k].add(x, dim)
                del w, w0, d
            item1[ch] = finish_item1(it); del it
        out[name] = dict(kind=kind, **{k: a.stats(dim) for k, a in accs.items()}, item1=item1)
    return out


def run_spans():
    STEPS = (0, 1000, 10000, 50000, 143000)
    F = {t: safe_open(os.path.join(MD, f"pythia-70m-step{t}", "model.safetensors"), "pt") for t in STEPS}
    C = {t: cval("70m", t) for t in STEPS}
    SP = [(0, 1000), (1000, 10000), (10000, 50000), (50000, 143000), (0, 10000), (0, 50000), (0, 143000)]
    out = {}
    for name, key, kind, dim in plate_list(6):
        acc = {f"{t1}->{t2}": (Acc(), Acc()) for t1, t2 in SP}
        for rows in blocks(F[0].get_slice(key).get_shape(), dim):
            W = {t: tensors(F[t], key, rows) for t in STEPS}
            for t1, t2 in SP:
                a, b = acc[f"{t1}->{t2}"]; a.add(W[t2] - W[t1], dim); b.add(W[t2] - (C[t2] / C[t1]) * W[t1], dim)
            del W
        out[name] = {k: dict(raw=a.stats(dim), push=b.stats(dim)) for k, (a, b) in acc.items()}
    return out


def cmp(lab, mine, theirs, rel, absol):
    dlt = abs(mine - theirs); ok = dlt <= max(absol, rel * abs(theirs))
    return ok, f"{lab} {mine:.4f}/{theirs:.4f}"


report = []; worst = {}
def note(key, ok):
    worst.setdefault(key, [0, 0]); worst[key][0] += 1; worst[key][1] += 0 if ok else 1

for size in SIZES:
    M = json.load(open(os.path.join(HERE, f"zerosum_{size}.json")))["plates"]
    R = run_size(size)
    for name, r in R.items():
        m = M[name]; bad = []
        for k, mk in (("D", "D"), ("Dp", "Dp"), ("W", "W"), ("W0", "W0")):
            ok, s = cmp(f"{k} rms", r[k]["rms"], m[mk]["real"]["units"]["rms"], 0.02, 0.01); note("rms z", ok); bad += [] if ok else [s]
            ok, s = cmp(f"{k} >3", r[k]["over3"], m[mk]["real"]["units"]["over3"], 0, 0.005); note("share beyond 3", ok); bad += [] if ok else [s]
            ok, s = cmp(f"{k} plate z", r[k]["plate_z"], m[mk]["real"]["plate"]["net_z"], 0.02, 0.1); note("plate z", ok); bad += [] if ok else [s]
        for k in ("D", "Dp"):
            for f in ("d_lt_blank", "within20", "d_lt_fin", "sign_kept"):
                ok, s = cmp(f"{k} {f}", r["item1"][k][f], m["size"][k][f], 0, 0.01); note("item-1 shares", ok); bad += [] if ok else [s]
            for f in ("med_d_over_blank", "med_d_over_fin"):
                ok, s = cmp(f"{k} {f}", r["item1"][k][f], m["size"][k][f], 0.02, 0.0); note("item-1 medians", ok); bad += [] if ok else [s]
        line = (f"{size:4s} {name:9s} rms z D {r['D']['rms']:.3f}/{m['D']['real']['units']['rms']:.3f} D' {r['Dp']['rms']:.3f}/{m['Dp']['real']['units']['rms']:.3f} "
                f"W {r['W']['rms']:.3f}/{m['W']['real']['units']['rms']:.3f} W0 {r['W0']['rms']:.3f}/{m['W0']['real']['units']['rms']:.3f} | "
                f"10->12 {100 * r['item1']['D']['within20']:.2f}/{100 * m['size']['D']['within20']:.2f} |D|<|W0| {100 * r['item1']['D']['d_lt_blank']:.1f}/{100 * m['size']['D']['d_lt_blank']:.1f} "
                f"| {'agree' if not bad else 'DIFFER: ' + '; '.join(bad)}")
        print(line, flush=True); report.append(line)
if SPANS:
    M = json.load(open(os.path.join(HERE, "zerosum_spans70.json")))["plates"]
    R = run_spans()
    for name, r in R.items():
        bad = []; parts = []
        for k, v in r.items():
            for lab in ("raw", "push"):
                ok, s = cmp(f"{k} {lab}", v[lab]["rms"], M[name]["spans"][k][lab]["units"]["rms"], 0.02, 0.01); note("span rms z", ok); bad += [] if ok else [s]
            parts.append(f"{k} {v['raw']['rms']:.3f}/{M[name]['spans'][k]['raw']['units']['rms']:.3f} {v['push']['rms']:.3f}/{M[name]['spans'][k]['push']['units']['rms']:.3f}")
        line = f"70m spans {name:9s} " + " | ".join(parts[:4]) + f" | {'agree' if not bad else 'DIFFER: ' + '; '.join(bad)}"
        print(line, flush=True); report.append(line)
summary = "checks (count, outside tolerance): " + ", ".join(f"{k} {v[0]} / {v[1]}" for k, v in worst.items())
print(summary); report.append(summary)
open(os.path.join(HERE, f"zerosum_checker_{TAG}.log"), "w").write("\n".join(report) + "\n")
