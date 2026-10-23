"""WEIGHT-ONLY NUMBERS: what carries to wider models without the light (item 5), GPT-2 beside Qwen3-0.6B, and a proxy for
a 14B model's widths.

  A. GPT-2's plates (base2_gpt2.npz: qkv 768 x 2304, proj 768 x 768, fc 768 x 3072, mproj 3072 x 768, the table 768 x
     50,257), the same weight-only numbers as banks.py and table.py: 1s and signed digits per weight, dark weights, way
     2's gap lit digits per weight and zero gaps (each line's weights lined up across its neurons), way 1's (each
     neuron's weights lined up down its lines), and the gap law: each line's neurons drawn at random, S of them.
  B. THE PROXY FOR A 14B MODEL'S WIDTHS (5,120 hidden, 17,408 MLP; 40 query heads and 8 key/value heads of 128, so q and
     o have 5,120 neurons a line, k and v 1,024, gate and up 17,408, down 5,120). No 14B weights are here. Each line of
     the proxy is the same line of the same plate kind taken from several Qwen3-0.6B layers drawn at random, pooled, and
     S neurons drawn from the pool at random (rule: random groups only). Measured at the plate's own width too, as the
     calibration: the pooled line at the own width must give the plate's own gap numbers (within the spread of layers).
  C. THE LIGHT'S SIGNED DIGITS per lit value at every plate kind (Qwen), so the walk per pair can be written as
     (lit digits of a lit light value) x (gap lit digits per weight); checked against banks.py's measured walk per pair.

PREDICTIONS (written before the run)
  - GPT-2's weights: 1s per weight 6-7.5, signed/plain 0.72-0.80; way 2's gap lit digits per weight fall with the
    neurons a line: 768 about 1.6-1.9, 3,072 about 1.0-1.3, the table about 0.04-0.06 (from the walk per pair 9.8 / 6.4 /
    0.28 and about 5.5 lit digits per lit light value). Way 2's zero gaps at GPT-2's 768-3,072 below 10%: fp32 weights
    rarely tie; Qwen's ties (28-60% at layer 0) come from bf16.
  - The proxy: calibration within 5% of the plate's own; gap lit digits per weight at 5,120 about 0.6-0.8 (q, o, down) and
    at 17,408 about 0.3-0.5 (gate, up), most gaps zero there (65-85%).
  - The relation: walk per pair = (light's lit digits per lit value) x (gap lit digits per weight) within 3% at every
    kind (it is exact if the light's lit digits do not depend on which line they sit on; the light's own spread across
    lines makes it a little off).
    python3 qwen06b/weights_only.py
"""
import os, sys, json, time, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/qwen06b"
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def ones(A): return np.bitwise_count(np.abs(np.asarray(A, dtype=np.int64))).astype(np.int64)
def naf(A): a = np.abs(np.asarray(A, dtype=np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)

def gap_numbers(w, S=None, rng=None, blk=None):
    """each line's weights (S drawn at random, or all) lined up across its neurons: gap lit digits (first rank's whole
    weight included) per weight, zero gaps' share"""
    w = np.asarray(w); L, N = w.shape; S = N if S is None else S
    blk = blk or max(1, min(256, (4 << 20) // max(N, 1)))           # about 4M numbers a block
    lit = zero = 0
    for i0 in range(0, L, blk):
        wb = w[i0:i0 + blk].astype(np.int64)
        if S < N: wb = np.take_along_axis(wb, np.argsort(rng.random(wb.shape), axis=1)[:, :S], 1)
        ws = np.sort(wb, 1); g = np.diff(ws, axis=1)
        lit += int(naf(ws[:, 0]).sum() + naf(g).sum()); zero += int((g == 0).sum())
    return lit / (L * S), zero / (L * (S - 1))
def plate_numbers(w, rng, sizes):
    w = np.asarray(w); o1 = n1 = dk = 0
    for c in range(0, w.shape[1], 4096): o1 += int(ones(w[:, c:c + 4096]).sum()); n1 += int(naf(w[:, c:c + 4096]).sum()); dk += int((w[:, c:c + 4096] == 0).sum())
    r = dict(ones=o1 / w.size, naf=n1 / w.size, dark=dk / w.size)
    r["way2_lit"], r["way2_zero"] = gap_numbers(w)
    r["way1_lit"], r["way1_zero"] = gap_numbers(np.ascontiguousarray(w.T))
    r["law"] = {S: gap_numbers(w, S, rng) for S in sizes if S < w.shape[1]}
    return r

if __name__ == "__main__":
    t0 = time.time(); rng = np.random.default_rng([2026, 5]); res = dict(gpt2={}, proxy={}, light={})
    # ---- A. GPT-2 ----
    Mg = np.load(f"{HERE}/base2_gpt2.npz")
    for kind in ("qkv", "proj", "fc", "mproj"):
        acc = []
        for l in range(12):
            acc.append(plate_numbers(Mg[f"{kind}.{l}.w"], rng, [128, 256, 512, 768, 1024, 2048]))
        res["gpt2"][kind] = dict(shape=list(Mg[f"{kind}.0.w"].shape), **{k: float(np.mean([a[k] for a in acc])) for k in ("ones", "naf", "dark", "way2_lit", "way2_zero", "way1_lit", "way1_zero")},
                                 law={S: [float(np.mean([a["law"][S][0] for a in acc])), float(np.mean([a["law"][S][1] for a in acc]))] for S in acc[0]["law"]})
        print(f"  GPT-2 {kind:5s} {res['gpt2'][kind]['shape']}: 1s/w {res['gpt2'][kind]['ones']:.3f} signed/w {res['gpt2'][kind]['naf']:.3f} "
              f"way2 gap lit/w {res['gpt2'][kind]['way2_lit']:.4f} zero {res['gpt2'][kind]['way2_zero']:.4f} way1 {res['gpt2'][kind]['way1_lit']:.4f} ({time.time() - t0:.0f}s, {rss()} MB)", flush=True)
    hw = Mg["head.w"]; h = plate_numbers(hw, rng, [128, 512, 2048, 3072, 5120, 17408]); h["shape"] = list(hw.shape); res["gpt2"]["head"] = h; del hw
    print(f"  GPT-2 head  {h['shape']}: 1s/w {h['ones']:.3f} signed/w {h['naf']:.3f} way2 gap lit/w {h['way2_lit']:.4f} zero {h['way2_zero']:.4f} ({time.time() - t0:.0f}s, {rss()} MB)", flush=True)
    del Mg
    # ---- B. the proxy for 14B widths, from Qwen3-0.6B's plates ----
    Mq = np.load(f"{HERE}/qwen/qwen_base2.npz")
    KINDS = {"q": ("qkv", 0, 2048), "k": ("qkv", 2048, 3072), "v": ("qkv", 3072, 4096), "o": ("o", 0, 1024),
             "gate": ("gu", 0, 3072), "up": ("gu", 3072, 6144), "down": ("down", 0, 1024)}
    TARGET = {"q": 5120, "k": 1024, "v": 1024, "o": 5120, "gate": 17408, "up": 17408, "down": 5120}
    NLINES = 256
    for kind, (slab, c0, c1) in KINDS.items():
        own = c1 - c0; need = TARGET[kind]; nl = -(-need // own)                   # layers pooled to reach the target
        reps = []
        for rep in range(3):                                                      # three independent draws of layers and lines
            layers = rng.choice(28, size=max(nl, 1), replace=False)
            Ls = Mq[f"{slab}.0.w"].shape[0]; lines = rng.choice(Ls, size=NLINES, replace=False)
            pool = np.concatenate([np.asarray(Mq[f"{slab}.{l}.w"][:, c0:c1])[lines] for l in layers], axis=1)      # (NLINES, nl x own)
            r = {"layers": layers.tolist()}
            r["own"] = gap_numbers(pool, own, rng)                                 # calibration: the pooled line at the own width
            r["target"] = gap_numbers(pool, need, rng) if need <= pool.shape[1] else None
            r["own_single"] = gap_numbers(np.asarray(Mq[f"{slab}.{layers[0]}.w"][:, c0:c1])[lines])   # one layer's own lines, all neurons
            reps.append(r)
        res["proxy"][kind] = dict(own_width=own, target=need, layers_pooled=nl, reps=reps)
        print(f"  proxy {kind:5s}: own {own} -> 14B {need} ({nl} layers pooled): own-width gap lit/w single layer "
              + ", ".join(f"{r['own_single'][0]:.4f}" for r in reps) + "; pooled at own width " + ", ".join(f"{r['own'][0]:.4f}" for r in reps)
              + "; at the 14B width " + ", ".join(f"{r['target'][0]:.4f} (zero {r['target'][1]:.3f})" for r in reps) + f" ({time.time() - t0:.0f}s, {rss()} MB)", flush=True)
    # ---- C. the light's signed digits per lit value (Qwen), every kind ----
    for slab in ("qkv", "o", "gu", "down"):
        Lt = np.load(f"{OUT}/light/{slab}_light.npy", mmap_mode="r"); lit_vals = digits = ones_ = 0
        for l in range(28):
            x = np.asarray(Lt[l]).astype(np.int64); lit_vals += int((x != 0).sum()); digits += int(naf(x).sum()); ones_ += int(ones(x).sum())
        res["light"][slab] = dict(lit_values=lit_vals, signed_per_lit=digits / lit_vals, ones_per_lit=ones_ / lit_vals,
                                  values=int(np.prod(Lt.shape)))
        print(f"  Qwen light at {slab:5s}: lit values {lit_vals / np.prod(Lt.shape) * 100:.2f}%; per lit value: 1s {ones_ / lit_vals:.3f}, signed digits {digits / lit_vals:.3f}", flush=True)
    json.dump(res, open(f"{OUT}/weights_only.json", "w"))
    print(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")
