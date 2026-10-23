"""One row of Qwen3-0.6B (bf16): real 8-bit and 4-bit (llama.cpp Q8_0 / Q4_0), each weight on the circle,
then the bf16 weights as base-2 circles (ups and holes, no two side by side) keeping each weight's top n circles."""
import glob, math, json
import numpy as np
from safetensors import safe_open

F = glob.glob("/Users/abundancemachine/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/model.safetensors")[0]
NAME = "model.layers.0.mlp.up_proj.weight"
with safe_open(F, framework="pt") as f:
    t = f.get_slice(NAME)
    print(NAME, t.get_shape(), t.get_dtype())
    row = t[0:1].float().numpy()[0].astype(np.float32)   # bf16 -> f32 is exact
N = row.size
w = row.astype(np.float64)

def roundf(x):  # C roundf: half away from zero
    return np.sign(x) * np.floor(np.abs(x) + np.float32(0.5))

def q8_0(x):
    out = np.empty_like(x); codes = np.empty(x.size, np.int64)
    for b in range(0, x.size, 32):
        v = x[b:b+32]
        amax = np.max(np.abs(v)); d = np.float32(amax / np.float32(127))
        idd = np.float32(1) / d if d else np.float32(0)
        q = roundf(v * idd).astype(np.int64)
        dh = np.float32(np.float16(d))
        codes[b:b+32] = q; out[b:b+32] = q.astype(np.float32) * dh
    return out, codes

def q4_0(x):
    out = np.empty_like(x); codes = np.empty(x.size, np.int64)
    for b in range(0, x.size, 32):
        v = x[b:b+32]
        j = int(np.argmax(np.abs(v)))          # llama.cpp: the value with the largest size, sign kept
        mx = v[j]; d = np.float32(mx / np.float32(-8))
        idd = np.float32(1) / d if d else np.float32(0)
        xi = np.minimum(15, np.trunc(v * idd + np.float32(8.5)).astype(np.int64))
        dh = np.float32(np.float16(d))
        codes[b:b+32] = xi; out[b:b+32] = (xi - 8).astype(np.float32) * dh
    return out, codes

q8, c8 = q8_0(row); q4, c4 = q4_0(row)

# --- base-2 circles of each bf16 weight: |w| = M * 2^p, M whole (8 significant places), written as ups/holes
def naf(M):
    d = []
    while M:
        if M & 1:
            s = 2 - (M & 3); M -= s
        else:
            s = 0
        d.append(s); M >>= 1
    return d  # d[k] is the digit at place k

circ = []   # per weight: list of (place, sign) from the top down
for v in w:
    if v == 0: circ.append([]); continue
    fr, e = math.frexp(abs(v)); M = int(fr * 256); assert M * 2.0 ** (e - 8) == abs(v)
    sg = 1 if v > 0 else -1
    d = naf(M)
    circ.append([(k + e - 8, sg * d[k]) for k in range(len(d) - 1, -1, -1) if d[k]])
for v, c in zip(w, circ):
    assert sum(s * 2.0 ** p for p, s in c) == v

def topn(n):
    return np.array([sum(s * 2.0 ** p for p, s in c[:n]) for c in circ])

def H(vals):
    _, cnt = np.unique(np.asarray(vals), return_counts=True); p = cnt / cnt.sum()
    return float(-(p * np.log2(p)).sum())

def bitsw(n):
    """bits a weight to write the kept circles exactly: count, sign + drop from the row's top place, sign + gap."""
    kept = [c[:n] for c in circ]
    T = max(c[0][0] for c in kept if c)
    cnts = [len(c) for c in kept]
    drops = [T - c[0][0] for c in kept if c]
    gaps = [c[i-1][0] - c[i][0] - 2 for c in kept for i in range(1, len(c))]
    signs = [s for c in kept for _, s in c]
    width = lambda xs: math.ceil(math.log2(max(xs) + 1)) if xs and max(xs) > 0 else 0
    cw = math.ceil(math.log2(len(set(cnts)))) if len(set(cnts)) > 1 else 0
    fixed = 8 + N * cw + len(drops) * width(drops) + len(gaps) * width(gaps) + len(signs)
    floor = 8 + N * H(cnts) + len(drops) * H(drops) + (len(gaps) * H(gaps) if gaps else 0) + len(signs) * H(signs)
    return fixed / N, floor / N, width(drops), width(gaps)

# --- the circle: R = the row's largest weight size. Each weight sits at angle arccos(w/R).
R = np.max(np.abs(w))
def ang(x):
    c = x / R; past = int(np.sum(np.abs(c) > 1))
    return np.degrees(np.arccos(np.clip(c, -1, 1))), past
th0, _ = ang(w)
def hand(x):  # whole row as one hand: angle between it and the bf16 hand
    c = np.dot(w, x) / (np.linalg.norm(w) * np.linalg.norm(x)); return math.degrees(math.acos(min(1.0, c)))

rows = []
def report(label, x, bits_fixed, bits_floor):
    th, past = ang(x); dth = np.abs(th - th0)
    rows.append(dict(label=label, bits_fixed=round(bits_fixed, 3), bits_floor=round(bits_floor, 3),
                     hand_deg=round(hand(x), 4), mean_deg=round(float(dth.mean()), 4),
                     max_deg=round(float(dth.max()), 4), past_edge=past))
    print(f"{label:16s} bits {bits_fixed:6.3f} (floor {bits_floor:6.3f})  hand {hand(x):8.4f} deg   "
          f"each weight: mean {dth.mean():7.4f} deg, largest {dth.max():8.4f} deg   past the edge {past}")

print(f"\nrow: {N} weights, largest size R = {R:.6g}; bf16 = 16 bits a weight\n")
report("Q8_0 (8-bit)", q8, 8.5, H(c8) + 0.5)
report("Q4_0 (4-bit)", q4, 4.5, H(c4) + 0.5)
ncirc = [len(c) for c in circ]
print(f"\nbf16 weights as base-2 circles: {np.mean(ncirc):.2f} lit circles a weight on average (most {max(ncirc)})")
for n in range(1, 6):
    bf, bl, wd, wg = bitsw(n)
    report(f"top {n} circle(s)", topn(n), bf, bl)

with open("/Users/abundancemachine/streaming-machine/circles_exercise/row_weights.tsv", "w") as f:
    f.write("i\tbf16\tq8\tq4\tq4_code\tcircles(place:sign)\ttop1\ttop2\ttop3\tdeg_bf16\tdeg_q8\tdeg_q4\tdeg_top1\tdeg_top2\tdeg_top3\n")
    t1, t2, t3 = topn(1), topn(2), topn(3)
    a8, _ = ang(q8.astype(np.float64)); a4, _ = ang(q4.astype(np.float64))
    a1, _ = ang(t1); a2, _ = ang(t2); a3, _ = ang(t3)
    for i in range(N):
        cs = " ".join(f"{p}:{'+' if s > 0 else '-'}" for p, s in circ[i])
        f.write(f"{i}\t{w[i]:.9g}\t{q8[i]:.9g}\t{q4[i]:.9g}\t{c4[i]-8}\t{cs}\t{t1[i]:.9g}\t{t2[i]:.9g}\t{t3[i]:.9g}\t"
                f"{th0[i]:.4f}\t{a8[i]:.4f}\t{a4[i]:.4f}\t{a1[i]:.4f}\t{a2[i]:.4f}\t{a3[i]:.4f}\n")
json.dump(rows, open("/Users/abundancemachine/streaming-machine/circles_exercise/results.json", "w"), indent=1)

# --- remove the lower circles across the whole row: every circle below place F is removed
def cut(F):
    return [[(p, s) for p, s in c if p >= F] for c in circ]
def bits_kept(kept):
    T = max(c[0][0] for c in kept if c)
    cnts = [len(c) for c in kept]
    drops = [T - c[0][0] for c in kept if c]
    gaps = [c[i-1][0] - c[i][0] - 2 for c in kept for i in range(1, len(c))]
    signs = [s for c in kept for _, s in c]
    width = lambda xs: math.ceil(math.log2(max(xs) + 1)) if xs and max(xs) > 0 else 0
    cw = math.ceil(math.log2(max(cnts) + 1))
    fixed = 8 + N * cw + len(drops) * width(drops) + len(gaps) * width(gaps) + len(signs)
    floor = 8 + N * H(cnts) + len(drops) * H(drops) + (len(gaps) * H(gaps) if gaps else 0) + len(signs) * H(signs)
    return fixed / N, floor / N
print("\nlower circles removed across the row (keep places >= F; the row's top place is", max(c[0][0] for c in circ if c), ")")
for F in range(-5, -13, -1):
    kept = cut(F)
    x = np.array([sum(s * 2.0 ** p for p, s in c) for c in kept])
    bf, bl = bits_kept(kept)
    report(f"keep >= 2^{F}", x, bf, bl)
    print(f"{'':16s} circles kept a weight {np.mean([len(c) for c in kept]):.2f}; weights with none {sum(1 for c in kept if not c)}")
json.dump(rows, open("/Users/abundancemachine/streaming-machine/circles_exercise/results.json", "w"), indent=1)
