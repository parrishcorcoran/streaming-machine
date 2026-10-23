"""A SECOND COUNT, BY OTHER CODE, of some of the plates in the runs' JSON files. It shares nothing with engine.py: it
writes each value's places itself (offset 2^(n-1), bit by bit, packed 64 lines to a word), counts the disagreeing lines
of every (light place, plate place) with a popcount of XOR on the CPU, forms T = N - 2d, and sums |T| per place pair and
|sum over a diagonal of T| per diagonal; it checks the tip identity for every position itself. It redraws every shuffle
and fresh draw from the run's own seeds, in the run's own order of draws, so it also checks that the runs met the light
and plates they say they met.
Plates (whole: every position, every neuron, every call that the run made for them):
  Pythia-70m proj.0, proj.3: fin, blank, own, fin_sh, blank_sh, own_sh          (pythia_70m.json)
  tiny w128 L4 B, step 3000, proj.2 and mproj.1: fin, blank, push, *_sh         (tiny_w128_L4_B.json; 17 plate places)
  GPT-2 proj.0: fin, fresh, fin_sh                                               (gpt2.json)
  Qwen3-0.6B k.0: fin, fresh, fin_sh                                             (qwen.json)
    python3 tips_blank/check.py > tips_blank/check.log
"""
import os, sys, json, time, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/tips_blank"

def packed_places(V, n):
    """(n, R, words) uint64: place a of V + 2^(n-1), lines packed 64 to a word (zero padding)"""
    V = np.asarray(V, np.int64); R, N = V.shape; U = V + (1 << (n - 1)); assert U.min() >= 0 and U.max() < 1 << n
    words = (N + 63) // 64; out = np.zeros((n, R, words), np.uint64)
    for a in range(n):
        bits = ((U >> a) & 1).astype(np.uint8)
        if words * 64 > N: bits = np.concatenate([bits, np.zeros((R, words * 64 - N), np.uint8)], 1)
        out[a] = np.ascontiguousarray(np.packbits(np.ascontiguousarray(bits), axis=1, bitorder="little")).view(np.uint64)
    return out

def count(x, w, nb):
    """FR (17, nb), DS (17 + nb - 1), and positions whose pile from the tips differs from x @ w"""
    x = np.asarray(x, np.int64); w = np.asarray(w, np.int64); P, N = x.shape; M = w.shape[1]
    K = 17 + nb - 1; FR = np.zeros((17, nb), np.int64); DS = np.zeros(K, np.int64); bad = 0
    Wb = packed_places(w.T, nb)                                        # (nb, M, words): a neuron's column across the lines
    big = np.abs(x).max(1) >= 1 << 15
    for sel, n in ((np.flatnonzero(~big), 16), (np.flatnonzero(big), 17)):
        for r0 in range(0, len(sel), 16):
            rows = sel[r0:r0 + 16]
            if len(rows) == 0: continue
            Xb = packed_places(x[rows], n)                             # (n, r, words)
            Dg = np.zeros((K, len(rows), M), np.int64)
            for a in range(n):
                d = np.bitwise_count(Xb[a][:, None, None, :] ^ Wb[None, :, :, :]).sum(-1, dtype=np.int64)   # (r, nb, M)
                T = N - 2 * d
                FR[a] += np.abs(T).sum(axis=(0, 2))
                for b in range(nb): Dg[a + b] += T[:, b, :]
            DS += np.abs(Dg).sum(axis=(1, 2))
            S = sum(np.left_shift(Dg[k], k) for k in range(K))         # |Dg| <= 17 N < 2^16, k <= 32: below 2^48
            V = S - (2 * x[rows].sum(1) + N)[:, None] - (2 * w.sum(0) + N)[None, :] + N
            pile = x[rows] @ w                                          # int64 sums, below 2^43
            bad += int((V != 4 * pile).any(1).sum())
    return FR, DS, bad

def compare(tag, run, x, w, nb=16):
    t = time.time(); FR, DS, bad = count(x, w, nb)
    same = np.array_equal(FR, np.asarray(run["FR"], np.int64)) and np.array_equal(DS, np.asarray(run["DS"], np.int64))
    print(f"  {tag:34s} tips per place pair and per diagonal {'SAME' if same else 'DIFFERENT'} as the run; piles from the tips differing here: {bad}  ({time.time() - t:.0f}s)", flush=True)
    return same and bad == 0

ok = True
print("SECOND COUNT (other code, CPU popcounts)")
# Pythia-70m
f = f"{OUT}/pythia_70m.json"
if os.path.exists(f):
    sys.path.insert(0, f"{HERE}/lineblanks")
    R = json.load(open(f))["plates"]; Z = np.load(f"{HERE}/foundation0/base2_pythia-70m.npz")
    LF = np.load(f"{HERE}/lineblanks/light_70m_finished.npz"); LS = np.load(f"{HERE}/lineblanks/light_70m_start.npz")
    order = [f"{k}.{l}" for l in range(6) for k in ("qkv", "proj", "fc", "mproj")] + ["head"]
    for pl in ("proj.0", "proj.3"):
        i = order.index(pl); xf = LF[f"light.{pl}"].astype(np.int64); xo = LS[f"light.{pl}"].astype(np.int64)
        W = Z[f"{pl}.w"].astype(np.int64); S = Z[f"{pl}.s"].astype(np.int64)
        rng = np.random.default_rng([2026, 1006, 70, i]); xf_sh = rng.permuted(xf, axis=1); xo_sh = rng.permuted(xo, axis=1)
        for nm, x, w in (("fin", xf, W), ("blank", xf, S), ("own", xo, S), ("fin_sh", xf_sh, W), ("blank_sh", xf_sh, S), ("own_sh", xo_sh, S)):
            ok &= compare(f"pythia-70m {pl} {nm}", R[pl][nm], x, w)
# tiny w128 L4 B, step 3000
f = f"{OUT}/tiny_w128_L4_B.json"
if os.path.exists(f):
    BG = f"{HERE}/blankgo"; sys.path.insert(0, BG); cwd = os.getcwd(); os.chdir(BG)
    from model import data, held_windows
    from bmachine import Machine, machine_file
    R = json.load(open(f))["snaps"]["3000"]
    _, held = data(); WIN = held_windows(held)[:32]; M = machine_file(128, 4, "B", 3000); m = Machine(128, 4, "B", 3000, "finished", capture=True, M=M)
    caps = {}
    for win in WIN:
        m.cap = {}; m.forward(win[:-1], 128)
        for k, v in m.cap.items(): caps.setdefault(k, []).append(v[0][0])
    os.chdir(cwd)
    order = [f"{k}.{l}" for l in range(4) for k in ("qkv", "proj", "fc", "mproj")] + ["head"]
    for pl in ("proj.2", "mproj.1"):
        i = order.index(pl); x = np.concatenate(caps[pl]).astype(np.int64)
        W = M[pl + ".w"].astype(np.int64); S = M[pl + ".s"].astype(np.int64); Pp = W - S
        xs = np.random.default_rng([2026, 1006, ord("B"), 3000, i]).permuted(x, axis=1)
        for nm, xx, w in (("fin", x, W), ("blank", x, S), ("push", x, Pp), ("fin_sh", xs, W), ("blank_sh", xs, S), ("push_sh", xs, Pp)):
            ok &= compare(f"tiny B step 3000 {pl} {nm}", R[pl][nm], xx, w, nb=17)
# GPT-2 proj.0
f = f"{OUT}/gpt2.json"
if os.path.exists(f):
    R = json.load(open(f))["plates"]; Z = np.load(f"{HERE}/base2_gpt2.npz")
    x = np.load(f"{HERE}/groups/light_gpt2/proj.npy", mmap_mode="r")[0].astype(np.int64); W = Z["proj.0.w"].astype(np.int64)
    rng = np.random.default_rng([2026, 1006, 2, 1]); n, mm = W.shape
    sig = np.sqrt(np.maximum((W * W).mean(0) - W.mean(0) ** 2, 0)); F = np.empty(W.shape, np.int64)
    for i0 in range(0, n, 64):                                          # the run's own order of draws
        k = min(64, n - i0); F[i0:i0 + k] = np.clip(np.rint(rng.normal(0.0, 1.0, (k, mm)) * sig[None, :]), -32767, 32767)
    xs = rng.permuted(x, axis=1)
    for nm, xx, w in (("fin", x, W), ("fresh", x, F), ("fin_sh", xs, W)):
        ok &= compare(f"gpt-2 proj.0 {nm}", R["proj.0"][nm], xx, w)
# Qwen k.0
f = f"{OUT}/qwen.json"
if os.path.exists(f):
    R = json.load(open(f))["plates"]; Z = np.load(f"{HERE}/qwen/qwen_base2.npz")
    x = np.load(f"{HERE}/qwen06b/light/qkv_light.npy", mmap_mode="r")[0].astype(np.int64); W = Z["qkv.0.w"][:, 2048:3072].astype(np.int64)
    rng = np.random.default_rng([2026, 1006, 3, 1])
    sig = np.sqrt(np.maximum((W * W).mean(0) - W.mean(0) ** 2, 0))
    F = np.clip(np.rint(rng.normal(0.0, 1.0, W.shape) * sig[None, :]), -32767, 32767).astype(np.int64)
    xs = rng.permuted(x, axis=1)
    for nm, xx, w in (("fin", x, W), ("fresh", x, F), ("fin_sh", xs, W)):
        ok &= compare(f"qwen k.0 {nm}", R["k.0"][nm], xx, w)
print(f"ALL SAME: {ok}")
