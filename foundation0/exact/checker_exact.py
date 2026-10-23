"""Independent rebuild of remove_exact.py and store_exact.py, by a separate path. 2026-10-06.

It shares no machine code with remove_exact.py: it imports ../pythia_base2.py itself (the machine as it stands, file
unchanged), lets it build its own finished plates, and swaps in the shaved plates computed here a different way:
round(v x W0) by divmod (quotient, and one more where twice the remainder reaches 2^30), not by add-half-and-shift.
  1. cases fin, k, p: every head pile at every position of the 30 texts (each text's words, then the finished machine's
     16 greedy words fed in), sha256 per text, against remove_exact_<size>.json; top-1 against the finished machine
     counted again here.
  2. weights changed on the grid, per kind, counted again here, against remove_exact_<size>.json.
  3. lit places by digit-by-digit signed writing (a loop that takes off one digit at a time, not popcount(x ^ 3x)), and
     each neuron's spread by np.std, for W, Dk and Dp per kind, against store_exact_<size>.json.

    python3 checker_exact.py size=14m
"""
import os, sys, json, math, time, hashlib, importlib.util, resource
import numpy as np

A = dict(a.split("=") for a in sys.argv[1:]); SIZE = A.get("size", "14m")
HERE = os.path.dirname(os.path.abspath(__file__)); UP = os.path.dirname(HERE)
sys.argv = ["pythia_base2.py", f"size={SIZE}"]                          # the machine reads its switches at import
spec = importlib.util.spec_from_file_location("pb2", os.path.join(UP, "pythia_base2.py")); pb2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pb2)
M = pb2.M; L = pb2.L
CHK = json.load(open(os.path.join(UP, "foundation_check.json")))[f"{SIZE}@143000"]
V = {"k": int(round(CHK["embed"]["c_med"] * 2 ** 30)), "p": int(round(CHK["product"] * 2 ** 30))}
RX = json.load(open(os.path.join(HERE, f"remove_exact_{SIZE}.json")))["cases"]
SX = json.load(open(os.path.join(HERE, f"store_exact_{SIZE}.json")))["kinds"]
LOGF = open(os.path.join(HERE, f"checker_exact_{SIZE}.log"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
def mem(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30

def shave(v, w0):                                                        # round(v x w0 / 2^30), half up, by divmod
    q, r = np.divmod(v * w0, 1 << 30)
    return q + (2 * r >= (1 << 30))

def lit_loop(x):                                                         # signed digits, one at a time
    a = np.abs(x).astype(np.int64); n = np.zeros(a.shape, np.int64)
    while a.any():
        odd = a & 1; dgt = np.where(odd == 1, 2 - (a & 3), 0); n += odd; a = (a - dgt) >> 1
    return n

def hb(sig):                                                             # bits of a bell curve of spread sig, whole numbers
    out = np.zeros_like(sig)
    for i, s in enumerate(sig):
        if s >= 2: out[i] = math.log2(s * math.sqrt(2 * math.pi * math.e))
        elif s > 0:
            k = np.arange(-int(14 * s) - 3, int(14 * s) + 4)
            p = np.array([0.5 * (math.erf((kk + .5) / (s * math.sqrt(2))) - math.erf((kk - .5) / (s * math.sqrt(2)))) for kk in k]); p = p[p > 0]
            out[i] = float(-(p * np.log2(p)).sum())
    return out

def shaved(name, v, dtype):                                            # W - round(v x W0), a block of rows at a time
    W, S = M[name + ".w"], M[name + ".s"]; out = np.empty(W.shape, dtype); n = 0; B = max(1, (1 << 20) // W.shape[1])
    for a in range(0, W.shape[0], B):
        r = shave(v, S[a:a + B].astype(np.int64)); out[a:a + B] = W[a:a + B].astype(np.int64) - r; n += int((r != 0).sum())
    return out, n

names = [f"{nm}.{l}" for l in range(L) for nm in ("qkv", "proj", "fc", "mproj")] + ["head", "embed"]
refs = json.load(open(os.path.join(UP, f"refs_pythia-{SIZE}.json")))
log(f"checker_exact.py  size {SIZE}  v30 {V}")
fin_top = None; ok_all = True
for case in ("fin", "k", "p"):                                           # fin: the machine's own finished plates, as it built them
    chg = {}
    if case != "fin":
        for name in names:
            kd = name.split(".")[0]
            if name == "embed": new, n = shaved(name, V[case], np.int32); pb2.EMB_F = new
            else:
                _, e_, b_, _ = pb2.PL[name]; pb2.PL[name] = None
                new, n = shaved(name, V[case], np.float64); pb2.PL[name] = (new, e_, b_, float(np.abs(new).max()))
            chg[kd] = chg.get(kd, 0) + n; del new
    shas, tops = [], []
    for r in refs:
        seq = r["ids"] + r["ref"][:15]
        got, lg = pb2.forward(seq, last=len(seq))
        shas.append(hashlib.sha256(np.ascontiguousarray(lg).tobytes()).hexdigest()); tops.append(got)
    same = sum(a == b for a, b in zip(shas, RX[case]["sha_all"]))
    if case == "fin": fin_top = tops; t1 = None
    else: t1 = sum(a == b for x, y in zip(tops, fin_top) for a, b in zip(x, y)) / sum(len(x) for x in tops)
    log(f"  {case}: head piles at every position identical (sha256) to remove_exact.py's in {same} of {len(refs)} texts"
        + ("" if t1 is None else f";  top-1 against the finished machine {t1 * 100:.2f}% (remove_exact.py {RX[case]['top1_all'] * 100:.2f}%)")
        + f"   peak {mem():.2f} GB")
    ok_all &= same == len(refs) and (t1 is None or abs(t1 - RX[case]["top1_all"]) < 1e-12)
    if case != "fin":
        for kd, n in chg.items():
            m = RX[case]["changed"][kd]["changed"]; ok_all &= n == m
            log(f"     {kd:6s} weights changed: {n:,} here, {m:,} in remove_exact.py  {'same' if n == m else 'DIFFERENT'}")
pb2.PL.clear()
# 3. storage, a separate count
log("  storage (lit places a weight; bits a weight at the floor), here vs store_exact.py:")
acc = {}
for name in names:
    kd = name.split(".")[0]
    W, S = M[name + ".w"], M[name + ".s"]
    if name != "embed": W, S = W.T, S.T                                  # neurons as rows
    B = max(1, (1 << 20) // W.shape[1])
    for a in range(0, W.shape[0], B):
        w = np.ascontiguousarray(W[a:a + B]).astype(np.int64); w0 = np.ascontiguousarray(S[a:a + B]).astype(np.int64)
        for f, X in (("W", w), ("Dk", w - shave(V["k"], w0)), ("Dp", w - shave(V["p"], w0))):
            lit = int(lit_loop(X).sum()); bits = float(hb(np.asarray(X, np.float64).std(1)).sum()) * X.shape[1]
            q = acc.setdefault((kd, f), [0, 0.0, 0]); q[0] += lit; q[1] += bits; q[2] += X.size
    del W, S
for kd in ["qkv", "proj", "fc", "mproj", "head", "embed"]:
    parts = []
    for f in ("W", "Dk", "Dp"):
        lit, bits, n = acc[(kd, f)]; l2, b2 = SX[kd][f]["lit"], SX[kd][f]["bits"]
        ok = abs(lit / n - l2) < 1e-12 and abs(bits / n - b2) < 1e-9; ok_all &= ok
        parts.append(f"{f} lit {lit / n:.6f}/{l2:.6f} bits {bits / n:.7f}/{b2:.7f} {'same' if ok else 'DIFFERENT'}")
    log(f"     {kd:6s} " + "   ".join(parts))
log(f"ALL CHECKS {'PASS' if ok_all else 'FAIL'}   peak memory {mem():.2f} GB")
