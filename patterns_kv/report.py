"""The report for patterns_kv.py: reads patterns_kv/refs.npz and patterns_kv/judge_*.npz, prints every number.
Whole numbers only: shares are printed from whole-number counts (k/n and k*1000//n as a percent to one place).
    python3 patterns_kv/report.py
"""
import glob, json, numpy as np
D = "/Users/abundancemachine/streaming-machine/patterns_kv"
CMP = ["own", "M1", "M2"]; NAME = {"own": "own pattern alone", "M1": "M1 patterns+KV, middle stored", "M2": "M2 patterns+KV, nothing met", "plain": "plain machine"}
PLACES = [f"{b}.{l}" for l in range(12) for b in ("qkv", "fc")] + ["top"]
def pc(k, n):
    k, n = int(k), int(n)
    if n == 0: return "  n/a"
    q = (2000 * k + n) // (2 * n); return f"{q // 10:3d}.{q % 10}%"
def med(a):
    a = np.sort(np.asarray(a)); return int(a[(len(a) - 1) // 2]) if len(a) else -1
def mean(a):
    a = np.asarray(a); n = len(a); return (2 * int(a.sum()) + n) // (2 * n) if n else -1

def topfive(t1m, t5m, t1p, t5p):                             # (word in plain top 5, plain word in its top 5, same, set kept)
    a = (t5p == t1m[:, None]).any(1); b = (t5m == t1p[:, None]).any(1); c = t1m == t1p
    d = np.array([set(x) == set(y) for x, y in zip(t5m.tolist(), t5p.tolist())], bool)
    return a, b, c, d

# ---------------------------------------------------------------- refs
R = np.load(f"{D}/refs.npz"); CH = json.loads(str(R["checks"]))
print("PATTERNS + KV. THIS TEST CHANGES VALUES: judged by the top 5 against the plain machine at the same position.")
print("His sentence: \"There are only 50k fringe patterns. The KV shows the rotation and the angles. That's it.\"")
print("M1 and M2 are the session's construction from it (see patterns_kv.py).\n")
print(f"refs checks: position 0, every machine == plain at all 25 places: {CH['pos0']}/{CH['pos0_n']} (machine x text)")
print(f"             places fixed by definition equal to plain: {CH['defin']}/{CH['defin_n']}")
print(f"             plain machine's word == the refs continuation: {CH['words_same']}/{CH['words_n']}")
print(f"             order-free (M1 arrival, last position, keys/values in a random order; my lines == the machine's): {CH['orderfree']}/{CH['orderfree_n']} layers")
print("             every M1/M2 light == own stream + the KV's pushes == the plain-form recurrence (asserted at every layer, every text)")
pos, part = R["pos"], R["part"]; N = len(pos)

def top_table(title, sel, get, cget):
    print(f"\n{title}")
    print(f"  {'machine':34s} {'word in plain top 5':>22s} {'plain word in its top 5':>24s} {'same top word':>20s} {'top-5 set kept':>20s}")
    for m in CMP:
        a, b, c, d = get(m); n = int(sel.sum())
        print(f"  {NAME[m]:34s} " + " ".join(f"{int(x[sel].sum()):6d}/{n:<5d}{pc(x[sel].sum(), n)}" for x in (a, b, c, d)))
    for m in CMP:
        (a, b, c, d), cs = cget(m); n = int(cs.sum())
        print(f"  {'CONTROL ' + m + ' (other word)':34s} " + " ".join(f"{int(x[cs].sum()):6d}/{n:<5d}{pc(x[cs].sum(), n)}" for x in (a, b, c, d)))

t1p, t5p = R["t1_plain"], R["t5_plain"]
def rget(m): return topfive(R[f"t1_{m}"], R[f"t5_{m}"], t1p, t5p)
okc = part >= 0; pp = np.where(okc, part, 0)
def rcget(m): return topfive(R[f"t1_{m}"][pp], R[f"t5_{m}"][pp], t1p, t5p), okc & (pos > 0)
top_table(f"THE 30 REFS TEXTS, positions 1 and on ({int((pos > 0).sum())}; position 0 is the plain machine by definition)", pos > 0, rget, rcget)
def rcget_all(m): return topfive(R[f"t1_{m}"][pp], R[f"t5_{m}"][pp], t1p, t5p), okc
top_table(f"THE 30 REFS TEXTS, all {N} positions (position 0 counted)", np.ones(N, bool), rget, rcget_all)
print("\n  by position (refs), word in plain top 5:   position: own | M1 | M2")
for lo, hi in [(1, 1), (2, 3), (4, 7), (8, 15), (16, 40)]:
    s = (pos >= lo) & (pos <= hi); n = int(s.sum())
    print(f"    {lo:2d}-{hi:2d} ({n:3d}): " + " | ".join(pc(rget(m)[0][s].sum(), n) for m in CMP))

# ---------------------------------------------------------------- judge5
files = sorted(glob.glob(f"{D}/judge_*.npz"), key=lambda f: int(f.split("_")[-2]))
J = [np.load(f) for f in files]
def cat(k): return np.concatenate([j[k] for j in J]).reshape(-1, *J[0][k].shape[2:])
jt1p, jt5p = cat("t1_plain"), cat("t5_plain"); ref = cat("ref"); words = cat("words")
jch = [json.loads(str(j["checks"])) for j in J]
NW = sum(len(j["t1_plain"]) for j in J); NJ = len(jt1p)
print(f"\njudge5 files: {[f.split('/')[-1] for f in files]}  ({NW} windows, {NJ} judged positions)")
print(f"judge5 checks: plain top 5 == judge5_ref.npz: {sum(c['ref_same'] for c in jch)}/{sum(c['ref_n'] for c in jch)} windows;"
      f" places fixed by definition: {sum(c['defin'] for c in jch)}/{sum(c['defin_n'] for c in jch)}")
# the control partner: the window before, within each file; each file's first window takes its last
PJ = []; off = 0
for j in J:
    n = len(j["t1_plain"]); idx = np.arange(n); prt = np.where(idx > 0, idx - 1, n - 1) + off
    PJ.append((prt[:, None] * 20 + np.arange(20)[None, :]).ravel()); off += n
PJ = np.concatenate(PJ); jok = words.ravel() != words.ravel()[PJ]
def jget(m): return topfive(cat(f"t1_{m}"), cat(f"t5_{m}"), jt1p, jt5p)
def jcget(m): return topfive(cat(f"t1_{m}")[PJ], cat(f"t5_{m}")[PJ], jt1p, jt5p), jok
top_table(f"JUDGE5'S WINDOWS, positions 3..22 ({NJ} judged positions)", np.ones(NJ, bool), jget, jcget)
print(f"  plain machine against judge5_ref (its own top 5): {int((ref.reshape(-1, 5) == jt1p[:, None]).any(1).sum())}/{NJ}")

# ---------------------------------------------------------------- angles
def angle_table(title, get, sel, csel):
    print(f"\n{title}")
    print("  degrees between the machine's light and the plain light at the same place, after the water level, mean blocked"
          "\n  (median / mean); CONTROL = another word's same machine against this plain light")
    hdr = "  place   " + "".join(f"{m:>11s}" for m in CMP) + "".join(f"{'ctl ' + m:>11s}" for m in CMP + ["plain"])
    print(hdr + f"{'   raw x16: ':>12s}" + "".join(f"{m:>9s}" for m in CMP) + f"{'ctl M1':>9s}")
    for i, k in enumerate(PLACES):
        line = f"  {k:7s} "
        for m in CMP: a = get(f"cen_{m}")[sel, i]; line += f"{med(a):5d}/{mean(a):<5d}"
        for m in CMP + ["plain"]: a = get(f"ccen_{m}")[csel, i]; line += f"{med(a):5d}/{mean(a):<5d}"
        line += " " * 12
        for m in CMP: line += f"{med(get(f'raw_{m}')[sel, i]):9d}"
        line += f"{med(get('craw_M1')[csel, i]):9d}"
        print(line)
def rg(k): return R[k]
angle_table("ANGLES, THE 30 REFS TEXTS (positions 1 and on)", rg, pos > 0, (pos > 0) & okc)
def jg(k): return cat(k).reshape(NJ, 25)
jcs = (jg("ccen_plain")[:, 0] >= 0)
angle_table(f"ANGLES, JUDGE5'S WINDOWS ({NJ} positions; control {int(jcs.sum())} positions with a different word)", jg, np.ones(NJ, bool), jcs)
