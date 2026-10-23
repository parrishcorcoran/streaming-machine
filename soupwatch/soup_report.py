"""Add up everything soup_watch.py recorded and print it. Whole numbers only: every share is printed as count/count,
with a whole-number percent (100 x a // b) beside it for reading.

    python3 soupwatch/soup_report.py
"""
import os, glob, numpy as np
D = "/Users/abundancemachine/streaming-machine/soupwatch/"
files = sorted(glob.glob(D + "out_real/text_*.npz"))
X = [dict(np.load(f)) for f in files]
BANKS = [f"{k}.{l}" for l in range(12) for k in ("qkv", "proj", "fc", "mproj")] + ["head"]
KIND = {k: [i for i, n in enumerate(BANKS) if n.split(".")[0] == k] for k in ("qkv", "proj", "fc", "mproj", "head")}
def sh(a, b): a, b = int(a), int(b); return f"{a:,}/{b:,} ({100 * a // b if b else 0}%)"
def pc(a, b): a, b = int(a), int(b); return f"{100 * a // b if b else 0:3d}%"

# ---------------------------------------------------------------- control (a): every line and neuron relabelled
print("CONTROL (a): every line and neuron in a random order (seed 11), moved through the whole machine")
same = tot = 0
for f in files:
    g = f.replace("out_real", "out_lines")
    if not os.path.exists(g): continue
    a, b = np.load(f), np.load(g)
    for k in a.files: tot += 1; same += int(np.array_equal(a[k], b[k]))
print(f"  arrays identical, real vs relabelled: {same}/{tot}\n")

# ---------------------------------------------------------------- W1
print("W1  THE WAVES")
words = waves_settle = waves_run = free = at_front = slots = agree = trans = 0
per_wave = {}
for x in X:
    T, S, ow = x["T"], int(x["S"]), list(x["ow_words"]); nw = len(T) - 1; N = len(ow); fin = T[-1, S:]
    settle = []
    for j in range(N):
        v = nw
        while v > 0 and T[v - 1, S + j] == fin[j]: v -= 1
        settle.append(v)                                   # the wave after which slot j holds its final word for good
    words += N; waves_settle += max(settle); waves_run += nw
    free += sum(settle[j] <= settle[j - 1] for j in range(1, N)); at_front += sum(settle[j] == settle[j - 1] + 1 for j in range(1, N)); slots += N - 1
    for w in range(1, nw + 1):                             # wave w pours T[w-1], reads T[w]
        unsettled = [j for j in range(N) if any(T[v, S + j] != fin[j] for v in range(w - 1, nw + 1))]
        changed = [j for j in range(N) if T[w, S + j] != T[w - 1, S + j]]
        e = per_wave.setdefault(w, [0, 0, 0, 0, 0])
        e[0] += 1; e[1] += sum(int(T[w, S + j] == ow[j]) for j in range(N)); e[2] += len(changed)
        e[3] += (unsettled[0] if unsettled else N); e[4] += sum(1 for j in changed if j > (unsettled[0] if unsettled else N))
        if changed: trans += 1; agree += int(unsettled and changed[0] == unsettled[0])
print(f"  {len(X)} texts, {words} words: all settled after {waves_settle} waves ({words}/{waves_settle} words a wave); {waves_run} waves run (the last one reads no change)")
print(f"  words settled no later than the word before: {sh(free, slots)}; settled exactly one wave after the word before (when the front reached them): {sh(at_front, slots)}")
print(f"  the first slot to change in a wave is the first unsettled slot: {agree}/{trans} waves")
print("  wave | texts | words right after it (sum) | slots whose word changed (sum) | front slot (sum) | changed slots ahead of the front")
for w in sorted(per_wave):
    e = per_wave[w]; print(f"  {w:4d} | {e[0]:5d} | {e[1]:4d}/{16 * e[0]:4d} | {e[2]:4d} | {e[3]:4d} | {e[4]:4d}")
x = X[0]; T, S = x["T"], int(x["S"])
print(f"  text 0, the words entering each wave (slots 0..15):")
for w in range(len(T)): print("   ", " ".join(f"{int(v):5d}" for v in T[w, S:]))

# ---------------------------------------------------------------- W2
print("\nW2  THE CHANGE d = R(w+1) - R(w), per snapshot (0 = entry, 2l+1 after layer l's arrival write, 2l+2 after its bend write)")
REG = ("behind", "front", "ahead, own word changed", "ahead, own word same")
acc = {r: {k: np.zeros(25, np.int64) for k in "pos zero nzv phd phn fl sf nzv_a phd_a fl_a nzv_b phd_b fl_b".split()} for r in REG}
dist = {}
for x in X:
    T = x["T"]; nt = len(x["nzv"])
    for k in range(nt):                                    # transition between wave k+1 and wave k+2
        a, b = T[k], T[k + 1]; ch = np.flatnonzero(a != b); f = ch[0]
        for p in range(len(a)):
            r = "behind" if p < f else "front" if p == f else ("ahead, own word changed" if a[p] != b[p] else "ahead, own word same")
            A = acc[r]; A["pos"] += 1; A["zero"] += (x["nzv"][k, :, p] == 0)
            for key in "nzv phd phn fl sf nzv_a phd_a fl_a nzv_b phd_b fl_b".split(): A[key] += x[key][k, :, p]
            if p > f:
                e = dist.setdefault(int(p - f), np.zeros((4, 25), np.int64))
                e[0] += x["phd"][k, :, p]; e[1] += x["phn"][k, :, p]; e[2] += x["nzv"][k, :, p]; e[3] += 768
for r in REG:
    A = acc[r]; n = int(A["pos"][0])
    print(f"\n  {r}: {n} position-waves")
    print("  snap | d exactly 0 | values moved     | photons of d / of R(w+1) | flips / photons of R(w+1) | sign flips | SHUFFLE A: values moved, photons of d / R(w+1), flips | SHUFFLE B (no position 0): photons of d / R(w+1)")
    for s in range(25):
        print(f"  {s:4d} | {int(A['zero'][s]):4d}/{n:<4d}   | {pc(A['nzv'][s], 768 * n)} {int(A['nzv'][s]):>9,} | {sh(A['phd'][s], A['phn'][s]):>30s} | {pc(A['fl'][s], A['phn'][s])} | {int(A['sf'][s]):>7,} |"
              f" {pc(A['nzv_a'][s], 768 * n)}  {sh(A['phd_a'][s], A['phn'][s]):>30s}  {pc(A['fl_a'][s], A['phn'][s])} | {sh(A['phd_b'][s], A['phn'][s])}")
    print(f"  all 25 | photons of d / of R(w+1): {sh(A['phd'].sum(), A['phn'].sum())}; shuffle A: {sh(A['phd_a'].sum(), A['phn'].sum())}; shuffle B: {sh(A['phd_b'].sum(), A['phn'].sum())}; "
          f"values moved: {sh(A['nzv'].sum(), 768 * 25 * n)}; flips: {sh(A['fl'].sum(), A['phn'].sum())}, shuffle A flips: {sh(A['fl_a'].sum(), A['phn'].sum())}")
print("\n  ahead of the front, by distance from the front: photons of d / photons of R(w+1), at snapshots 1, 4, 12, 24; values moved at 24")
for k in sorted(dist):
    e = dist[k]; print(f"  +{k:2d}: " + "  ".join(f"s{s}: {pc(e[0, s], e[1, s])}" for s in (1, 4, 12, 24)) + f"  | values moved s24: {pc(e[2, 24], e[3, 24])}  ({int(e[3, 24]) // 768} position-waves)")

# ---------------------------------------------------------------- W3
print("\nW3  WHAT A DELTA WAVE POURS (waves 2 onward, moving positions; work = light photons x weight photons over every meeting)")
w3 = {r: {k: np.zeros(49, np.int64) for k in ("full", "delta", "dsh", "lmov", "lines", "pmov", "piles")} for r in ("front", "ahead")}
NIN = np.array([768 if n.split(".")[0] in ("qkv", "fc", "head") else 768 if n.startswith("proj") else 3072 for n in BANKS])
NOUT = np.array([2304 if n.startswith("qkv") else 3072 if n.startswith("fc") else 768 if n.split(".")[0] in ("proj", "mproj") else 50257 for n in BANKS])
gate = {k: np.zeros(12, np.int64) for k in ("pmov", "Hmov", "Gmov", "gmov", "n")}
attw = {r: np.zeros(3, np.int64) for r in ("front", "ahead")}
for x in X:
    T = x["T"]; S = int(x["S"]); P = int(x["P"])
    for k in range(len(x["nzv"])):
        a, b = T[k], T[k + 1]; f = np.flatnonzero(a != b)[0]
        for p in range(f, P):
            r = "front" if p == f else "ahead"; W = w3[r]; rd = S - 1 <= p <= P - 2
            W["full"] += x["full"][k + 1, p]; W["delta"] += x["delta"][k + 1, p]; W["dsh"] += x["dsh"][k + 1, p]
            W["lmov"] += x["lmov"][k, p]; W["lines"] += NIN * np.array([1] * 48 + [int(rd)])
            W["pmov"][:48] += x["pmov"][k, p]; W["piles"][:48] += NOUT[:48]
            attw[r] += [x["attf"][k + 1, p].sum(), x["attd"][k + 1, p].sum(), x["attsh"][k + 1, p].sum()]
            gate["pmov"] += x["pmov"][k, p, KIND["fc"]]; gate["Hmov"] += x["Hmov"][k, :, p]; gate["Gmov"] += x["Gmov"][k, :, p]
            gate["gmov"] += x["gmov"][k, :, p]; gate["n"] += 1
for r in ("front", "ahead"):
    W = w3[r]
    print(f"\n  {r}:  bank | pour d / pour whole | SHUFFLE A: pour d against another position / pour whole | input lines moved | piles moved")
    for kd, ix in KIND.items():
        print(f"  {kd:6s} | {sh(W['delta'][ix].sum(), W['full'][ix].sum()):>44s} | {sh(W['dsh'][ix].sum(), W['full'][ix].sum()):>44s} | {sh(W['lmov'][ix].sum(), W['lines'][ix].sum()):>26s} | "
              + (sh(W['pmov'][ix].sum(), W['piles'][ix].sum()) if kd != "head" else "-"))
    print(f"  all    | {sh(W['delta'].sum(), W['full'].sum()):>44s} | {sh(W['dsh'].sum(), W['full'].sum()):>44s}")
    print(f"  the arrival (light on light) | {sh(attw[r][1], attw[r][0])} | shuffle A {sh(attw[r][2], attw[r][0])}")
print("\n  by layer, moving positions (front + ahead): pour d / pour whole for each bank")
for l in range(12):
    print(f"  layer {l:2d}: " + "  ".join(f"{kd} {pc(w3['front']['delta'][4 * l + i] + w3['ahead']['delta'][4 * l + i], w3['front']['full'][4 * l + i] + w3['ahead']['full'][4 * l + i])}"
                                    for i, kd in enumerate(("qkv", "proj", "fc", "mproj"))))
print("\n  the bend (3,072 piles a position), moving positions: fc piles moved | bend inputs moved | bend outputs moved | positions whose gauge moved")
for l in range(12):
    n = int(gate["n"][l])
    print(f"  layer {l:2d}: {sh(gate['pmov'][l], 3072 * n):>26s} | {sh(gate['Hmov'][l], 3072 * n):>26s} | {sh(gate['Gmov'][l], 3072 * n):>26s} | {sh(gate['gmov'][l], n)}")

# ---------------------------------------------------------------- W4
print("\nW4  WHOLE RUNS (photon meetings: light photons x weight photons)")
tw = {k: 0 for k in ("full_moving", "full_all", "delta", "dsh", "ow", "ow_from_soup", "full_moving_noP", "delta_noP")}
ta = {k: 0 for k in ("full_moving", "full_all", "delta", "ow")}
pours = {"full_moving": 0, "full_all": 0, "ow": 0}; nzd_nonmoving = 0
for x in X:
    mv = x["moving"]; P = int(x["P"])
    tw["full_moving"] += int((x["full"].sum(-1) * mv).sum()); tw["full_all"] += int(x["full"].sum())
    tw["delta"] += int(x["delta"].sum()); tw["dsh"] += int((x["dsh"].sum(-1) * mv).sum())
    tw["full_moving_noP"] += int((x["full"].sum(-1) * mv)[:, :P - 1].sum()); tw["delta_noP"] += int(x["delta"][:, :P - 1].sum())
    tw["ow"] += int(x["ow_bank"].sum()); tw["ow_from_soup"] += int(x["full"][-1, :P - 1].sum())
    nzd_nonmoving += int((x["delta"].sum(-1) * ~mv).sum())
    ta["full_moving"] += int((x["attf"].sum(-1) * mv).sum()); ta["full_all"] += int(x["attf"].sum()); ta["delta"] += int(x["attd"].sum()); ta["ow"] += int(x["ow_att"].sum())
    pours["full_moving"] += int(mv.sum()); pours["full_all"] += mv.size; pours["ow"] += P - 1
print(f"  position-pours: full soup {pours['full_moving']:,} (every position every wave: {pours['full_all']:,}); one-word machine {pours['ow']:,}")
print(f"  one-word machine run on its own = the soup's final light at positions 0..P-2: {'YES' if tw['ow'] == tw['ow_from_soup'] else 'NO'}; delta work at positions that did not move: {nzd_nonmoving}")
print(f"  BANKS   one-word machine        {tw['ow']:>20,}")
print(f"          full soup (moving)      {tw['full_moving']:>20,}   = {sh(tw['full_moving'], tw['ow'])} of one-word")
print(f"          full soup (every pos)   {tw['full_all']:>20,}")
print(f"          delta soup              {tw['delta']:>20,}   = {sh(tw['delta'], tw['full_moving'])} of full soup; {sh(tw['delta'], tw['ow'])} of one-word")
print(f"          delta soup, shuffle A   {tw['dsh']:>20,}   = {sh(tw['dsh'], tw['full_moving'])} of full soup")
print(f"          without position P-1 (never read): full soup {tw['full_moving_noP']:,}, delta soup {tw['delta_noP']:,} = {sh(tw['delta_noP'], tw['ow'])} of one-word")
w1 = sum(int(x["full"][0].sum()) for x in X)
print(f"          of which wave 1 (poured whole in both soups): {w1:,} = {sh(w1, tw['ow'])} of one-word")
print(f"  ARRIVAL one-word {ta['ow']:,}; full soup (moving) {ta['full_moving']:,}; delta soup {ta['delta']:,} = {sh(ta['delta'], ta['full_moving'])} of full soup")
tot_ow, tot_full, tot_delta = tw["ow"] + ta["ow"], tw["full_moving"] + ta["full_moving"], tw["delta"] + ta["delta"]
print(f"  BOTH    one-word {tot_ow:,}; full soup {tot_full:,} = {sh(tot_full, tot_ow)}; delta soup {tot_delta:,} = {sh(tot_delta, tot_full)} of full, {sh(tot_delta, tot_ow)} of one-word")
