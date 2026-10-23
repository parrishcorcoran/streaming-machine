"""reach/base_check.py: THE MACHINE OF reach/fastm.py AGAINST base2.py'S OWN RECORDED RUNS, AND THE TOP-5 GAPS.

Checks (whole numbers, exact equality):
  - every pile of the 48 banks and the stream after every half layer, at all 648 refs positions, against
    recording/C_ref/plain_refs.npz (base2.py's own steps, one text at a time, made by recording/C_plain.py);
  - the light at the top (x16, ex, LS) at the 6,480 read positions against collapse_top/light_top.npz;
  - the top 5 (set and order) at the 480 refs read positions against plain_refs.npz and at the 6,000 judge positions
    against judge5_ref.npz; every word's score's sha256 at the refs positions against plain_refs.npz.
Then, at each of the 6,480 read positions, the top 6 words and their scores (whole numbers at the scores' scale,
2^-16): the four gaps that keep the order of the top 5 and the gap between the 5th and the 6th.
Saved: reach/base_top.npz (x16, ex, LS at the 7,848 positions' top; top6 and their scores at the 6,480 read rows).
PREDICTION: every check identical. The smallest of the 30,000 judge gaps is a few units (2^-16 of a score unit), and
  the refs hold an exact tie for 5th at refs position 366 (FINDINGS: words 655 and 1654, both -9,021,364), so its
  smallest gap there is 0.
    python3 reach/base_check.py      -> reach/base_check.log, reach/base_top.npz
"""
import sys, os, hashlib, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastm as F
HERE = F.HERE; OUT = f"{HERE}/reach"; LOG = open(f"{OUT}/base_check.log", "w"); t0 = time.time()
def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz"); NR = int(F.OFF[30]); assert NR == 648
bad = []; checked = [0]
def hook(name, l, d):
    if name == "stream in": return
    if name in ("stream a", "stream"):
        key = f"{name.replace(' ', '_')}_{l}"; v = d
    else:
        key = f"{name.split('.')[0]}_pile_{l}"; v = d["pile"]
    checked[0] += 1
    if not np.array_equal(REF[key], v[:NR]): bad.append(key)
x16, ex, LS = F.run(hook=hook, log=log)
log(f"piles and streams at the 648 refs positions: {checked[0]} arrays checked, differing: {bad if bad else 'none'}")
LT = np.load(f"{HERE}/collapse_top/light_top.npz"); RR = F.READ_ROWS
same_light = bool(np.array_equal(LT["x16"], x16[RR]) and np.array_equal(LT["ex"], ex[RR]) and np.array_equal(LT["LS"], LS[RR]))
log(f"light at the top at the 6,480 read rows identical to collapse_top/light_top.npz: {same_light}")
T6 = np.zeros((len(RR), 6), np.int64); S6 = np.zeros((len(RR), 6), np.int64); T5 = np.zeros((len(RR), 5), np.int64); sha_ok = 0
for sl, S in F.top_scores(x16[RR], ex[RR], LS[RR]):
    t6 = F.top6(S); T6[sl] = t6; S6[sl] = np.take_along_axis(S, t6, 1)
    T5[sl] = F.top5(S)
    for k, r in enumerate(range(sl.start, min(sl.stop, len(RR)))):
        if r < F.NREF_ROWS:
            sha_ok += int(np.array_equal(np.frombuffer(hashlib.sha256(np.ascontiguousarray(S[k], np.int64).tobytes()).digest(), np.uint8), REF["score_sha"][RR[r]]))
J5 = np.load(f"{HERE}/judge5_ref.npz")["top5"].reshape(-1, 5)
t5_refs = int((T5[:480] == REF["top5"][RR[:480]]).all(1).sum()); t5_win = int((T5[480:] == J5).all(1).sum())
log(f"every word's score (sha256) at the 480 refs read rows: {sha_ok}/480; top 5 (set and order): refs {t5_refs}/480, "
    f"judge {t5_win}/6000")
G = S6[:, :5] - S6[:, 1:]                                      # gaps s1-s2, s2-s3, s3-s4, s4-s5, s5-s6
gmin = G.min(1)
for nm, sl in (("refs (480)", slice(0, 480)), ("judge (6,000)", slice(480, None))):
    g = gmin[sl]; srt = np.sort(g)
    log(f"{nm}: smallest gap at each position (of the five): min {srt[0]}, then {srt[1:10].tolist()}; positions with gap 0: "
        f"{int((g == 0).sum())}; below 16: {int((g < 16).sum())}; below 256: {int((g < 256).sum())}; median {int(np.median(g))}")
    log(f"   the 5th-6th gap alone: min {int(G[sl, 4].min())}, median {int(np.median(G[sl, 4]))}; the order gaps (s1-s2..s4-s5): "
        f"min {int(G[sl, :4].min())}, median {int(np.median(G[sl, :4]))}")
np.savez(f"{OUT}/base_top.npz", x16=x16.astype(np.int32), ex=ex, LS=LS, top6=T6, s6=S6, top5=T5)
log(f"saved reach/base_top.npz ({time.time() - t0:.0f}s, max memory {F.rss()} MB)")
