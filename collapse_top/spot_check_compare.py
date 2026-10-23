"""THE CHECKER'S REBUILD AGAINST THE SPOT-BOUND RUN, position by position and stage by stage (whole numbers only).
Reads collapse_top/spot_check_<mode>_<write>[_p<i>of<n>].npz (spot_check_rebuild.py) and the run's own
collapse_top/spot_<write>_real_s1.npz (spot_top.py), and compares at the same positions, for every G and schedule:
words still in and words entering at every stage, meetings at every stage, every part of the bound's own work at every
stage, bound broken, the words read to the end and their divides and standing amounts, the top 5 (set and order) and
the race winner; and the plain read's own numbers (its top 5, its winner, 916M basis, its divide, standing amounts, the
light's lit pixels). On the refs also the race winner saved with the light; on the windows judge5's reference top 5.
    python3 collapse_top/spot_check_compare.py
"""
import glob, re, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; D = f"{HERE}/collapse_top"
L = np.load(f"{D}/light_top.npz"); SRC, WIN, REFW = L["src"], L["win"], L["refs_winner"]
REF = np.load(f"{HERE}/judge5_ref.npz")["top5"]
FIELDS = ("surv", "enter", "meet", "parts", "bad", "top", "win", "final", "findiv", "finst")
PLF = ("top", "win", "plain916", "plainw", "pldiv", "plst", "lightph")
groups = {}
for f in sorted(glob.glob(f"{D}/spot_check_*_*.npz")):
    m = re.match(r".*/spot_check_(bound|top)_(signed|plain)(?:_p(\d+)of(\d+))?\.npz$", f)
    if m: groups.setdefault((m.group(1), m.group(2), m.group(4)), []).append(f)
for (mode, write, n), files in sorted(groups.items()):
    if n is not None and len(files) != int(n): print(f"{mode} {write}: {len(files)} of {n} parts so far")
    Zs = [np.load(f) for f in sorted(files, key=lambda s: int(re.search(r"_p(\d+)of\d+\.npz$", s).group(1)) if re.search(r"_p(\d+)of\d+\.npz$", s) else 0)]
    sel = np.concatenate([z["sel"] for z in Zs]); A = np.load(f"{D}/spot_{write}_real_s1.npz")
    assert np.array_equal(A["sel"], np.arange(len(A["sel"])))
    def cat(k): return np.concatenate([z[k] for z in Zs])
    print(f"\n==== {mode} {write}: {len(sel)} positions ({len(files)} file(s)); against spot_{write}_real_s1.npz")
    for G in Zs[0]["gs"]:
        G = int(G)
        for light in ("real", "shuffled"):
            pfx = f"{light}_{G}"
            if f"{pfx}_tot" not in Zs[0].files: continue
            keys = Zs[0][f"{pfx}_tot_keys"].tolist(); tot = sum(z[f"{pfx}_tot"] for z in Zs); t = dict(zip(keys, tot.tolist()))
            print(f"  {light:8s} G={G:2d}: [own checks] pairs = product bad {t['pile_bad']}; pairs above the group bound "
                  f"{t['viol']} of {t['cells']:,} (word, a, b); unread part outside [-low, high] {t['stage_bad']} of {t['stage_cells']:,}"
                  f" (word, stage); score outside [lo, hi] {t['score_bad']}; slow recount bad {t['slow_bad']} of {t['slow_checked']};"
                  f" meetings = plain meetings bad {t['meet_sum_bad']}; collapse bound broken {t['collapse_bad']};"
                  f" top 5 differs from its own plain read {t['top_vs_plain']}")
            if light != "real": continue
            diffs = {}
            for k in PLF:
                mine = cat(f"{pfx}_{k}_pl"); theirs = A[{"top": "top_pl", "win": "win_pl"}.get(k, k)][sel]
                diffs[f"plain {k}"] = int((mine.reshape(len(sel), -1) != theirs.reshape(len(sel), -1)).any(1).sum())
            print("     plain read vs the run: positions differing: " + ", ".join(f"{k} {v}" for k, v in diffs.items()))
            for s in ("diag", "light"):
                d = {}
                for k in FIELDS:
                    mine = cat(f"{pfx}_{k}_{s}"); theirs = A[f"{k}_{G}_{s}"][sel]
                    d[k] = int((mine.reshape(len(sel), -1) != theirs.reshape(len(sel), -1)).any(1).sum())
                mt = cat(f"{pfx}_top_{s}"); mw = cat(f"{pfx}_win_{s}")
                rr = SRC[sel] == 0; wq = np.flatnonzero(SRC[sel] == 1)
                ref_same = sum(int((mt[j] == REF[WIN[sel[j]], int(sel[j]) - 480 - 20 * WIN[sel[j]]]).all()) for j in wq)
                pl_same = int((mt == cat(f"{pfx}_top_pl")).all(1).sum())
                print(f"     {s:5s}: positions differing from the run: " + ", ".join(f"{k} {v}" for k, v in d.items()) +
                      f" | top 5 = own plain read {pl_same}/{len(sel)}; windows: = judge5 reference {ref_same}/{len(wq)};"
                      f" refs: race winner = saved {int((mw[rr] == REFW[sel[rr]]).sum())}/{int(rr.sum())}")
                sv, me, pr = cat(f"{pfx}_surv_{s}"), cat(f"{pfx}_meet_{s}"), cat(f"{pfx}_parts_{s}")
                own = pr[:, :, :4].sum(2).sum(1) + cat(f"{pfx}_lightph_pl"); pl = cat(f"{pfx}_plain916_pl").astype(np.float64)
                print(f"            mean over these positions (mine): meetings {me.sum(1).mean() / 1e6:.1f}M + bound's own {own.mean() / 1e6:.1f}M"
                      f" = {100 * (me.sum(1) + own).mean() / pl.mean():.1f}% of plain {pl.mean() / 1e6:.1f}M; survivors median by stage "
                      + " ".join(str(int(np.median(sv[:, k]))) for k in range(sv.shape[1])))
            st = Zs[0][f"store_{G}"]
            print(f"     stored counts: largest {int(st[0])}, room {int(st[1])} bits")
