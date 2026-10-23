"""The fine-grid staged top read, every run side by side (from fine_<sched>_<light>.npz of fine_staged_top.py). Whole
numbers only. Work per position in photon meetings (millions, M), the average over the positions.
  - Per stage: words entering, survivors (median, mean, min, max), meetings, and the bound's work stated apart, both in
    staged_top.py's unit (table reads and adds: 15 a word a stage for the plain bound, 60 for the signed, plus the
    tables built from the light) and in photons by combined/machine.py's rule.
  - Totals against the plain top read measured here on the same positions (916M), and on the 904M basis
    (staged_top's report: the rest of the machine 3,667M - 904M = 2,763M, plus the top read scaled to 904M).
  - The combined machine (combined/refs_counts.json, its own totals): its top read is 506.0M meetings + 61.5M bound
    work a token. Its top read is replaced by this one, scaled from the 480 refs positions here (the combined count
    covers all 648 positions of the 30 texts; the 168 prompt positions are not in light_top.npz): an estimate.
  - The share of the plain read in the top k diagonals, from the light's and the stamps' photons alone.
    python3 collapse_top/fine_report.py
"""
import json, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; D = f"{HERE}/collapse_top"
PL, NW, LINES = 15, 50257, 768
REST, TOP = 3667 - 904, 904
def r(v): return int(round(v))
def M_(v): return f"{v / 1e6:,.1f}M"
L = np.load(f"{D}/light_top.npz"); SRC = L["src"]; WINL = L["win"]; REFW = L["refs_winner"]
REF5 = np.load(f"{HERE}/judge5_ref.npz")["top5"]

# ---------- the combined machine's own counts ----------
C = json.load(open(f"{HERE}/combined/refs_counts.json")); n = C["npos"]; T = C["total"]
def per_token(total): return (total + n // 2) // n
comb_total = per_token(sum(v for k, v in T.items() if k.split("|")[0] in ("comb", "both")))
plain_total = per_token(sum(v for k, v in T.items() if k.split("|")[0] in ("plain", "both")))
c_meet = per_token(sum(v for k, v in T.items() if k.startswith("comb|top read|")))
c_bound = per_token(sum(v for k, v in T.items() if k.startswith("comb|extra|top read")))
c_rest = per_token(sum(v for k, v in T.items() if k.startswith("comb|rest|top ")))
p_top = per_token(T["plain|top read|top meetings, plain"])
print(f"combined machine (refs_counts.json, {n} positions): whole {comb_total:,} of plain {plain_total:,} a token "
      f"({100 * comb_total / plain_total:.2f}%); its top read: meetings {c_meet:,} + bound work {c_bound:,} + words read to the end "
      f"{c_rest:,} (divide, standing amount); plain top read {p_top:,}")

# ---------- the read's share by diagonal and by light place (photons alone) ----------
W16 = np.load(f"{HERE}/base2_gpt2.npz")["head.w"]                                  # (768, NW) int16
TA = np.abs(W16.astype(np.int32))
CNT_ALL = np.stack([((TA >> b) & 1).sum(1) for b in range(PL)]).astype(np.float64)  # (15 b, 768): words lit at b, per line
del W16, TA
def pair_meetings(X):                                                             # (positions, 15 a, 15 b), summed below
    tot = np.zeros((PL, PL))
    for i in range(0, len(X), 500):
        ax = np.abs(X[i:i + 500])
        bits = np.stack([((ax >> a) & 1) for a in range(PL)], 1).astype(np.float64)   # (P, 15 a, 768)
        tot += np.einsum("pal,bl->ab", bits, CNT_ALL)
    return tot / len(X)
X_real = L["x16"].astype(np.int64)
rng = np.random.default_rng(2026); X_shuf = np.stack([x[rng.permutation(LINES)] for x in X_real])   # as fine_staged_top.py
a_, b_ = np.arange(PL)[:, None], np.arange(PL)[None, :]
for name, X in (("real", X_real), ("shuffled", X_shuf)):
    PM = pair_meetings(X); tot = PM.sum()
    dg = " ".join(f"{k}:{100 * PM[a_ + b_ >= 29 - k].sum() / tot:.1f}%" for k in range(2, 29, 2))
    lt = " ".join(f"{k}:{100 * PM[(a_ >= 15 - k)[:, 0]].sum() / tot:.1f}%" for k in range(2, 13, 2))
    print(f"\nshare of the plain read ({M_(tot)} a position, light {name}) in the top k diagonals: {dg}")
    print(f"                                         in the light's top k places: {lt}")

# ---------- every run ----------
SUMMARY = {}
for sched in ("diag", "light"):
    for light in ("real", "shuffled"):
        Z = np.load(f"{D}/fine_{sched}_{light}.npz"); KS = Z["ks"].tolist(); NS = len(KS)
        bad = dict(zip(Z["bad_keys"].tolist(), Z["bad"].tolist()))
        CS = [(a_ >= PL - k) if sched == "light" else (a_ + b_ >= 2 * (PL - 1) + 1 - k) for k in KS]
        CS = [np.broadcast_to(c, (PL, PL)) for c in CS]
        TABLE_ADDS = [int((~CS[s]).any(1).sum()) * PL * (LINES + 1) for s in range(NS)]
        P = Z["plain"]; pl = P.mean(); src = SRC[Z["sel"]]
        print(f"\n=== {sched} schedule, light {light}" + ("  (CONTROL: each position's light shuffled across its lines)" if light == "shuffled" else "")
              + f": {len(P)} positions; stages top {KS} then all")
        print("  checks: " + ", ".join(f"{k} {v}" for k, v in bad.items()))
        for row, sx, tr, nt in (("plain bound", "", 15, 1), ("signed bound", "_s", 60, 2)):
            S, E, Wk, BW, FW = Z["surv" + sx], Z["enter" + sx], Z["work" + sx], Z["bw" + sx], Z["finw" + sx]
            same5 = int((Z["top_st" + sx] == Z["top_pl"]).all(1).sum()); samew = int((Z["win_st" + sx] == Z["win_pl"]).sum())
            line = f"  [{row}] top 5 (set and order) = the plain read {same5}/{len(P)}; race winner = the plain read's {samew}/{len(P)}"
            if light == "real":
                m = src == 0; line += f"; refs: race winner = the one saved with the light {int((Z['win_st' + sx][m] == REFW[Z['sel'][m]]).sum())}/{int(m.sum())}"
                wq = np.flatnonzero(src == 1); NREF = int((SRC == 0).sum())
                rows = [(q, *divmod(int(Z["sel"][q]) - NREF, 20)) for q in wq]; assert all(WINL[Z["sel"][q]] == i for q, i, j in rows)
                line += f"; windows: top 5 = judge5's reference {sum(int((Z['top_st' + sx][q] == REF5[i, j]).all()) for q, i, j in rows)}/{len(wq)}"
            print(line)
            print(f"    {'stage':>10s} {'entering':>9s} {'surv med':>9s} {'mean':>7s} {'min':>6s} {'max':>6s} {'meetings':>10s} {'bound reads+adds':>17s} {'bound photons':>14s}")
            for s, k in enumerate(KS):
                v = S[:, s]
                print(f"    {'top ' + str(k):>10s} {r(E[:, s].mean()):9d} {r(np.median(v)):9d} {r(v.mean()):7d} {v.min():6d} {v.max():6d} "
                      f"{M_(Wk[:, s].mean()):>10s} {M_(E[:, s].mean() * tr + nt * TABLE_ADDS[s]):>17s} {M_(BW[:, s].sum(1).mean()):>14s}")
            print(f"    {'all':>10s} {'':9s} {'':9s} {'':7s} {'':6s} {'':6s} {M_(Wk[:, NS].mean()):>10s}")
            tot = Wk.sum(1).mean(); ops = (E.sum(1) * tr).mean() + nt * sum(TABLE_ADDS)
            bwp = BW.sum(1).mean(0); bw = bwp.sum(); fw = FW.sum(1).mean()
            print(f"    meetings {M_(tot)} of plain {M_(pl)} = {100 * tot / pl:.1f}% of plain;  on the 904M basis: top read {r(TOP * tot / pl)}M, "
                  f"whole machine {r(REST + TOP * tot / pl)}M of 3,667M = {100 * (REST + TOP * tot / pl) / 3667:.1f}%")
            print(f"    bound, staged_top's unit: table reads and adds {M_(ops)} = {100 * ops / pl:.2f}% of the plain meetings; gauge reads "
                  f"{M_(2 * E.sum(1).mean())}, compares {M_(E.sum(1).mean())}")
            print(f"    bound, in photons (combined's rule): {M_(bw)} = {100 * bw / pl:.1f}% of the plain meetings  (tables {M_(bwp[0])}, "
                  f"adds {M_(bwp[1])}, divides {M_(bwp[2])}, standing amounts {M_(bwp[3])}); words read to the end: divide and standing amount {M_(fw)}")
            nosa = (S == E)                                                                   # stages that set no word aside there
            print(f"    bound photons at stages that set no word aside at that position: {100 * (BW.sum(2) * nosa).sum() / BW.sum():.1f}%; "
                  f"first stage setting a word aside: median top {KS[int(np.median(np.argmax(~nosa, 1)))]}, earliest top {KS[int(np.argmax(~nosa, 1).min())]}")
            print(f"    meetings + bound photons: {100 * (tot + bw) / pl:.1f}% of plain")
            for nm, m in (("refs", src == 0), ("windows", src == 1)):
                print(f"      {nm:7s}: meetings {100 * Wk[m].sum(1).mean() / P[m].mean():.1f}% of plain, bound photons {100 * BW[m].sum((1, 2)).mean() / P[m].mean():.1f}%")
            m = src == 0
            SUMMARY[(sched, light, row)] = (Wk[m].sum(1).mean() / P[m].mean(), BW[m].sum((1, 2)).mean() / P[m].mean(),
                                            FW[m].sum(1).mean() / P[m].mean(), tot / pl, bw / pl)

# ---------- what it means for the combined machine ----------
print(f"\n=== the combined machine with this top read in place of its own (scaled from the 480 refs positions, real light)")
print(f"  now: top read {c_meet + c_bound + c_rest:,} a token ({100 * (c_meet + c_bound + c_rest) / p_top:.1f}% of its plain top read {p_top:,}); "
      f"whole {comb_total:,} = {100 * comb_total / plain_total:.2f}% of plain")
for (sched, light, row), (fm, fb, ff, _, _) in SUMMARY.items():
    if light != "real": continue
    new_m, new_b, new_f = r(p_top * fm), r(p_top * fb), r(p_top * ff)
    whole = comb_total - c_meet - c_bound - c_rest + new_m + new_b + new_f
    print(f"  {sched:5s} [{row}]: meetings {M_(new_m)} + bound {M_(new_b)} + words read to the end {M_(new_f)} = {M_(new_m + new_b + new_f)} "
          f"({100 * (new_m + new_b + new_f) / p_top:.1f}% of the plain top read);  whole machine {whole:,} = {100 * whole / plain_total:.2f}% of plain")
