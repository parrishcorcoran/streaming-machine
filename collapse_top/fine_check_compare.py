"""CHECKER: the rebuild (fine_check_rebuild.py parts) against the fine run's saved results, position by position, and the
report's headline numbers recomputed from the fine run's own saved results (fine_diag_*.npz, fine_light_*.npz) and from
combined/refs_counts.json. Whole numbers; percentages printed only at the end.
    python3 collapse_top/fine_check_compare.py
"""
import glob, json, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; D = f"{HERE}/collapse_top"
L = np.load(f"{D}/light_top.npz"); SRC = L["src"]
def join(tag):
    fs = sorted(glob.glob(f"{D}/fine_check_{tag}_p*of*.npz"))
    fs = [f for f in fs if f[len(f"{D}/fine_check_{tag}_p"):].split("of")[0].isdigit()]
    if not fs: return None
    n = int(fs[0].split("of")[-1][:-4]); fs = [f"{D}/fine_check_{tag}_p{i}of{n}.npz" for i in range(1, n + 1)]
    Z = [np.load(f) for f in fs]
    out = {k: (np.concatenate([z[k] for z in Z]) if k not in ("ks", "bad_keys", "bad") else Z[0][k]) for k in Z[0].files}
    out["bad"] = sum(z["bad"] for z in Z); return out

print("=== the rebuild against the fine run, position by position")
for tag, ref in (("real", "fine_diag_real.npz"), ("shuffled", "fine_diag_shuffled.npz")):
    C = join(tag)
    if C is None: print(f"  {tag}: no rebuild parts"); continue
    F = np.load(f"{D}/{ref}"); q = C["sel"]; N = len(q)
    print(f"  light {tag}: {N} positions ({int((SRC[q] == 0).sum())} refs, {int((SRC[q] == 1).sum())} windows); rebuild checks: "
          + ", ".join(f"{k} {v}" for k, v in zip(C["bad_keys"].tolist(), C["bad"].tolist())))
    pairs = [("surv", "surv"), ("enter", "enter"), ("work", "work"), ("plain", "plain"), ("top_pl", "top_pl"), ("win_pl", "win_pl"),
             ("top_st", "top_st"), ("win_st", "win_st"), ("surv_s", "surv_s"), ("enter_s", "enter_s"), ("work_s", "work_s"),
             ("top_ss", "top_st_s"), ("win_ss", "win_st_s")]
    for mine, theirs in pairs:
        a, b = C[mine], F[theirs][q]
        eq = (a == b).reshape(N, -1).all(1)
        print(f"    {mine:8s} equal at {int(eq.sum())}/{N} positions")
    for i, nm in enumerate(("tables", "adds", "divides", "standing")):
        eq = (C["bw"][:, :, i] == F["bw"][q][:, :, i]).all(1); print(f"    bound photons, {nm:8s} equal at {int(eq.sum())}/{N}")
    same5 = (C["top_st"] == C["top_pl"]).all(1); samesc = (C["sc_st"] == C["sc_pl"]).all(1)
    same5s = (C["top_ss"] == C["top_pl"]).all(1); samescs = (C["sc_ss"] == C["sc_pl"]).all(1)
    tie = C["sc56"][:, 0] == C["sc56"][:, 1]
    print(f"    rebuild top 5 = plain read (set and order): plain bound {int(same5.sum())}/{N}, signed {int(same5s.sum())}/{N}; five scores equal "
          f"{int(samesc.sum())}/{N}, {int(samescs.sum())}/{N}; race {int((C['win_st'] == C['win_pl']).sum())}/{N}, {int((C['win_ss'] == C['win_pl']).sum())}/{N};"
          f" positions with a tie between 5th and 6th in the plain read: {int(tie.sum())} {q[tie].tolist()[:10]}")
    pl = C["plain"].sum(); wk = C["work"].sum(); wks = C["work_s"].sum()
    print(f"    rebuild on these positions: meetings {wk}/{pl} = {100 * wk / pl:.2f}% (signed {100 * wks / pl:.2f}%); bound photons {100 * C['bw'].sum() / pl:.2f}%")
    for s, k in enumerate(C["ks"]):
        v = C["surv"][:, s]
        print(f"      top {k:2d}: survivors median {int(np.median(v))}, mean {v.mean():.0f}, min {v.min()}, max {v.max()}; meetings {C['work'][:, s].mean() / 1e6:.2f}M")
    for nm, m in (("refs", SRC[q] == 0), ("windows", SRC[q] == 1)):
        print(f"      {nm}: meetings {100 * C['work'][m].sum() / C['plain'][m].sum():.2f}% of plain, bound photons {100 * C['bw'][m].sum() / C['plain'][m].sum():.2f}%")
C7 = join("shuffled_seed7")
if C7 is not None:
    pl = C7["plain"].sum(); print(f"  light shuffled, another seed (7): {len(C7['sel'])} positions; checks " + ", ".join(f"{k} {v}" for k, v in zip(C7["bad_keys"].tolist(), C7["bad"].tolist()))
                                  + f"; top 5 = plain {int((C7['top_st'] == C7['top_pl']).all(1).sum())}; meetings {100 * C7['work'].sum() / pl:.2f}%; "
                                  f"bound photons {100 * C7['bw'].sum() / pl:.2f}%; top 14 median {int(np.median(C7['surv'][:, 6]))}, top 16 median {int(np.median(C7['surv'][:, 7]))}")

print("\n=== the report's numbers from the fine run's own saved results")
for sched in ("diag", "light"):
    for light in ("real", "shuffled"):
        Z = np.load(f"{D}/fine_{sched}_{light}.npz"); KS = Z["ks"].tolist(); NS = len(KS); P = Z["plain"]; N = len(P)
        for row, sx, tr, nt in (("plain", "", 15, 1), ("signed", "_s", 60, 2)):
            S, E, Wk, BW = Z["surv" + sx], Z["enter" + sx], Z["work" + sx], Z["bw" + sx]
            a_, b_ = np.arange(15)[:, None], np.arange(15)[None, :]
            rows = [min(15, 29 - k) if sched == "diag" else 15 for k in KS]
            if sched == "light": rows = [15 - k for k in KS]               # light places still unread at the stage
            ops = (E.sum(1) * tr).sum() + N * nt * sum(rows) * 15 * 769
            same = int((Z["top_st" + sx] == Z["top_pl"]).all(1).sum())
            nosa = S == E; bws = BW.sum(2)
            first = np.argmax(~nosa, 1)
            print(f"  {sched} {light} [{row}]: top 5 = plain {same}/{N}; meetings {100 * Wk.sum() / P.sum():.2f}%; table reads+adds {100 * ops / P.sum():.3f}%; "
                  f"bound photons {100 * BW.sum() / P.sum():.2f}% (" + ", ".join(f"{BW[:, :, i].sum() / N / 1e6:.1f}M" for i in range(4))
                  + f"); at no-set-aside stages {100 * (bws * nosa).sum() / bws.sum():.1f}%; first set-aside median top {KS[int(np.median(first))]}, earliest top {KS[first.min()]}"
                  f"; plain {P.mean() / 1e6:.1f}M")
            if light == "real" and row == "plain":
                for s, k in enumerate(KS):
                    v = S[:, s]; print(f"      top {k:2d}: median {int(np.median(v))} mean {v.mean():.0f} min {v.min()} max {v.max()}; meetings {Wk[:, s].mean() / 1e6:.1f}M")
                m = SRC[Z["sel"]] == 0
                print(f"      refs: meetings {100 * Wk[m].sum(1).mean() / P[m].mean():.2f}%, bound photons {100 * BW[m].sum((1, 2)).mean() / P[m].mean():.2f}%")
O = np.load(f"{D}/fine_diag_real_order.npz")
print(f"  order run: top 5 (set and order) = its own plain read: plain bound {int((O['top_st'] == O['top_pl']).all(1).sum())}/{len(O['plain'])}, "
      f"signed {int((O['top_st_s'] == O['top_pl']).all(1).sum())}/{len(O['plain'])}")

print("\n=== the combined machine (combined/refs_counts.json)")
J = json.load(open(f"{HERE}/combined/refs_counts.json")); n = J["npos"]; T = J["total"]
comb = sum(v for k, v in T.items() if k.split("|")[0] in ("comb", "both")); plain = sum(v for k, v in T.items() if k.split("|")[0] in ("plain", "both"))
cm = sum(v for k, v in T.items() if k.startswith("comb|top read|")); cb = sum(v for k, v in T.items() if k.startswith("comb|extra|top read"))
cr = sum(v for k, v in T.items() if k.startswith("comb|rest|top ")); pt = T["plain|top read|top meetings, plain"]
print(f"  per token: whole {comb / n:,.1f} of {plain / n:,.1f} ({100 * comb / plain:.2f}%); top read meetings {cm / n:,.1f}, bound {cb / n:,.1f}, rest {cr / n:,.1f}; "
      f"top read {(cm + cb + cr) / n:,.1f} = {100 * (cm + cb + cr) / pt:.2f}% of plain top read {pt / n:,.1f}")
Z = np.load(f"{D}/fine_diag_real.npz"); m = SRC[Z["sel"]] == 0; P = Z["plain"][m]
fm = Z["work"][m].sum(1).mean() / P.mean(); fb = Z["bw"][m].sum((1, 2)).mean() / P.mean(); ff = Z["finw"][m].sum(1).mean() / P.mean()
new = (pt / n) * (fm + fb + ff); whole = comb / n - (cm + cb + cr) / n + new
print(f"  diag plain bound in its place (refs ratios: meetings {100 * fm:.2f}%, bound {100 * fb:.2f}%): top read {new / 1e6:.1f}M = {100 * new / (pt / n):.2f}%; "
      f"whole {whole:,.0f} = {100 * whole / (plain / n):.2f}%")
