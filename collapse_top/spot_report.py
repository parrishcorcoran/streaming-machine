"""THE SPOT BOUND, every run side by side (from spot_<write>_<light>_s<seed>[_order][_sub].npz). Whole numbers only.
Work per position, photons, in millions (M). The plain top read: sign and photons, every word, measured here on the
same positions (916M on all 6,480).
  meetings        light photons x word photons on each line, the pairs met, over the words still in (in the writing used)
  bound's own     stored counts read + table values added + tables built + the two bound adds + the light's lit pixels
                  counted into the groups
  total           meetings + bound's own, against the plain read
  every add       total + divides and standing amounts of the bounds + the top divide and standing amount of the words
                  read to the end, against the plain read + its top divide and standing amount for every word
    python3 collapse_top/spot_report.py
"""
import os, numpy as np
D = "/Users/abundancemachine/streaming-machine/collapse_top"
def r(v): return int(round(v))
def M(v): return f"{v / 1e6:,.1f}M"
def load(name):
    p = f"{D}/spot_{name}.npz"
    return np.load(p) if os.path.exists(p) else None
def totals(Z, G, s, idx=None):
    q = slice(None) if idx is None else idx
    meet = Z[f"meet_{G}_{s}"][q].sum(1).astype(np.float64); prt = Z[f"parts_{G}_{s}"][q].sum(1).astype(np.float64)
    own = prt[:, :4].sum(1) + Z["lightph"][q]
    divst = prt[:, 4] + prt[:, 5]; fin = Z[f"findiv_{G}_{s}"][q] + Z[f"finst_{G}_{s}"][q]
    pl = Z["plain916"][q].astype(np.float64); plall = pl + Z["pldiv"][q] + Z["plst"][q]
    return dict(meet=meet.mean(), own=own.mean(), tot=(meet + own).mean(), pl=pl.mean(), every=(meet + own + divst + fin).mean(),
                plall=plall.mean(), prt=prt.mean(0), divst=divst.mean(), plainw=Z["plainw"][q].mean())
def pct(a, b): return f"{100 * a / b:.1f}%"

for write in ("signed", "plain"):
    Z = load(f"{write}_real_s1")
    if Z is None: continue
    N = len(Z["sel"]); bad = dict(zip(Z["bad_keys"].tolist(), Z["bad"].tolist()))
    print(f"\n==================== {write.upper()} {'DIGITS (ups and downs counted apart)' if write == 'signed' else 'PHOTONS (lit counts)'}: {N} positions, groups seed 1")
    print(f"checks: " + ", ".join(f"{k} {v}" for k, v in sorted(bad.items())))
    print(f"plain read {M(Z['plain916'].mean())} a position (sign and photons); in this writing {M(Z['plainw'].mean())} "
          f"= {pct(Z['plainw'].mean(), Z['plain916'].mean())} of it before anything is set aside")
    for s in ("diag", "light"):
        thr = Z["diag_t"] if s == "diag" else Z["light_c"]
        lab = [f"top {2 * (k + 1)}" for k in range(len(thr))] + ["all"]
        print(f"\n--- schedule {s}: stages " + ", ".join(lab))
        for G in Z["gs"]:
            G = int(G); sv, en, me, pr = Z[f"surv_{G}_{s}"], Z[f"enter_{G}_{s}"], Z[f"meet_{G}_{s}"], Z[f"parts_{G}_{s}"]
            same = int((Z[f"top_{G}_{s}"] == Z["top_pl"]).all(1).sum()); bb = int(Z[f"bad_{G}_{s}"].sum())
            t = totals(Z, G, s)
            print(f"G={G:2d}: top 5 = plain read {same}/{N}; bound broken {bb}; read to the end: median {r(np.median(Z[f'final_{G}_{s}']))}")
            print("   survivors (median)  " + " ".join(f"{r(np.median(sv[:, k])):>6d}" for k in range(sv.shape[1])))
            print("   survivors (mean)    " + " ".join(f"{r(sv[:, k].mean()):>6d}" for k in range(sv.shape[1])))
            print("   meetings (M)        " + " ".join(f"{me[:, k].mean() / 1e6:6.1f}" for k in range(me.shape[1])))
            own_k = pr[:, :, :4].sum(2).mean(0)
            print("   bound's own (M)     " + " ".join(f"{v / 1e6:6.1f}" for v in own_k) + f"   + light into groups {M(Z['lightph'].mean())}")
            print(f"   meetings {M(t['meet'])} + bound's own {M(t['own'])} (counts read {M(t['prt'][0])}, table values {M(t['prt'][1])}, "
                  f"tables built {M(t['prt'][2])}, bound adds {M(t['prt'][3])}) = {M(t['tot'])} = {pct(t['tot'], t['pl'])} of plain;"
                  f"  meetings alone {pct(t['meet'], t['pl'])}")
            print(f"   with the bounds' divides and standing amounts ({M(t['divst'])}) and the final words' divides: {M(t['every'])} "
                  f"against {M(t['plall'])} = {pct(t['every'], t['plall'])}; compares {r(t['prt'][6] / 1e3)}K")
    st = Z["store"]
    print("\nstored counts (per word, per place, per group" + (", ups and downs" if write == "signed" else "") + "):")
    for i, G in enumerate(Z["gs"]):
        mx, room, by, byx = st[i]
        print(f"   G={int(G):2d}: room {room} bits a count -> {by / 1e6:.1f} MB ({100 * by / 77.19e6:.0f}% of the 77.19 MB word table);"
              f" at the largest count's width ({int(mx)}): {byx / 1e6:.1f} MB")

print("\n==================== CONTROLS (refs + every 6th window, 1,480 positions; the real run read on the same positions)")
for write in ("signed", "plain"):
    Z = load(f"{write}_real_s1")
    if Z is None: continue
    for ctl, name in (("shuffled", f"{write}_shuffled_s1_sub"), ("seed 2", f"{write}_real_s2_sub")):
        C = load(name)
        if C is None: continue
        idx = C["sel"]; bad = dict(zip(C["bad_keys"].tolist(), C["bad"].tolist()))
        print(f"\n{write} / {ctl}: {len(idx)} positions; checks " + ", ".join(f"{k} {v}" for k, v in sorted(bad.items())))
        for s in ("diag", "light"):
            for G in C["gs"]:
                G = int(G); a, c = totals(Z, G, s, idx), totals(C, G, s)
                same = int((C[f"top_{G}_{s}"] == C["top_pl"]).all(1).sum()); bb = int(C[f"bad_{G}_{s}"].sum())
                sa, sc = Z[f"surv_{G}_{s}"][idx], C[f"surv_{G}_{s}"]
                print(f"  {s:5s} G={G:2d}: real {pct(a['tot'], a['pl'])} (meetings {pct(a['meet'], a['pl'])}) | {ctl} {pct(c['tot'], c['pl'])}"
                      f" (meetings {pct(c['meet'], c['pl'])}); {ctl}: top 5 = its plain read {same}/{len(idx)}, bound broken {bb}")
                print("      survivors median, real   " + " ".join(f"{r(np.median(sa[:, k])):>6d}" for k in range(sa.shape[1])))
                print(f"      survivors median, {ctl:7s}" + " ".join(f"{r(np.median(sc[:, k])):>6d}" for k in range(sc.shape[1])))
