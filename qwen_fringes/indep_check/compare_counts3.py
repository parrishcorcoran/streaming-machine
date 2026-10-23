"""Checker: own counts (counter3.npz) against the delivered machine's (machine10_counts.npz, machine10_top.npz) and the
builder's saved arrays (machine_counts.npz, top_machine.npz) at the same positions; then the shares by part, own counts."""
import json, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/qwen_fringes/indep_check"; QFO = f"{HERE}/qwen_fringes"
T10 = [3, 4, 5, 7, 8, 9, 12, 16, 18, 22]; T3 = [4, 7, 12]; L = 28
meta = json.load(open(f"{HERE}/qwen06b/light/meta.json")); MJ = json.load(open(f"{OUT}/machine10.json"))
CM = np.load(f"{OUT}/counter3.npz"); M10 = np.load(f"{OUT}/machine10_counts.npz"); CB = np.load(f"{QFO}/machine_counts.npz")
loc = np.concatenate([np.arange(*MJ["rows"][T10.index(t)]) for t in T3]); glob_ = np.concatenate([np.arange(*meta["rows"][t]) for t in T3])
mk = [k for k in CM.files if not k.startswith("TOP__")]
d10 = [k for k in mk if not np.array_equal(CM[k], M10[k][loc])]; dB = [k for k in mk if not np.array_equal(CM[k], CB[k][glob_])]
print(f"plate + arrival count arrays: mine {len(mk)}, machine's {len(M10.files)}; differing from the delivered machine's run: {len(d10)}; from the builder's saved: {len(dB)}")
for k in d10[:12]: print("   ", k, CM[k][:4], M10[k][loc][:4], "sum mine", CM[k].sum(), "machine", M10[k][loc].sum())
zg = json.load(open(f"{OUT}/counter3.json"))["zg"]; print("zero gaps per plate equal:", zg == MJ["zg"])
TM = np.load(f"{OUT}/machine10_top.npz"); TB = np.load(f"{QFO}/top_machine.npz")
tloc = loc; tglob = glob_
for k in ("walk", "writing", "end", "plain_meet", "plain_div", "pours", "bound", "surv", "top5", "win"):
    a = CM[f"TOP__{k}"]; b = TM[k][tloc]; c = TB[k][tglob]
    print(f"  band read {k:10s}: equal to the delivered machine's {np.array_equal(a, b)}, to the builder's saved {np.array_equal(a, c)}" + ("" if np.array_equal(a, b) else f"  mine {a.sum()} machine {b.sum()}"))
RF = json.load(open(f"{OUT}/ref10.json"))
rt5 = sum((RF["read"][str(t)]["top5"] for t in T3), []); rw = sum((RF["read"][str(t)]["win"] for t in T3), [])
print(f"  own collapse's top 5 = the reference's {int((CM['TOP__top5'] == np.array(rt5)).all(1).sum())}/{len(rt5)}; race winner {int((CM['TOP__win'] == np.array(rw)).sum())}/{len(rw)}")

# ---- shares by part on these 54 positions, own counts ----
NP = len(loc); KN = ["q", "k", "v", "o", "gate", "up", "down"]
def S(kind, it): return int(sum(CM[f"{kind}.{l}__{it}"].sum() for l in range(L)))
def SL(it): return int(sum(CM[f"{s}.{l}__{it}"].sum() for s in ("qkv", "o", "gu", "down") for l in range(L)))
def A(it): return int(sum(CM[f"arr.{l}__{it}"].sum() for l in range(L)))
plain_pl = sum(S(k, "plain") for k in KN); gaps, pours, psd, w3, w1 = (sum(S(k, it) for k in KN) for it in ("gaps", "pours", "pours_sd", "walk3", "write1"))
wr, wr3 = SL("writing"), SL("writing3"); sig = sum(S(k, "signed") for k in KN)
top_plain = int(CM["TOP__plain_meet"].sum() + CM["TOP__plain_div"].sum())
top_c = int(CM["TOP__walk"].sum() + CM["TOP__writing"].sum() + CM["TOP__pours"].sum() + CM["TOP__bound"].sum() + CM["TOP__end"].sum())
arr_plain = A("plain_scores") + A("plain_values"); arr_d = A("sd_scores") + A("sd_values") + A("write_q") + A("write_k") + A("write_v")
pa = gaps + pours + wr; pab = gaps + psd + w3 + wr + wr3; P0 = plain_pl + top_plain + arr_plain
f = lambda v: f"{v / NP:,.0f}"
print(f"\nOWN COUNT, {NP} positions (texts {T3}), a token:")
print(f"  plain: plates {f(plain_pl)} + read {f(top_plain)} + arrival {f(arr_plain)} = {f(P0)}")
print(f"  plates (a) {pa / plain_pl * 100:.2f}% (gaps {f(gaps)}, pours {f(pours)}, writing {f(wr)}); (a)+(b) {pab / plain_pl * 100:.2f}%; signed alone {sig / plain_pl * 100:.2f}%")
print(f"  pours sd / pours {psd / pours:.4f}; 3x walk / walk {w3 / gaps:.4f}; W1 / pours {w1 / pours:.4f}")
print(f"  read (c) {top_c / top_plain * 100:.2f}% (walk {f(CM['TOP__walk'].sum())}, pours {f(CM['TOP__pours'].sum())}, bound {f(CM['TOP__bound'].sum())}); survivors after band 2 median {np.median(CM['TOP__surv'][:, 2]):.0f}, band 3 {np.median(CM['TOP__surv'][:, 3]):.0f}")
print(f"  arrival (d) {arr_d / arr_plain * 100:.2f}%")
steps = [("plain", P0), ("(a)", pa + top_plain + arr_plain), ("(a)+(c)", pa + top_c + arr_plain), ("(a)+(c)+(d)", pa + top_c + arr_d), ("all four", pab + top_c + arr_d), ("all four, W1", gaps + psd + w1 + wr + top_c + arr_d)]
for n_, v in steps: print(f"  {n_:12s} {v / P0 * 100:7.2f}%")
# control (i), own count
for k in KN:
    pl = S(k, "plain"); kk = {"q": "qkv", "k": "qkv", "v": "qkv", "o": "o", "gate": "gu", "up": "gu", "down": "down"}[k]
    print(f"  ctl {k:5s} (a) {(S(k, 'gaps') + S(k, 'pours')) / pl * 100:6.2f}%  shuffled-gap (a) {(S(k, 'gaps_shuf') + S(k, 'pours')) / pl * 100:6.2f}%  shuffled (a)+(b) {(S(k, 'gaps_shuf') + S(k, 'pours_sd') + S(k, 'walk3_shuf')) / pl * 100:6.2f}%")
