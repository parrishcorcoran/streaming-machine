"""Checker: the report's numbers re-added from the builder's saved arrays (613 positions), with the checker's own sums."""
import json, glob, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; Q = f"{HERE}/qwen_fringes"; L = 28
C = np.load(f"{Q}/machine_counts.npz"); T = np.load(f"{Q}/top_machine.npz"); TC = np.load(f"{Q}/top_control.npz")
NP = len(T["walk"]); KN = ["q", "k", "v", "o", "gate", "up", "down"]
def S(kind, it): return int(sum(int(C[f"{kind}.{l}__{it}"].sum()) for l in range(L)))
def SL(it): return int(sum(int(C[f"{s}.{l}__{it}"].sum()) for s in ("qkv", "o", "gu", "down") for l in range(L)))
def A(it): return int(sum(int(C[f"arr.{l}__{it}"].sum()) for l in range(L)))
plain_pl = sum(S(k, "plain") for k in KN); gaps, pours, psd, w3, w1, sig = (sum(S(k, it) for k in KN) for it in ("gaps", "pours", "pours_sd", "walk3", "write1", "signed"))
wr, wr3 = SL("writing"), SL("writing3")
def tot(o): return o["walk"].sum() + o["writing"].sum() + o["pours"].sum() + o["bound"].sum() + o["end"].sum()
def pl(o): return o["plain_meet"].sum() + o["plain_div"].sum()
top_plain, top_c = int(pl(T)), int(tot(T)); arr_plain = A("plain_scores") + A("plain_values"); arr_d = A("sd_scores") + A("sd_values") + A("write_q") + A("write_k") + A("write_v")
pa = gaps + pours + wr; pab = gaps + psd + w3 + wr + wr3; P0 = plain_pl + top_plain + arr_plain
f = lambda v: f"{v / NP:,.1f}"
print(f"{NP} positions. plain a token: plates {f(plain_pl)}, read {f(top_plain)}, arrival {f(arr_plain)}, all {f(P0)}")
print(f"plates (a) {f(pa)} = {pa / plain_pl * 100:.2f}%: gaps {f(gaps)} pours {f(pours)} writing {f(wr)}")
print(f"plates (a)+(b) {f(pab)} = {pab / plain_pl * 100:.2f}%: pours sd {f(psd)} 3x walk {f(w3)} writing {f(wr + wr3)}; signed alone {sig / plain_pl * 100:.2f}%")
print(f"ratios: sd pours/pours {psd / pours:.4f}  3x walk/walk {w3 / gaps:.4f}  W1/pours {w1 / pours:.4f}  saving {f(pours - psd)}  (b) net {f(pab - pa)}")
print(f"read (c) {f(top_c)} = {top_c / top_plain * 100:.2f}%: walk {f(T['walk'].sum())} pours {f(T['pours'].sum())} bound {f(T['bound'].sum())} rest {f(T['writing'].sum() + T['end'].sum())}; every band into every word {f(T['hall'].sum())}")
print(f"  survivors median after bands 0..6: {[int(np.median(T['surv'][:, b])) for b in range(7)]}; least after band 2 {T['surv'][:, 2].min()}")
print(f"arrival (d) {f(arr_d)} = {arr_d / arr_plain * 100:.2f}%: scores {f(A('sd_scores'))} (plain {f(A('plain_scores'))}), values {f(A('sd_values'))} (plain {f(A('plain_values'))}), writing {f(A('write_q') + A('write_k') + A('write_v'))}")
for n_, v in [("(a)", pa + top_plain + arr_plain), ("(a)+(c)", pa + top_c + arr_plain), ("(a)+(c)+(d)", pa + top_c + arr_d), ("all four", pab + top_c + arr_d), ("all four W1", gaps + psd + w1 + wr + top_c + arr_d)]:
    print(f"  {n_:12s} {f(v)}  {v / P0 * 100:.2f}%")
BK = {}
for fn in sorted(glob.glob(f"{HERE}/qwen06b/banks_*_*.json")): BK.update(json.load(open(fn)))
SLB = dict(q="qkv", k="qkv", v="qkv", o="o", gate="gu", up="gu", down="down"); NK = dict(qkv=3, o=1, gu=2, down=1)
for k in KN:
    pairs = sum(BK[f"{k}.{l}"]["pairs"] for l in range(L)); p_ = S(k, "plain")
    wrk = sum(int(C[f"{SLB[k]}.{l}__writing"].sum()) for l in range(L)) / NK[SLB[k]]; wr3k = sum(int(C[f"{SLB[k]}.{l}__writing3"].sum()) for l in range(L)) / NK[SLB[k]]
    a_ = S(k, "gaps") + S(k, "pours") + wrk; ab = S(k, "gaps") + S(k, "pours_sd") + S(k, "walk3") + wrk + wr3k
    print(f"  {k:5s} walk/pr {S(k, 'gaps') / pairs:.2f} 3x/pr {S(k, 'walk3') / pairs:.2f} pour/pr {S(k, 'pours') / pairs:.2f} sd/pr {S(k, 'pours_sd') / pairs:.2f} net/pr {(ab - a_) / pairs:+.2f} | ctl (a) {(S(k, 'gaps_shuf') + S(k, 'pours') + wrk) / p_ * 100:.1f}% (a)+(b) {(S(k, 'gaps_shuf') + S(k, 'pours_sd') + S(k, 'walk3_shuf') + wrk + wr3k) / p_ * 100:.1f}%")
idx = TC["sel"]; Tr = {k: T[k][idx] for k in T.files if k != "sel"}
print(f"control (ii): real {tot(Tr) / pl(Tr) * 100:.2f}% vs shuffled {tot(TC) / pl(TC) * 100:.2f}% at {len(idx)} positions; after band 2 median real {np.median(Tr['surv'][:, 2]):.0f} shuffled {np.median(TC['surv'][:, 2]):.0f}")
# the over-3GB first run vs the repeated run
A1, A2 = np.load(f"{Q}/top_machine_over3GB.npz"), np.load(f"{Q}/top_machine.npz"); J1, J2 = json.load(open(f"{Q}/top_machine_over3GB.json")), json.load(open(f"{Q}/top_machine.json"))
print("first band-read run (4.17 GB) vs repeat: arrays equal", all(np.array_equal(A1[k], A2[k]) for k in A2.files), "json equal", J1 == J2)
