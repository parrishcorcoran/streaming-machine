"""THE REPORT: every measured share at Qwen3-0.6B's plates and word table, beside GPT-2's, and two ESTIMATES (labelled):
the whole Qwen3-0.6B machine with these cuts, and what the weight-only numbers predict at a 14B model's widths.
Reads only qwen06b/'s own results (banks_*.json, table_*.json, top_read_*.json, weights_only.json, light/meta.json).
    python3 qwen06b/report.py
"""
import json, glob, math, numpy as np
from fractions import Fraction
OUT = "/Users/abundancemachine/streaming-machine/qwen06b"
def coin(N): return float(Fraction(math.comb(N, N // 2), 2 ** N))
R = {}
for f in sorted(glob.glob(f"{OUT}/banks_*_*.json")): R.update(json.load(open(f)))
TW, TU = json.load(open(f"{OUT}/table_walk.json")), json.load(open(f"{OUT}/table_updown.json"))
TR = {m: json.load(open(f"{OUT}/top_read_{m}.json")) for m in ("main", "shuffle", "wordperm")}
WO = json.load(open(f"{OUT}/weights_only.json")); META = json.load(open(f"{OUT}/light/meta.json")); NP = META["npos"]
KINDS = ["q", "k", "v", "o", "gate", "up", "down"]; SLAB = dict(q="qkv", k="qkv", v="qkv", o="o", gate="gu", up="gu", down="down")
GPT2 = {"proj (768)": (9.85, 10.81, 35.12, 57.60, 58.85), "mproj (768)": (9.23, 10.39, 32.39, 59.59, 60.59), "qkv (2,304)": (7.03, 10.32, 32.08, 58.21, 54.06),
        "fc (3,072)": (6.37, 10.26, 31.68, 58.32, 52.51), "table (50,257)": (0.283, 8.57, 23.25, 60.75, 38.07)}   # lineup2/indep_check.log
S = {}
for k in KINDS:
    rows = [R[f"{k}.{l}"] for l in range(28)]; s = {key: sum(r.get(key, 0) for r in rows) for key in rows[0] if isinstance(rows[0][key], (int, float)) and not isinstance(rows[0][key], bool)}
    s["N"], s["out"] = rows[0]["N"], rows[0]["out"]
    s["exact"] = dict(slab_digests=all(r["slab_digest_ok"] for r in rows), ud=sum(r["ud_piles_differ"] for r in rows), ctl_ud=sum(r["ctl_ud_piles_differ"] for r in rows),
                      signed=sum(r["signed_piles_differ"] for r in rows), walk=sum(r["walk_piles_differ"] for r in rows), ctl_walk=sum(r["ctl_walk_piles_differ"] for r in rows),
                      of_walk=sum(r.get("orderfree_walk_piles_differ", 0) for r in rows), of_open=sum(r.get("orderfree_open", 0) for r in rows),
                      of_pours=all(r.get("orderfree_pours_equal", True) for r in rows), of_ud=all(r.get("orderfree_ud_same", True) for r in rows),
                      of_layers_walk=sum("orderfree_walk_piles_differ" in r for r in rows), of_layers_ud=sum("orderfree_ud_same" in r for r in rows))
    S[k] = s
t = dict(TW); t.update({k: v for k, v in TU.items()}); t["N"], t["out"] = 1024, 151936; S["table"] = t
TOP = TR["main"]; top_plain = sum(v for kk, v in TOP["count"].items() if kk.startswith("plain|")); top_cut = sum(v for kk, v in TOP["count"].items() if kk.startswith("cut|"))
ARR = META["arrival_scores_total"] + META["arrival_values_total"]

print(f"QWEN3-0.6B, ALL BASE 2 (qwen/qwen_base2.py's own light), 30 refs texts in Qwen's words, {NP} positions; work a token (photon meetings)\n")
print("1-3. EVERY PLATE KIND (28 layers each), shares of the plain meetings of the same light")
print(f"  {'kind':6s} {'lines':>6s} {'n/line':>7s} {'plain a token':>14s} | {'straight-through':>16s} {'control':>8s} {'coin':>8s} | {'signed':>7s} {'ctl':>7s} | {'walk/pr':>7s} {'pour/pr':>7s} {'plain/pr':>8s} {'way 2':>7s} {'ctl':>7s} {'order shuffled':>14s}")
for k in KINDS + ["table"]:
    s = S[k]; N = s["N"]; st = 1 - s["ud_fringe"] / (N * s["ud_meetings"]); stc = 1 - s["ctl_ud_fringe"] / (N * s["ctl_ud_meetings"])
    w2 = (s["gaps"] + s["pours"] + s["writing"]) / s["plain"]; w2c = (s["ctl_gaps"] + s["ctl_pours"] + s["writing"]) / s["ctl_plain"]
    w2s = (s["gaps_shuffled"] + s["pours"] + s["writing"]) / s["plain"]
    print(f"  {k:6s} {N:6d} {s['out']:7d} {s['plain'] / NP:14,.0f} | {st * 100:15.3f}% {stc * 100:7.3f}% {(1 - coin(N)) * 100:7.3f}% | {s['signed'] / s['plain'] * 100:6.2f}% {s['ctl_signed'] / s['ctl_plain'] * 100:6.2f}% | "
          f"{s['gaps'] / s['pairs']:7.3f} {s['pours'] / s['pairs']:7.2f} {s['plain'] / s['pairs']:8.2f} {w2 * 100:6.2f}% {w2c * 100:6.2f}% {w2s * 100:13.2f}%")
    s.update(st=st, stc=stc, w2=w2, w2c=w2c, w2s=w2s, sg=s["signed"] / s["plain"])
print("  fringe share against the coin-toss value (real / control): " + ", ".join(f"{k} {(1 - S[k]['st']) / coin(S[k]['N']):.3f}/{(1 - S[k]['stc']) / coin(S[k]['N']):.3f}" for k in KINDS + ["table"]))
banks_plain = sum(S[k]["plain"] for k in KINDS); print(f"  all 196 plates: plain {banks_plain / NP:,.0f} a token; signed digits {sum(S[k]['signed'] for k in KINDS) / banks_plain * 100:.2f}%; "
      f"way 2 {sum(S[k]['gaps'] + S[k]['pours'] + S[k]['writing'] for k in KINDS) / banks_plain * 100:.2f}% (gap meetings {sum(S[k]['gaps'] for k in KINDS) / banks_plain * 100:.2f}% + pours {sum(S[k]['pours'] for k in KINDS) / banks_plain * 100:.2f}%)")
print("\n  EXACT (every position, every plate): " + "; ".join(f"{k}: up/down {S[k]['exact']['ud']}+{S[k]['exact']['ctl_ud']}, signed {S[k]['exact']['signed']}, walk {S[k]['exact']['walk']}+{S[k]['exact']['ctl_walk']}, "
      f"order-free walk {S[k]['exact']['of_walk']} ({S[k]['exact']['of_layers_walk']} layers, open {S[k]['exact']['of_open']}, pours same {S[k]['exact']['of_pours']}), order-free up/down same {S[k]['exact']['of_ud']} ({S[k]['exact']['of_layers_ud']} layers), slabs = capture {S[k]['exact']['slab_digests']}" for k in KINDS))
print(f"  the table: up/down {TU['ud_piles_differ']}+{TU['ctl_ud_piles_differ']}, walk {TW['walk_piles_differ']}+{TW['ctl_walk_piles_differ']}, walk down {TW['orderfree_walk_piles_differ']}/{TW['orderfree_positions']} (pours same {TW['orderfree_pours_equal']})")

print("\n3. WAY 2 PER WIDTH, beside GPT-2 (neurons a line: walk / pour / plain per pair, signed digits, way 2)")
for name, (wk, pr, pl, sg, w2) in GPT2.items(): print(f"  GPT-2 {name:16s} {wk:7.3f} {pr:6.2f} {pl:6.2f}  {sg:6.2f}%  {w2:6.2f}%")
for k in KINDS + ["table"]:
    s = S[k]; print(f"  Qwen  {k + ' (' + format(s['out'], ',') + ')':16s} {s['gaps'] / s['pairs']:7.3f} {s['pours'] / s['pairs']:6.2f} {s['plain'] / s['pairs']:6.2f}  {s['sg'] * 100:6.2f}%  {s['w2'] * 100:6.2f}%")

print("\n4. THE TOP READ (151,936 words), every position")
sh, wp = TR["shuffle"], TR["wordperm"]
def share(T_): return sum(v for kk, v in T_["count"].items() if kk.startswith("cut|")) / sum(v for kk, v in T_["count"].items() if kk.startswith("plain|"))
print(f"  exact: top 5 {TOP['ok5']}/{TOP['n']}, race winner {TOP['okw']}/{TOP['n']}; checks {sum(TOP['bad'].values())} failures")
print(f"  plain read {top_plain / NP:,.0f} a position; the cut {top_cut / NP:,.0f} = {top_cut / top_plain * 100:.2f}% (meetings alone {TOP['count']['cut|meetings, signed digits, staged'] / TOP['count']['plain|meetings, plain'] * 100:.2f}% of the plain meetings); GPT-2: 16.05% (14.47%)")
print(f"  control, light shuffled across lines (every 4th position): {share(sh) * 100:.2f}%, top 5 {sh['ok5']}/{sh['n']}; the real light at the same positions (word-order run): {share(wp) * 100:.2f}%; order-free top 5 {wp['ok5']}/{wp['n']}")
print(f"  survivors (median) after a+b>=17, 15, 13: real {TOP['surv_median'][5]:.0f}, {TOP['surv_median'][6]:.0f}, {TOP['surv_median'][7]:.0f}; shuffled {sh['surv_median'][5]:.0f}, {sh['surv_median'][6]:.0f}, {sh['surv_median'][7]:.0f} (GPT-2: 50,257, 31,291, 200)")

print("\n5. WEIGHT-ONLY (per weight; way 2: each line's weights lined up across its neurons; way 1: each neuron's down its lines)")
print(f"  {'':22s} {'1s':>6s} {'signed':>7s} {'sgn/1s':>7s} {'dark':>7s} {'way2 lit':>9s} {'zero gaps':>9s} {'way1 lit':>9s}")
for k in KINDS + ["table"]:
    s = S[k]; nw = s["w_count"]; ng = nw - s["N"] * (28 if k != "table" else 1)
    print(f"  Qwen {k:6s} {s['N']:5d}x{s['out']:<7d} {s['w_ones'] / nw:6.3f} {s['w_naf'] / nw:7.3f} {s['w_naf'] / s['w_ones']:7.3f} {s['w_dark'] / nw * 100:6.2f}% {s['w_gap_lit'] / nw:9.4f} {s['w_zero_gaps'] / ng * 100:8.2f}% {s['w1_gap_lit'] / nw:9.4f}")
for k, g in WO["gpt2"].items():
    print(f"  GPT-2 {k:5s} {g['shape'][0]:5d}x{g['shape'][1]:<7d} {g['ones']:6.3f} {g['naf']:7.3f} {g['naf'] / g['ones']:7.3f} {g['dark'] * 100:6.2f}% {g['way2_lit']:9.4f} {g['way2_zero'] * 100:8.2f}% {g['way1_lit']:9.4f}")
print("  the relation: walk per pair = (light's signed digits per lit value) x (gap lit digits per weight):")
for k in KINDS:
    s = S[k]; lt = WO["light"][SLAB[k]]["signed_per_lit"]; pred = lt * s["w_gap_lit"] / s["w_count"]
    print(f"    {k:5s} measured {s['gaps'] / s['pairs']:.3f}  light {lt:.3f} x gaps {s['w_gap_lit'] / s['w_count']:.4f} = {pred:.3f} ({(pred / (s['gaps'] / s['pairs']) - 1) * 100:+.1f}%)")

# ---------------- ESTIMATES ----------------
print("\nESTIMATE 1 (not a built machine): the whole Qwen3-0.6B base-2 machine with these cuts, from the measured parts")
plain_total = banks_plain + top_plain + ARR
sg_banks = sum(S[k]["signed"] for k in KINDS); w2_banks = sum(S[k]["gaps"] + S[k]["pours"] + S[k]["writing"] for k in KINDS)
print(f"  plain a token: plates {banks_plain / NP:,.0f} ({banks_plain / plain_total * 100:.1f}%), top read {top_plain / NP:,.0f} ({top_plain / plain_total * 100:.1f}%), "
      f"arrival (q.k and shares x values, plain) {ARR / NP:,.0f} ({ARR / plain_total * 100:.2f}%); total {plain_total / NP:,.0f}. Not counted: water levels, turns, bends, the plates' divides (GPT-2: 0.12%).")
for name, banks_cut in (("signed digits at every plate", sg_banks), ("way 2 at every plate", w2_banks)):
    tot = banks_cut + top_cut + ARR
    print(f"  {name:30s} + the collapsing top read, the arrival unchanged: {tot / NP:,.0f} a token = {tot / plain_total * 100:.2f}% of plain")
print("  per plate kind, the lower of the two measured cuts: " + ", ".join(f"{k} {'way 2' if S[k]['w2'] < S[k]['sg'] else 'signed'} {min(S[k]['w2'], S[k]['sg']) * 100:.1f}%" for k in KINDS))

print("\nESTIMATE 2 (weight-only prediction, no 14B weights here): a 14B model's widths (5,120 hidden, 17,408 MLP, 40 query and")
print("  8 key/value heads of 128, 40 layers, 151,936 words). Neurons a line: q 5,120, k 1,024, v 1,024, o 5,120, gate and up 17,408,")
print("  down 5,120, table 151,936. Lines: 5,120 everywhere but down (17,408).")
TBL = {int(k_): v["lit_per_weight"] for k_, v in TW["subsample"].items()}
def law_ratio(a, b): return TBL[b] / TBL[a]
P = WO["proxy"]
gap14 = {}
gap14["q"] = (S["gate"]["w_gap_lit"] / S["gate"]["w_count"] * law_ratio(3072, 5120),          # q's law is gate's (2,048: 1.171 / 1.173)
              float(np.mean([r["target"][0] for r in P["q"]["reps"]])))
for k in ("gate", "up"): gap14[k] = (S[k]["w_gap_lit"] / S[k]["w_count"] * law_ratio(3072, 17408), float(np.mean([r["target"][0] for r in P[k]["reps"]])))
for k in ("o", "down"): gap14[k] = (S[k]["w_gap_lit"] / S[k]["w_count"] * law_ratio(1024, 5120), float(np.mean([r["target"][0] for r in P[k]["reps"]])))
for k in ("k", "v"): gap14[k] = (S[k]["w_gap_lit"] / S[k]["w_count"],) * 2
gap14["table"] = (S["table"]["w_gap_lit"] / S["table"]["w_count"],) * 2
lines14 = dict(q=5120, k=5120, v=5120, o=5120, gate=5120, up=5120, down=17408, table=5120); nl14 = dict(q=5120, k=1024, v=1024, o=5120, gate=17408, up=17408, down=5120, table=151936)
layers14 = 40
print(f"  {'kind':6s} {'gap lit/w now':>13s} {'14B: law':>9s} {'14B: pooled (upper)':>20s} {'walk/pair':>12s} {'way 2':>14s} {'signed':>7s} {'straight-through':>17s}")
tot_plain = tot_sg = tot_w2lo = tot_w2hi = 0
for k in KINDS + ["table"]:
    s = S[k]; lt = WO["light"][SLAB[k]]["signed_per_lit"] if k != "table" else s["gaps"] / s["pairs"] / (s["w_gap_lit"] / s["w_count"])
    g_lo, g_hi = gap14[k]; wk_lo, wk_hi = lt * g_lo, lt * g_hi
    pour, plain_pp = s["pours"] / s["pairs"], s["plain"] / s["pairs"]; wr = s["writing"] / s["pairs"] * s["out"] / nl14[k]
    w2_lo, w2_hi = (wk_lo + pour + wr) / plain_pp, (wk_hi + pour + wr) / plain_pp
    lit = s["pairs"] / (NP * s["N"] * s["out"] * (28 if k != "table" else 1))     # lit share of the light's values
    pairs14 = lines14[k] * nl14[k] * lit * (layers14 if k != "table" else 1); plain14 = pairs14 * plain_pp
    if k == "table":
        sh_top = top_cut / top_plain; tot_plain += plain14; tot_sg += plain14 * sh_top; tot_w2lo += plain14 * sh_top; tot_w2hi += plain14 * sh_top
    else:
        tot_plain += plain14; tot_sg += plain14 * s["sg"]; tot_w2lo += plain14 * min(w2_lo, w2_hi); tot_w2hi += plain14 * max(w2_lo, w2_hi)
    print(f"  {k:6s} {s['w_gap_lit'] / s['w_count']:13.4f} {g_lo:9.4f} {g_hi:20.4f} {wk_lo:5.2f}-{wk_hi:<5.2f} {min(w2_lo, w2_hi) * 100:5.1f}-{max(w2_lo, w2_hi) * 100:5.1f}% {s['sg'] * 100:6.2f}% {(1 - coin(lines14[k])) * 100:16.3f}%")
print(f"  plain a token at 14B (Qwen3-0.6B's per-pair numbers): {tot_plain / 1e9:.1f}G photon meetings; the top read {S['table']['plain'] / NP / 1e9 * 5:.1f}G of it (5 x 0.6B's lines)")
print(f"  14B with signed digits at every plate + the top read at 0.6B's share: {tot_sg / tot_plain * 100:.1f}%; with way 2 at every plate: {tot_w2lo / tot_plain * 100:.1f}-{tot_w2hi / tot_plain * 100:.1f}%")
json.dump({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (dict, list))} for k, v in S.items()}, open(f"{OUT}/report.json", "w"))
