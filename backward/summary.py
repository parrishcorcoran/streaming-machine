"""The backward runs gathered into one report (reads backward/out only; nothing is recomputed).
    python3 backward/summary.py      -> backward/out/summary.log
"""
import json, glob, numpy as np
OUT = "/Users/abundancemachine/streaming-machine/backward/out"
LOG = open(f"{OUT}/summary.log", "w")
def log(s=""): print(s); LOG.write(s + "\n")
PLAIN_WHOLE = 3666596259; ARRIVAL = 13053756; REST = 4560481; GROUP = 711431068           # docs/09, refs, a token

log("STEP 1, THE TOKEN (top.py): the final light read top place first; the top 5 (set and order) proved by bounds")
for tag in ("refs", "windows", "refs_shuffled", "refs_order7"):
    try: Z = np.load(f"{OUT}/top_{tag}.npz")
    except FileNotFoundError: continue
    c5 = Z["c5"]; c1 = Z["c1"]; h = np.bincount(c5, minlength=17)
    log(f"  {tag:14s} n={len(c5)}: place c5 median {int(np.median(c5))}, mean {c5.mean():.2f}; every place (c=0) {np.mean(c5 == 0):.1%}; "
        f"by place " + " ".join(f"{c}:{h[c]}" for c in range(17) if h[c]) + f" | winner alone median {int(np.median(c1))}"
        f" | lines read median {int(np.median(Z['lanes']))} | photons read {Z['phot_read'].sum() / Z['phot_all'].sum():.2%}"
        f" | meetings of the read part with every word {Z['meet5'].sum() / Z['plain_meet'].sum():.2%}"
        + (f" | extra {Z['ext_all'].sum() / Z['plain_meet'].sum():.2%}" if "ext_all" in Z.files else ""))
    if tag == "refs_order7":
        R = np.load(f"{OUT}/top_refs.npz"); log(f"    order-free: c5 identical {int((R['c5'] == c5).sum())}/{len(c5)}, c1 identical {int((R['c1'] == c1).sum())}/{len(c1)}")

log("\nSTEPS 2-3, THE REQUIREMENT AT EACH POINT (req.py): share of positions needing the exact value; median k (box +-2^k) elsewhere")
REQ = {}
for f in sorted(glob.glob(f"{OUT}/req_*.npz")):
    Z = np.load(f); s = "windows" if "windows" in f else "refs"
    for k in Z.files: REQ[(s, k)] = Z[k]
for s in ("refs", "windows"):
    keys = sorted([k for (ss, k) in REQ if ss == s], key=lambda k: (-int(k.split("__")[0]), k))
    for k in keys:
        v = REQ[(s, k)]; ne = v >= 0
        log(f"  {s:7s} layer {k.split('__')[0]:>2s} {':'.join(k.split('__')[1:]):6s}: exact {np.mean(v < 0):6.1%}; median k {int(np.median(v[ne])) if ne.any() else '-'}")

log("\nSTEPS 2-5, THE PILES (stops.py): needed meetings, share of plain and multiple of the extra")
for f in sorted(glob.glob(f"{OUT}/stops_*.json")):
    J = json.load(open(f)); R = J["rec"]; tag = f.split("/")[-1][:-5]
    log(f"  {tag}")
    per_layer = {}
    for name, r in R.items():
        l = int(name.split(".")[1]); d = per_layer.setdefault(l, dict(plain=0, extra=0, a=0, b=0))
        for k in d: d[k] += r[k]
    tot = dict(plain=0, extra=0, a=0, b=0)
    for l in sorted(per_layer):
        d = per_layer[l]
        for k in tot: tot[k] += d[k]
        log(f"    layer {l:2d}: extra {d['extra'] / d['plain']:6.2%} | (a) {d['a'] / d['plain']:6.2%} = {d['a'] / d['extra']:5.1f}x extra | (b) {d['b'] / d['plain']:6.2%} = {d['b'] / d['extra']:5.1f}x extra")
    log(f"    banks   : extra {tot['extra'] / tot['plain']:6.2%} | (a) {tot['a'] / tot['plain']:6.2%} = {tot['a'] / tot['extra']:5.1f}x extra | (b) {tot['b'] / tot['plain']:6.2%} = {tot['b'] / tot['extra']:5.1f}x extra")
    for name, v in J.get("ver", {}).items():
        log(f"    verified {name}: (a) {v['a_same']}/{v['a_n']}" + (f"; (b) proved {v['b_proved']}/{v['b_n']}, naming other words {v['b_wrong']}" if v["b_n"] else ""))
    if tag == "stops_refs_L0-1-2-3-4-5-6-7-8-9-10-11":
        n = 648; banks_a = tot["a"] / n; banks_b = tot["b"] / n; banks_plain = tot["plain"] / n
        T = np.load(f"{OUT}/top_refs.npz"); top_plain_meet = T["plain_meet"].sum() / n
        top_need = 0.179 * top_plain_meet                       # FINDINGS sec. 29: the diagonal top read, refs: 17.9% of the meetings
        log(f"    banks plain a token {banks_plain:,.0f} (docs/09: 2,744,964,708)")
        m11 = R["mproj.11"]; banks_b1 = (tot["a"] - m11["a"] + m11["b"]) / n
        for crit, banks in (("(a)", banks_a), ("(b) mproj.11's box alone (verified)", banks_b1), ("(b) all four layer-11 boxes (joint not verified)", banks_b)):
            whole = banks + ARRIVAL + top_need + REST; whole2 = banks + ARRIVAL + 343.3e6 + REST
            log(f"    WHOLE {crit}: banks {banks:,.0f} + arrival {ARRIVAL:,} + top read's needed meetings {top_need:,.0f} (sec. 29) + rest {REST:,} "
                f"= {whole:,.0f} = {whole / PLAIN_WHOLE:.2%} of plain; with the top read's bound (sec. 29's 343.3M) {whole2 / PLAIN_WHOLE:.2%} "
                f"(group machine {GROUP / PLAIN_WHOLE:.2%})")
        ext = tot["extra"] / n + T["ext_all"].sum() / n
        log(f"    the extra of the banks and the top read: {ext:,.0f} a token = {ext / PLAIN_WHOLE:.2%} of plain; "
            f"(a) whole = {(banks_a + ARRIVAL + top_need + REST) / ext:.1f}x the extra")
