"""reach/answer_report.py: the table of (2) from reach/answer/<plate>_b1.json (no machine runs here).
    python3 reach/answer_report.py      -> reach/answer_report.log
"""
import json, os, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; D = f"{HERE}/reach/answer"
LOG = open(f"{HERE}/reach/answer_report.log", "w")
def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
BANKS = [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")]
BT = np.load(f"{HERE}/reach/base_top.npz"); S6 = BT["s6"][480:]; GAP = (S6[:, :5] - S6[:, 1:]).min(1)
log(f"the 6,000 judge positions' smallest gap (of the five that keep the top 5 and its order): min {GAP.min()}, "
    f"median {int(np.median(GAP))}; below 16: {(GAP < 16).sum()}, below 256: {(GAP < 256).sum()}, below 4,096: {(GAP < 4096).sum()}")
log(f"{'plate':9s} {'place-0 digits':>14s} {'proven':>9s} {'blown':>6s} {'bound median':>13s} {'bound max':>12s} {'real median':>11s} "
    f"{'real max':>9s} {'top5 changed':>12s} {'bound/real':>10s}")
tot = dict(digits=0, proven_plates=0); rows = []
for p in ["head"] + BANKS[::-1]:
    f = f"{D}/{p}_b1.json"
    if not os.path.exists(f): log(f"{p:9s} (not run)"); continue
    r = json.load(open(f)); rows.append(r)
    br = "-" if not r["bound_over_real_median"] else f"{r['bound_over_real_median']:.3g}"
    log(f"{p:9s} {r['digits_removed']:>14,} {r['proven']:>5}/{r['rows']} {r['blown_at_top']:>6} {str(r['bound_median']):>13s} "
        f"{str(r['bound_max']):>12s} {r['real_median']:>11} {r['real_max']:>9} {r['top5_changed']:>12} "
        f"{br:>10s}")
    tot["digits"] += r["digits_removed"]; tot["proven_plates"] += int(r["proven"] == r["rows"])
log(f"\nplates whose place-0 digits are PROVEN removable at all 6,000 positions: {tot['proven_plates']} of {len(rows)}")
log("\nLAYER BY LAYER (the stream's range after each layer, largest half-width over rows not blown; rows blown)")
for r in rows:
    st = [x for x in r["layers"] if x["where"].startswith("stream ") and not x["where"].startswith("stream a")]
    if not st: continue
    log(f"  {r['plate']:9s} " + "  ".join(f"L{x['where'].split()[-1]}: {x.get('range_max', 'inf')}/{x['blown_rows']}b real {x['real_max']}" for x in st))
