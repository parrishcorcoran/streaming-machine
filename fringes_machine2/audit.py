"""THE SAVED RESULTS SET SIDE BY SIDE (reads only saved files, runs no machine):
  - ADD=0 is base2_fringes.py: its top read's counts equal fringes_machine/top_refs.json key by key, and the stream
    counts at ADD=1 (the same stream path) equal fringes_machine/refs_counts.json key by key;
  - the order-free top reads (ADD=1, 2): every count equal to the run in the machine's own order, and the one position
    whose top 5 differs is an exact tie at 5th place (the two words' plain scores read from base2's own head);
  - ADD=4's own count (fr) equals ADD=3's key by key (the rule kept every plate in signed digits);
  - the light at the top from the ADD=1, 3 and 4 runs equals base2.py's own saved light (refs and windows).
    python3 fringes_machine2/audit.py   -> fringes_machine2/audit.log
"""
import sys, os, json, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/fringes_machine2"; OLD = f"{HERE}/fringes_machine"
LOG = open(f"{OUT}/audit.log", "w")
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def J(p): return json.load(open(p))
def same_counts(a, b, who=None):
    ka = {k for k in a if who is None or k.split("|")[0] in who}; kb = {k for k in b if who is None or k.split("|")[0] in who}
    return ka == kb and all(a[k] == b[k] for k in ka), len(ka), sorted(ka ^ kb)[:5], [k for k in ka & kb if a[k] != b[k]][:5]
r = same_counts(J(f"{OUT}/top_a0_refs.json")["counts"], J(f"{OLD}/top_refs.json")["counts"])
log(f"ADD=0 top read counts = base2_fringes.py's, key by key: {r[0]} ({r[1]} keys; only in one {r[2]}; differing {r[3]})")
r = same_counts(J(f"{OUT}/refs_counts_a1.json")["counts"], J(f"{OLD}/refs_counts.json")["counts"])
log(f"ADD=1 stream counts = base2_fringes.py's, key by key: {r[0]} ({r[1]} keys; only in one {r[2]}; differing {r[3]})")
for a in (1, 2):
    r = same_counts(J(f"{OUT}/top_a{a}_refs_order7.json")["counts"], J(f"{OUT}/top_a{a}_refs.json")["counts"])
    log(f"ADD={a} order-free top read counts = the run's own order, key by key: {r[0]} ({r[1]} keys; differing {r[3]})")
    z, zo = np.load(f"{OUT}/top_a{a}_refs.npz"), np.load(f"{OUT}/top_a{a}_refs_order7.npz")
    same = {k: bool(np.array_equal(z[k], zo[k])) for k in ("win", "surv", "work", "start")}
    dif = np.flatnonzero((z["top5"] != zo["top5"]).any(1))
    log(f"  same race winner, survivors, meetings, start at every position: {same}; top 5 differs at positions {dif.tolist()}")
    for p in dif:
        log(f"    position {p}: in order {z['top5'][p].tolist()}, order-free {zo['top5'][p].tolist()}")
# the tie: the plain scores of the words involved, from base2's own head and saved light
if True:
    _src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
    _src = _src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np").replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
    _src = _src.replace('[f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]', '[]').replace('M = np.load("base2_gpt2.npz")', f'M = np.load("{HERE}/base2_gpt2.npz")')
    B = {}; _argv = sys.argv; sys.argv = ["x"]; exec(_src, B); sys.argv = _argv; M = B["M"]
    REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz"); HW = np.asarray(M["head.w"]).astype(np.int64)
    for p in np.flatnonzero((np.load(f"{OUT}/top_a1_refs.npz")["top5"] != np.load(f"{OUT}/top_a1_refs_order7.npz")["top5"]).any(1)):
        x, ex, LS = REF["x16"][p].astype(np.int64), int(REF["ex"][p]), int(REF["LS"][p])
        ws = sorted(set(np.load(f"{OUT}/top_a1_refs.npz")["top5"][p].tolist()) | set(np.load(f"{OUT}/top_a1_refs_order7.npz")["top5"][p].tolist()))
        sc = {w: int(B["power"](np.array([int(x @ HW[:, w])]), (int(M["head.e"][w]) + ex + B["ES"]) * 65536 - LS)[0] + int(M["head.b"][w])) for w in ws}
        log(f"    position {p}: the words' plain scores {sc}")
a3, a4 = J(f"{OUT}/refs_counts_a3.json")["counts"], J(f"{OUT}/refs_counts_a4.json")["counts"]
r = same_counts(a3, a4, who=("plain", "fr", "cmp", "reads"))
log(f"ADD=4 count (plain, fr, compares, reads) = ADD=3's, key by key: {r[0]} ({r[1]} keys; differing {r[3]})")
P = np.load(f"{OLD}/plain_windows_top_light.npz")
for a in (1, 3, 4):
    z = np.load(f"{OUT}/refs_top_light_a{a}.npz")
    ok = all(np.array_equal(z[k], REF[k][z["rows"]]) for k in ("x16", "ex", "LS"))
    msg = f"ADD={a}: the light at the top, refs, = base2.py's saved light: {ok}"
    if os.path.exists(f"{OUT}/windows_top_light_a{a}.npz"):
        w = np.load(f"{OUT}/windows_top_light_a{a}.npz"); msg += f"; windows: {all(np.array_equal(w[k], P[k]) for k in ('x16', 'ex', 'LS'))}"
    log(msg)
