"""Independent recheck (a separate checker, 2026-10-06): the zero-sum GPT-2 files through base2.py's own machine,
every pile digested, on 5 of the 30 refs texts; and a recount of the level's work in the water level.

HIS WORDS (2026-10-06): "It's a zero sum game. If one weight goes up another goes down. A 10 that becomes a 12 is a 2
/ A 10 that becomes a 8 is -2"

Shares no code with zerosum_run.py / zerosum_way.py / zerosum_waycost.py. base2.py's own functions are taken by
exec of its source up to the tokenizer (torch lines dropped; nothing else changed), and its product() is wrapped so
every pile it makes is digested row by row (a row = a position; the arrival's score rows only up to the position
itself, the rest is never read; the arrival's value piles are one position each). The top is read at every position.
Files: plain base2_gpt2.npz; cut zerosum_gpt2_cut.npz (proj.11.b); fine zerosum_gpt2.npz (every write line's level
taken off at the format's finest grain); ctl zerosum_gpt2_ctl.npz (the same constants, half the lanes up, half down).

Also: (a) the files themselves: every changed write line differs from the plain line by one constant across the
lanes (fine, cut) and the line's level afterwards; the ctl lines' change sums to zero over the lanes;
(b) the 1s of every write array before and after (the build log's lit places); (c) the level's work in the water
level on the plain machine, all 30 texts (S1's adds = the 1s of the cut light, + ones(S1)^2 + ones(S1^2)) against
plain 3,666,596,259 a token; the pour before the cut (2 x the 1s of R + 768 x ones(sum R)); S1 = 0 counts; whether
the water level's log is the same with S1 left out.

PREDICTIONS (written before the first run), from the session's logs scaled to 5 texts:
  cut:  the first different pile in every text is fc.11's; piles differ only at fc.11, mproj.11 and the top
        (bend.11 and the stream follow); no pile at qkv.0..qkv.11, fc.0..fc.10, proj.*, mproj.0..10 differs;
        every position has some different pile (fc.11's at every position); race winner and top 5 (in order) the
        same at every position.
  fine: the first different pile is qkv.0's at every text; nearly every pile call differs; race winner the same at
        every position; top 5 in order the same at all but about 1% of positions.
  ctl:  differs at least as much as fine; race winner changes at more positions than fine (session: "top-word changes
        at least 3x fine's" was its prediction for ctl).
  files: fine and cut are line constants exactly (one whole number per line); 91,073 nonzero of 97,385 (fine), 1 (cut);
        1s of all writes 482,978,948 -> 483,452,657 (fine).
  work: level's work 98,773 photons a token (0.00269% of plain); pour 534,524 a token; S1 = 0 at 0 reads (plain);
        log unchanged without S1 at about 15% of reads (plain), about 97% (fine).
    python3 zerosum/recheck_indep_gpt2.py
"""
import sys, os, json, time, hashlib, resource, warnings
import numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")
HERE = "/Users/abundancemachine/streaming-machine"; Z = f"{HERE}/zerosum"; PLAIN = 3666596259
T0 = time.time()
def gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
def ones(X): return np.bitwise_count(np.abs(np.asarray(X, dtype=np.int64))).astype(np.int64)
def say(s=""): print(s, flush=True)

src = open(f"{HERE}/base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]
src = src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np").replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
cwd = os.getcwd(); os.chdir(HERE); argv = sys.argv; sys.argv = ["x"]; B = {}; exec(src, B); sys.argv = argv; os.chdir(cwd)
say(f"base2.py loaded ({time.time()-T0:.0f}s, {gb():.2f} GB)")
M0 = B["M"]; W = B["W"]; ES = B["ES"]
PLAIN_W = {}                                            # keep only references to the plain write plates' tuples
BANKS = [f"{b}.{l}" for l in range(12) for b in ("proj", "mproj")]
for n in BANKS: PLAIN_W[n] = W[n]
WTE0, WPE0 = M0["wte"].astype(np.int64), M0["wpe"].astype(np.int64)

# ---------------- (a), (b): the files themselves ----------------
def real_rows(w, we):   # each weight row (an input line across the 768 lanes) in units of 2^lo, exact whole numbers
    lo = int(we.min()); return w.astype(np.int64) * (np.int64(1) << (we - lo).astype(np.int64))[None, :], lo
FILES = {"fine": f"{Z}/zerosum_gpt2.npz", "cut": f"{Z}/zerosum_gpt2_cut.npz", "ctl": f"{Z}/zerosum_gpt2_ctl.npz"}
for var, path in FILES.items():
    Zf = np.load(path); nz = tot = notconst = 0; ones_b = ones_a = 0; zsum_bad = 0; maxlevel_after = 0.0
    for n in BANKS:
        for part in ("w", "b"):
            key = f"{n}.{part}"
            if part == "w":
                w0 = M0[key]; we = M0[f"{n}.e"]; tot += w0.shape[0]
                if key not in Zf.files: ones_b += int(ones(w0).sum()); ones_a += int(ones(w0).sum()); continue
                w1 = Zf[key]; ones_b += int(ones(w0).sum()); ones_a += int(ones(w1).sum())
                r0, lo = real_rows(w0, we); r1, _ = real_rows(w1, we); d = r1 - r0     # change per row in units 2^lo
                if var == "ctl":
                    zsum_bad += int((np.abs(d.sum(1)) > 0).sum()); nz += int((np.abs(d).sum(1) > 0).sum())   # change sums to 0 over the lanes?
                else:
                    const = np.all(d == d[:, :1], axis=1); notconst += int((~const).sum()); nz += int((d[:, 0] != 0).sum())
                    lev = r1.sum(1) / 768.0; maxlevel_after = max(maxlevel_after, float(np.max(np.abs(lev) / (2.0 ** (int(we.max()) - lo)))))
            else:
                b0 = M0[key]; tot += 1
                if key not in Zf.files: ones_b += int(ones(b0).sum()); ones_a += int(ones(b0).sum()); continue
                b1 = Zf[key]; d = b1.astype(np.int64) - b0.astype(np.int64); ones_b += int(ones(b0).sum()); ones_a += int(ones(b1).sum())
                if var == "ctl": zsum_bad += int(d.sum() != 0); nz += int(np.any(d != 0))
                else: notconst += int(not np.all(d == d[0])); nz += int(d[0] != 0)
    for t in ("wte", "wpe"):
        t0 = M0[t].astype(np.int64); tot += t0.shape[0]; ones_b += int(ones(t0).sum())
        if t not in Zf.files: ones_a += int(ones(t0).sum()); continue
        t1 = Zf[t].astype(np.int64); d = t1 - t0; ones_a += int(ones(t1).sum())
        if var == "ctl": zsum_bad += int((d.sum(1) != 0).sum()); nz += int((np.abs(d).sum(1) > 0).sum())
        else: notconst += int((~np.all(d == d[:, :1], axis=1)).sum()); nz += int((d[:, 0] != 0).sum())
    if var == "ctl":
        say(f"file {var}: {nz:,} of {tot:,} write lines changed; lines whose change does not sum to 0 over the lanes: {zsum_bad}; "
            f"1s of all writes {ones_b:,} -> {ones_a:,} ({100*(ones_a/ones_b-1):+.3f}%)")
    else:
        say(f"file {var}: {nz:,} of {tot:,} write lines with a nonzero constant; lines whose change is NOT one constant across the lanes: {notconst}; "
            f"weight rows' level afterwards, largest |level| / finest grain {maxlevel_after:.3f}; 1s of all writes {ones_b:,} -> {ones_a:,} ({100*(ones_a/ones_b-1):+.3f}%)")
    del Zf

# ---------------- the recorder ----------------
REC = []; STAGE = ["-"]; PCOUNT = [0]; CNT = {}
_product = B["product"]
def rowdig(a): a = np.ascontiguousarray(a); return [hashlib.blake2b(a[i].tobytes(), digest_size=8).digest() for i in range(a.shape[0])]
def product_rec(x, w):
    p = _product(x, w); st = STAGE[0]
    if st.startswith("arrival"):
        if p.shape[0] == p.shape[1] and p.shape[1] != 64:   # scores: row i up to i
            PCOUNT[0] = 0; REC.append((st, "scores", [hashlib.blake2b(np.ascontiguousarray(p[i, :i + 1]).tobytes(), digest_size=8).digest() for i in range(len(p))], None))
        else:
            REC.append((st, "values", rowdig(p), PCOUNT[0])); PCOUNT[0] += 1
    else:
        REC.append((st, "pile", rowdig(p), None))
    return p
B["product"] = product_rec
_gl = B["gauge_log"]
def gl_count(R):
    x16, ex, LS = _gl(R)
    S1, S2 = x16.sum(1), (x16 * x16).sum(1)
    CNT["reads"] = CNT.get("reads", 0) + len(R); CNT["S1=0"] = CNT.get("S1=0", 0) + int((S1 == 0).sum())
    CNT["level"] = CNT.get("level", 0) + int(ones(x16).sum() + (ones(S1) ** 2).sum() + ones(S1 * S1).sum())
    CNT["pour"] = CNT.get("pour", 0) + int(2 * ones(R).sum() + (768 * ones(R.sum(1))).sum())
    eps = np.array([B["EPS40"] >> (40 + 2 * int(e)) if 40 + 2 * int(e) >= 0 else B["EPS40"] << -(40 + 2 * int(e)) for e in ex])
    LS0 = np.right_shift(B["log2fix"](768 * S2 + eps), 1) - B["LOG768"] + ex * 65536
    CNT["log same w/o S1"] = CNT.get("log same w/o S1", 0) + int((LS0 == LS).sum())
    REC.append((STAGE[0], "cut light (level-free) + power", [hashlib.blake2b(np.ascontiguousarray(np.r_[x16[i] - x16[i, 0], ex[i]]).tobytes(), digest_size=8).digest() for i in range(len(x16))], None))
    return x16, ex, LS
B["gauge_log"] = gl_count

def forward_all(ids, wte, wpe):   # base2.py's forward, the top read at every position
    P = len(ids); R = wte[ids] + wpe[:P]
    for l in range(12):
        STAGE[0] = f"qkv.{l}"; Q = B["gauged_bank"](R, f"qkv.{l}")
        STAGE[0] = f"arrival.{l}"; A = B["attention"](Q)
        STAGE[0] = f"proj.{l}"; R = R + B["plain_bank"](A, f"proj.{l}")
        STAGE[0] = f"fc.{l}"; H = B["gauged_bank"](R, f"fc.{l}")
        STAGE[0] = f"bend.{l}"; E = B["edge"](H); REC.append((f"bend.{l}", "bend out", rowdig(E), None))
        STAGE[0] = f"mproj.{l}"; R = R + B["plain_bank"](E, f"mproj.{l}")
        REC.append((f"stream.{l}", "stream (level-free)", rowdig(R - R[:, :1]), None))
    STAGE[0] = "top"; w, we, b = W["head"]; x16, ex, LS = B["gauge_log"](R)
    S = B["power"](B["product"](x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b
    win = np.array([B["race"](S[i]) for i in range(P)])
    top5 = np.argsort(-S, axis=1, kind="stable")[:, :5]
    return win, top5

refs = json.load(open(f"{HERE}/refs_many.json"))
PICK = [2, 9, 15, 22, 28]
TEXTS = [refs[i]["ids"] + refs[i]["ref"] for i in PICK]

def run(wte, wpe):
    out = []
    for ids in TEXTS:
        REC.clear(); win, top5 = forward_all(ids, wte, wpe); out.append(dict(rec=list(REC), win=win, top5=top5))
    return out

def install(path):
    for n in BANKS: W[n] = PLAIN_W[n]
    if path is None: return WTE0, WPE0
    Zf = np.load(path)
    for n in BANKS:
        w0, we, b0 = PLAIN_W[n]
        w = Zf[n + ".w"].astype(np.float64) if n + ".w" in Zf.files else w0
        b = Zf[n + ".b"].astype(np.int64) if n + ".b" in Zf.files else b0
        W[n] = (w, we, b)
    wte = Zf["wte"].astype(np.int64) if "wte" in Zf.files else WTE0
    wpe = Zf["wpe"].astype(np.int64) if "wpe" in Zf.files else WPE0
    return wte, wpe

def compare(base, new, lab):
    npos = sum(len(x["win"]) for x in base); pos_same = 0; piles = piles_diff = 0; first = []; diff_stages = {}
    win_same = top5_same = top5_set = 0
    for b, n in zip(base, new):
        P = len(b["win"]); bad = np.zeros(P, bool); f = None
        assert len(b["rec"]) == len(n["rec"])
        for (st, kind, hb, pc), (st2, kind2, hn, pc2) in zip(b["rec"], n["rec"]):
            assert (st, kind) == (st2, kind2)
            d = np.array([x != y for x, y in zip(hb, hn)])
            is_pile = kind in ("pile", "scores", "values")
            if is_pile: piles += 1; piles_diff += int(d.any())
            if d.any():
                diff_stages[(st, kind)] = diff_stages.get((st, kind), 0) + 1
                if f is None and is_pile: f = f"{st} {kind}"
                if kind == "values": bad[pc % P] = True
                else: bad |= d
        first.append(f); pos_same += int((~bad).sum())
        win_same += int((b["win"] == n["win"]).sum()); top5_same += int(np.all(b["top5"] == n["top5"], axis=1).sum())
        top5_set += int(sum(set(x) == set(y) for x, y in zip(b["top5"], n["top5"])))
    say(f"\n{lab} AGAINST PLAIN, {len(TEXTS)} texts, {npos} positions")
    say(f"  pile calls {piles:,}: different {piles_diff:,}; positions with every digest identical {pos_same}/{npos}")
    say(f"  first different pile per text: {first}")
    say(f"  race winner same {win_same}/{npos}; top 5 same in order {top5_same}/{npos}, as a set {top5_set}/{npos}")
    stg = {}
    for (st, kind), v in diff_stages.items(): stg.setdefault(st.split('.')[0] if st != 'top' else 'top', set()).add(st)
    say("  digests that differ, by stage and kind: " + ", ".join(f"{st} {kind} x{v}" for (st, kind), v in list(diff_stages.items())[:40]) + (" ..." if len(diff_stages) > 40 else ""))
    return dict(npos=npos, piles=piles, piles_diff=piles_diff, pos_same=pos_same, first=first, win_same=win_same, top5_same=top5_same, top5_set=top5_set,
                stages=sorted({st for (st, _) in diff_stages}))

RES = {}
CNT.clear(); wte, wpe = install(None); base = run(wte, wpe); c_plain = dict(CNT)
say(f"plain run ({time.time()-T0:.0f}s, {gb():.2f} GB): reads {c_plain['reads']}, S1 = 0 at {c_plain['S1=0']}, log same w/o S1 {c_plain['log same w/o S1']} "
    f"({100*c_plain['log same w/o S1']/c_plain['reads']:.1f}%)")
for var in ("cut", "fine", "ctl"):
    CNT.clear(); wte, wpe = install(FILES[var]); new = run(wte, wpe); c = dict(CNT)
    RES[var] = compare(base, new, var); RES[var]["counts"] = c
    say(f"  reads {c['reads']}, S1 = 0 at {c['S1=0']}, log same w/o S1 {c['log same w/o S1']} ({100*c['log same w/o S1']/c['reads']:.1f}%)  ({time.time()-T0:.0f}s, {gb():.2f} GB)")
    del new

# ---------------- (c) the level's work, plain machine, all 30 texts ----------------
install(None); CNT.clear(); n = 0
for r in refs:
    ids = r["ids"] + r["ref"]; n += len(ids); REC.clear(); forward_all(ids, WTE0, WPE0)
REC.clear()
lv, po = CNT["level"] / n, CNT["pour"] / n
say(f"\nlevel's work, plain machine, 30 texts, {n} positions, {CNT['reads']} reads: {lv:,.0f} photons a token = {100*lv/PLAIN:.5f}% of plain; "
    f"pour before the cut {po:,.0f} a token ({100*po/PLAIN:.5f}%, {po/lv:.1f} x); S1 = 0 at {CNT['S1=0']}; log same w/o S1 {CNT['log same w/o S1']} ({100*CNT['log same w/o S1']/CNT['reads']:.1f}%)")
RES["work"] = dict(positions=n, level=lv, pour=po, counts=dict(CNT), plain_5=c_plain)
json.dump(RES, open(f"{Z}/recheck_indep_gpt2.json", "w"), indent=1, default=str)
say(f"done ({time.time()-T0:.0f}s, peak {gb():.2f} GB)")
