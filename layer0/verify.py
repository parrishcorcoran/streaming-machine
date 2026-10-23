"""VERIFY base2_layer0.py AGAINST base2.py, every pile and every stream.

  A  the 30 refs texts (ids + ref, 648 positions)    B  the 300 judge windows (7,200 positions)
     For every position: qkv.0's pile and q/k/v, the arrival's output o, proj.0's pile and output, the light R1, fc.0's
     pile, its value H, the bend's output, mproj.0's pile, the stream after layer 0; then the stream after every half
     layer of layers 1-11 and the top's whole-number scores and word at every position. Against base2.py's own layer 0
     run with its own functions (judge5), nothing read from any table. (At position 0 the new machine forms no pile
     past qkv.0: it reads the stream after layer 0 whole; that position is checked on the stream and the top.)
  J  judge5.judge(forward=base2_layer0.scores_all, n=300): in the top 5, and the same top word.
  C  the stored rows: every row of Z0 against base2.py's layer 0 on that word alone (one position, unbatched);
     an independent int64 rebuild (no float) of 300 random rows of FT5/FT6/FP5/FP6/OWN1-3; sha256 of every file;
     random sequences of random words (pairs not in OWN: the proj fallback; any fc.0 shift): every pile identical.
  D  controls: another word's stored rows read (Z0, FT, OWN) must break the piles; order-free: the lines met in a
     random order and the parts added in a random order give the same piles.
PREDICTION: everything identical (exact by construction: stored parts + met parts add to the plain piles as whole
numbers); judge5 6,000/6,000 and 6,000/6,000; every control breaks its piles; no fallback on the texts.
    python3 layer0/verify.py parts=A,B,J,C,D
"""
import sys, os, json, time, hashlib, resource, warnings, numpy as np
warnings.filterwarnings("ignore")
HERE = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, HERE); os.chdir(HERE)
PARTS = dict(a.split("=") for a in sys.argv[1:]).get("parts", "A,B,J,C,D").split(",")
import base2_layer0 as L
E, J = L.E, L.J; W, ES = J.W, J.ES
t0 = time.time()
def gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
def say(*a): print(*a, f"   ({time.time() - t0:.0f}s, {gb():.2f} GB)", flush=True)
REFS = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in REFS]
WIN = [w["ids"] for w in J.WINDOWS]
WM = W["mproj.0"][0]

def plain_layer0(ids):                                # base2.py's layer 0, its own functions, every piece kept
    ids = np.asarray(ids); n = len(ids); R = L.WTE[ids] + L.WPE[:n]
    x16, ex, LS = J.gauge_log(R); qpile = J.product(x16, W["qkv.0"][0]); QKV = J.gauged_bank(R, "qkv.0")
    o = J.attention(QKV); ppile = J.product(J.to16(o, -ES)[0], L.WP); att = J.plain_bank(o, "proj.0"); R1 = R + att
    fpile = J.product(J.gauge_log(R1)[0], L.WF); H = J.gauged_bank(R1, "fc.0"); G = J.edge(H)
    mpile = J.product(J.to16(G, -ES)[0], WM); R2 = R1 + J.plain_bank(G, "mproj.0")
    return dict(qkv_pile=qpile, QKV=QKV, o=o, proj_pile=ppile, att=att, R1=R1, fc_pile=fpile, H=H, G=G, mproj_pile=mpile, R2=R2)
def plain_rest(R):                                     # base2.py's layers 1-11 and the top at every position
    S = []
    for l in range(1, 12):
        R = R + J.plain_bank(J.attention(J.gauged_bank(R, f"qkv.{l}")), f"proj.{l}"); S.append(R)
        R = R + J.plain_bank(J.edge(J.gauged_bank(R, f"fc.{l}")), f"mproj.{l}"); S.append(R)
    w, we, b = W["head"]; x16, ex, LS = J.gauge_log(R); pile = J.product(x16, w)
    return S, pile, J.power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b
KEYS = ["o", "proj_pile", "att", "R1", "fc_pile", "H", "G", "mproj_pile"]
def check_text(ids):
    """every piece identical? returns (layer-0 pieces ok, streams ok, top ok, number of piles compared)"""
    P = plain_layer0(ids); got = []
    E.ON_PILE = got.append
    try:
        rec = {"streams": []}; R = L.stream(ids, rec=rec)
    finally: E.ON_PILE = None
    r = rec["rows"] if "rows" in rec else np.array([], np.int64)
    ok = np.array_equal(got[0], P["qkv_pile"]) and np.array_equal(rec["QKV"], P["QKV"])
    mp = J.product(J.to16(rec["G"], -ES)[0], WM) if len(r) else None
    for k in KEYS:
        mine = mp if k == "mproj_pile" else rec.get(k)
        if len(r): ok &= np.array_equal(mine, P[k][r])
    ok &= np.array_equal(rec["R2"], P["R2"])
    S, tp, ts = plain_rest(P["R2"])
    sok = all(np.array_equal(a, b) for a, b in zip(rec["streams"][1:], S)) and len(S) == 22
    w, we, b = W["head"]; x16, ex, LS = J.gauge_log(R); mtp = J.product(x16, w)
    mts = J.power(mtp, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b
    tok = np.array_equal(mtp, tp) and np.array_equal(mts, ts) and all(J.race(mts[i]) == J.race(ts[i]) for i in range(len(ids)))
    npiles = len(ids) * (2304 + 768) + len(r) * (768 + 3072 + 768)    # piles: qkv + stream at every position; proj, fc, mproj past position 0
    return ok, sok, tok, npiles

for part in PARTS:
    if part in ("A", "B"):
        texts = TEXTS if part == "A" else WIN; name = "30 refs texts" if part == "A" else "300 judge windows"
        before = dict(L.FALLBACK); n = [0, 0, 0, 0]
        for ids in texts:
            a, s, t, npl = check_text(ids); n[0] += a; n[1] += s; n[2] += t; n[3] += npl
        fb = {k: L.FALLBACK[k] - before[k] for k in before}
        say(f"{part}. {name} ({sum(len(t) for t in texts):,} positions): layer 0 every piece identical {n[0]}/{len(texts)} "
            f"({n[3]:,} piles and stream values), the stream after every half layer of layers 1-11 {n[1]}/{len(texts)}, "
            f"top piles, scores and words at every position {n[2]}/{len(texts)}; fallbacks {fb}")
    elif part == "J":
        inn, same, tot = J.judge(forward=L.scores_all, n=300)
        say(f"J. judge5.judge(forward=base2_layer0.scores_all, n=300): in top 5 {inn}/{tot}, same top word {same}/{tot}")
    elif part == "C":
        bad = 0; CH = 2048
        for i in range(0, 50257, CH):
            for w in range(i, min(i + CH, 50257)):
                R = L.WTE[w:w + 1] + L.WPE[:1]
                R1 = R + J.plain_bank(J.attention(J.gauged_bank(R, "qkv.0")), "proj.0")
                R2 = R1 + J.plain_bank(J.edge(J.gauged_bank(R1, "fc.0")), "mproj.0")
                bad += not np.array_equal(R2[0], L.Z0.read([w])[0])
        say(f"C. every row of Z0 against base2.py's layer 0 on that word alone (unbatched): identical {50257 - bad:,}/50,257")
        rng = np.random.default_rng(21); I64 = lambda a: np.asarray(a).astype(np.int64)
        PF, PP = I64(J.M["fc.0.w"]), I64(J.M["proj.0.w"])
        for s1 in (5, 6):
            ws = rng.choice(50257, 300, replace=False); ps = rng.choice(1024, 300, replace=False)
            okw = np.array_equal(J.rshift(L.WTE[ws], s1) @ PF, L.FT[s1].read(ws))           # int64 products, no float
            okp = np.array_equal(J.rshift(L.WPE[ps] + L.BP[None, :], s1) @ PF, L.FP[s1].read(ps))
            say(f"   independent int64 rebuild (no float): 300 rows of FT{s1} identical {okw}; 300 rows of FP{s1} identical {okp}")
        keys = np.load(f"{L.L0}/OWN1_ids.npy").astype(np.int64); ks = rng.choice(keys, 300, replace=False)
        w_, p_ = ks // 1024, ks % 1024
        v = J.gauged_bank(L.WTE[w_] + L.WPE[p_], "qkv.0")[:, 1536:]
        for so in (1, 2, 3):
            say(f"   independent int64 rebuild (no float): 300 rows of OWN{so} identical {np.array_equal(J.rshift(v, so) @ PP, L.OWN[so].read(ks))}")
        allok = True
        for f, meta in L.MAN0["files"].items():
            h = hashlib.sha256()
            with open(f"{L.L0}/{f}", "rb") as fh:
                for blk in iter(lambda: fh.read(1 << 24), b""): h.update(blk)
            allok &= h.hexdigest() == meta["sha256"] and os.path.getsize(f"{L.L0}/{f}") == meta["bytes"]
        say(f"   sha256 and size of all {len(L.MAN0['files'])} files against the manifest: {'all match' if allok else 'MISMATCH'}")
        for n_tok, count in ((2, 500), (12, 100)):
            before = dict(L.FALLBACK); n = [0, 0, 0]
            for _ in range(count):
                ids = list(rng.integers(0, 50257, n_tok)); a, s, t, _ = check_text(ids); n[0] += a; n[1] += s; n[2] += t
            fb = {k: L.FALLBACK[k] - before[k] for k in before}
            say(f"   random sequences of {n_tok} random words ({count}): layer 0 every piece {n[0]}/{count}, streams {n[1]}/{count}, "
                f"top {n[2]}/{count}; fallbacks taken {fb}")
    elif part == "D":
        rng = np.random.default_rng(22); n_z = n_f = n_o = 0; n_t = 0; ordok = True; ntr = 0
        for ids in TEXTS:
            ids = np.asarray(ids); P = plain_layer0(ids); rec = {}; L.layer0(ids, rec=rec); r = rec["rows"]; n_t += 1
            other = rng.integers(0, 50257, len(ids))
            n_z += np.array_equal(L.Z0.read(other[:1])[0], P["R2"][0])                     # another word's Z0 row
            s1 = rec["s1"]; fp = np.empty_like(rec["fc_pile"])                               # another word's FT row
            for sv in np.unique(s1):
                m = np.flatnonzero(s1 == sv)
                fp[m] = L.FT[sv].read(other[r[m]]) + L.FP[sv].read(r[m]) + J.product(rec["rest"][m], L.WF)
            n_f += np.array_equal(fp, P["fc_pile"][r])
            so = rec["so"]; keys = ids[r] * 1024 + r; okeys = np.roll(keys, 1); pp = np.empty_like(rec["proj_pile"])   # another pair's OWN row
            for sv in np.unique(so):
                m = np.flatnonzero(so == sv); hk = L.OWN[sv].has(okeys[m])
                pp[m] = np.where(hk[:, None], L.OWN[sv].read(np.where(hk, okeys[m], keys[m])), 0) + J.product(rec["o16"][m] - J.rshift(rec["QKV"][r[m], 1536:], sv), L.WP)
            n_o += np.array_equal(pp, P["proj_pile"][r])
            # order-free: fc.0 and proj.0 piles with the lines in a random order and the parts added in a random order
            perm = rng.permutation(768); WFi = J.M["fc.0.w"].astype(np.int64)[perm]; WPi = J.M["proj.0.w"].astype(np.int64)[perm]
            for sv in np.unique(s1):
                m = np.flatnonzero(s1 == sv)
                parts = [L.FT[sv].read(ids[r[m]]), L.FP[sv].read(r[m]), rec["rest"][m][:, perm] @ WFi]
                tot = np.zeros_like(parts[0])
                for i in rng.permutation(3): tot = tot + parts[i]
                ordok &= np.array_equal(tot, P["fc_pile"][r[m]]); ntr += 1
            for sv in np.unique(so):
                m = np.flatnonzero(so == sv)
                parts = [L.OWN[sv].read(keys[m]), (rec["o16"][m] - J.rshift(rec["QKV"][r[m], 1536:], sv))[:, perm] @ WPi]
                tot = np.zeros_like(parts[0])
                for i in rng.permutation(2): tot = tot + parts[i]
                ordok &= np.array_equal(tot, P["proj_pile"][r[m]]); ntr += 1
        say(f"D. CONTROLS on the 30 refs texts: another word's Z0 row gives the stream at position 0 {n_z}/{n_t}; another word's "
            f"FT rows give the fc.0 piles {n_f}/{n_t}; the previous position's OWN rows give the proj.0 piles {n_o}/{n_t}")
        say(f"   order-free: lines met in a random order, parts added in a random order: identical in {ntr} of {ntr} trials: {ordok}")
say(f"fallbacks over the whole run: {L.FALLBACK}")
