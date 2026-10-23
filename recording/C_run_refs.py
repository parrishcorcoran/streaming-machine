"""PHASE C, STEPS 2 AND 3(b, c): THE RECORDING MACHINE ON THE 30 REFS TEXTS (ids + ref, every position, 648), FROM
THE FILES IN recording/C/ ALONE (predictions: C_machine.py's header).
  - every pile of the 48 banks and the stream after every half layer, against base2.py's own run kept by C_plain.py
    (recording/C_ref/plain_refs.npz, made in a separate process);
  - the light at the top, and the top read at every position: each position's scores (sha256 of the whole row),
    top 5 (set and order) and race winner, against the same file;
  - the work per token: lit-pixel meetings in the banks against plain, the whole machine against 3,666,596,259,
    the extra adds apart, the light-shuffled control; lit places a weight and a light value.
    python3 recording/C_run_refs.py           -> recording/C_run_refs.log, recording/C_logs/refs_counts.json
    python3 recording/C_run_refs.py order=1   -> ORDER-FREE check: at every bank the lines in a random order (the
                                                 light's lines with them) and the 289 place pairs added in a random
                                                 order; every pile against the same reference (no counts)
"""
import sys, os, json, time, hashlib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ARGS = dict(a.split("=") for a in sys.argv[1:]); ORDER = ARGS.get("order") == "1"; NT = int(ARGS.get("n", 30))
HERE = "/Users/abundancemachine/streaming-machine"; os.makedirs(f"{HERE}/recording/C_logs", exist_ok=True)
LOG = open(f"{HERE}/recording/C_run_refs{'_order' if ORDER else ''}{'' if NT == 30 else '_try'}.log", "w"); t0 = time.time()
def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
import C_machine as K
REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")       # base2.py's own run, made by C_plain.py
TEXTS = K.TEXTS[:NT]; offs = np.concatenate([[0], np.cumsum([len(t) for t in TEXTS])]); NPOS = int(offs[-1])
assert int(REF["npos"][:NT].sum()) == NPOS
log(f"recording machine from {K.CDIR} ({len(K.MAN['plates'])} recorded plates); {NT} texts, {NPOS} positions; "
    f"{'ORDER-FREE run: lines and place pairs in a random order at every bank' if ORDER else 'counted run'}")

bad, checked = {}, []
def check(k, v):
    ref = REF[k.replace(" ", "_")][:NPOS]; assert ref.shape == v.shape, (k, ref.shape, v.shape)
    if not np.array_equal(ref, v):
        rows = np.flatnonzero((ref != v).any(1)); tx = sorted(set((np.searchsorted(offs, rows, side="right") - 1).tolist()))
        bad[k] = dict(positions=len(rows), texts=tx)
    checked.append(k)

C = None if ORDER else K.Count()
opts = {}
if ORDER:
    rng = np.random.default_rng(20261005)
    opts = {n: dict(line_perm=rng.permutation(K.MAN["plates"][n]["lines"])) for n in K.BANKS}
x16, ex, LS = K.run(TEXTS, C=C, check=check, plate_opts=opts, pair_seed=(777 if ORDER else None),
                    ctl_seed=(None if ORDER else 4242), log=log)
same_top_light = bool(np.array_equal(x16, REF["x16"][:NPOS]) and np.array_equal(ex, REF["ex"][:NPOS]) and np.array_equal(LS, REF["LS"][:NPOS]))
log(f"\nEVERY PILE AND THE STREAM: {len(checked)} arrays checked (48 banks' piles, 24 streams), each over all {NPOS} positions; "
    f"differing: {len(bad)}{'' if not bad else ' -- ' + json.dumps(bad)[:2000]}")
log(f"  texts with any difference: {sorted(set(t for b in bad.values() for t in b['texts']))}")
log(f"  the light at the top identical: {same_top_light}")

# ---------- the top read, untouched, its table rebuilt from the recording's sheets ----------
dig_same = t5_same = win_same = 0
for c0 in range(0, NPOS, 48):
    sl = slice(c0, min(c0 + 48, NPOS)); S = K.top_scores(x16[sl], ex[sl], LS[sl], C)
    dig = np.array([np.frombuffer(hashlib.sha256(np.ascontiguousarray(s, np.int64).tobytes()).digest(), np.uint8) for s in S])
    dig_same += int((dig == REF["score_sha"][sl]).all(1).sum()); t5_same += int((K.top5(S) == REF["top5"][sl]).all(1).sum())
    win_same += int(sum(K.race(s) == w for s, w in zip(S, REF["win"][sl]))); del S
log(f"TOP READ at {NPOS} positions: every word's score identical (sha256 of the row) {dig_same}/{NPOS}; top 5 (set and order) "
    f"{t5_same}/{NPOS}; race winner {win_same}/{NPOS}  ({time.time() - t0:.0f}s, {K.rss()} MB)")
if ORDER:
    log(f"\nORDER-FREE: every pile and stream identical with the lines and the place pairs in a random order: {not bad}")
    log(f"(max memory {K.rss()} MB, {time.time() - t0:.0f}s)"); sys.exit()

# ---------- the work per token ----------
def tot(who, part=None, name=None, start=None):
    s = 0
    for k, v in C.items():
        f = k.split("|")
        if len(f) == 3 and f[0] in who and (part is None or f[1] == part) and (name is None or name in f[2]) and (start is None or f[2].startswith(start)): s += v
    return s
plain = tot(("both", "plain")); rec = tot(("both", "rec"))
pt = lambda v: K.per_token(v, NPOS)
log(f"\nWORK PER TOKEN (photons; mean over the {NPOS} positions of the {NT} refs texts)")
log(f"  plain machine     {pt(plain):>15,}   (combined/machine.py's count: 3,666,596,259)   same: {pt(plain) == 3666596259}")
log(f"  recording machine {pt(rec):>15,}   = {K.pct2(rec, plain)}% of plain (extra adds included)")
log(f"\n  {'part':70s} {'plain':>15s} {'recording':>15s}   share")
for part in ("banks", "top read", "arrival", "rest", "extra"):
    p, r = tot(("both", "plain"), part), tot(("both", "rec"), part)
    log(f"  {part:70s} {pt(p):>15,} {pt(r):>15,}   {K.pct2(r, p) + '%' if p else '--'}")
    for k in sorted(C, key=lambda k: -C[k]):
        f = k.split("|")
        if len(f) == 3 and f[1] == part and f[0] in ("both", "plain", "rec"):
            log(f"      {f[2]:66s} {pt(C[k]) if f[0] != 'rec' else '':>15} {pt(C[k]) if f[0] != 'plain' else '':>15}")
bank_plain, bank_rec = tot(("plain",), "banks"), tot(("rec",), "banks"); bank_ctl = tot(("ctl",), "banks")
extra = tot(("rec",), "extra")
log(f"\n  BANKS ONLY (48 banks): lit-pixel meetings {pt(bank_rec):,} against plain {pt(bank_plain):,} = {K.pct2(bank_rec, bank_plain)}%")
for kind in ("qkv", "proj", "fc", "mproj"):
    p, r, c = tot(("plain",), "banks", start=kind + " "), tot(("rec",), "banks", start=kind + " "), tot(("ctl",), "banks", start=kind + " ")
    log(f"      {kind:6s} {pt(r):>15,} / {pt(p):>15,} = {K.pct2(r, p)}%     CONTROL light shuffled across its lines: {K.pct2(c, p)}%")
log(f"  CONTROL, every position's light shuffled across its lines (same values): {K.pct2(bank_ctl, bank_plain)}% of plain")
log(f"  extra adds, stated apart: writing the light in signed digits {pt(extra):,} a token "
    f"({K.pct2(extra, bank_plain)}% of the plain bank meetings)")
log(f"  whole machine: plain {pt(plain):,}; recording {pt(rec - extra):,} without the extra adds ({K.pct2(rec - extra, plain)}%), "
    f"{pt(rec):,} with them ({K.pct2(rec, plain)}%)")
lv, l1, ls = tot(("lit",), "light", "values"), tot(("lit",), "light", "plain 1s"), tot(("lit",), "light", "signed lit places")
log(f"\nLIT PLACES A LIGHT VALUE (every bank's input, every position): plain base 2 {l1 / lv:.4f} -> signed digits {ls / lv:.4f} "
    f"({K.pct2(ls, l1)}%)")
for kind in ("qkv", "proj", "fc", "mproj"):
    v, a, b = tot(("lit",), "light", start=kind + " values"), tot(("lit",), "light", start=kind + " plain 1s"), tot(("lit",), "light", start=kind + " signed lit places")
    log(f"      {kind:6s} {a / v:.4f} -> {b / v:.4f} ({K.pct2(b, a)}%)")
T = K.MAN["totals"]
for g in ("48 plates", "top table"):
    t = T[g]; log(f"LIT PLACES A WEIGHT, {g}: plain base 2 {t['lit_plain'] / t['weights']:.4f} -> signed digits "
                 f"{t['lit_signed'] / t['weights']:.4f} ({K.pct2(t['lit_signed'], t['lit_plain'])}%)")
json.dump(dict(npos=NPOS, total=dict(C), bad=bad, top=dict(sha_same=dig_same, top5_same=t5_same, win_same=win_same),
               same_top_light=same_top_light), open(f"{HERE}/recording/C_logs/refs_counts{'' if NT == 30 else '_try'}.json", "w"))
log(f"\nsaved recording/C_logs/refs_counts{'' if NT == 30 else '_try'}.json ({time.time() - t0:.0f}s, max memory {K.rss()} MB)")
