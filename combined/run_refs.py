"""RUN 1: THE 30 REFS TEXTS, EVERY POSITION (648), COUNTED AND CHECKED (prediction: combined/machine.py's header).
For each text:
  - the plain machine runs with base2.py's own steps, keeping every pile;
  - the combined machine runs (entry V6 + the line-up on every bank + the collapsing top read);
  - every pile and value of all 12 layers, the stream and the light at the top are compared exactly.
Then:
  - the collapsing top read at all 648 positions: top 5 (set and order) and the race winner against the plain read;
  - the light at the top against collapse_top/light_top.npz (saved from the plain machine);
  - the work per token by part, plain against combined, in the unit of all earlier results.
The plain count must come back as 3,666,596,259 a token (extra/extra_layer0.py's count of the same 648 positions).
    python3 combined/run_refs.py        -> combined/refs_counts.json (the counts, per text and in total)
"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
t0 = time.time()
def log(s): print(s, flush=True)
import machine as K
ARGS = dict(a.split("=") for a in sys.argv[1:]); NT = int(ARGS.get("n", 30))   # n=1: a first try on one text
K.ALT_COUNT = ARGS.get("alt") == "1"                        # alt=1: also count the running sums met as they are
TEXTS = K.TEXTS[:NT]
K.build_lineups()
log(f"loaded; line-ups stored for the 48 banks ({time.time() - t0:.0f}s, {K.rss()} MB)")

C = K.Count(); per_text = []; X, EX, LS, PT5, PW = [], [], [], [], []; bad_texts = {}
for t, ids in enumerate(TEXTS):
    ref = K.plain_run(ids)
    Ct = K.Count(); rec = {}
    K.forward(ids, Ct, rec)
    bad = K.same_as_plain(rec, ref)
    if bad: bad_texts[t] = bad
    for k, v in Ct.items(): C.add(k, v)
    per_text.append(dict(Ct))
    x16, ex, ls = rec["top light"]; X.append(x16); EX.append(ex); LS.append(ls)
    S = ref["top scores"]; PT5.append(K.top5(S)); PW.append(np.array([K.race(s) for s in S]))
    del ref, rec, S
    log(f"  text {t + 1:2d}/{NT}: {len(ids)} positions, every pile and value identical: {not bad}"
        f"{'' if not bad else ' -- differs at ' + ', '.join(bad[:6])} ({time.time() - t0:.0f}s, {K.rss()} MB)")
X, EX, LS, PT5, PW = np.concatenate(X), np.concatenate(EX), np.concatenate(LS), np.concatenate(PT5), np.concatenate(PW)
NPOS = len(X)
log(f"\nevery pile and value of all 12 layers, the stream and the light at the top identical to the plain machine: "
    f"{NT - len(bad_texts)}/{NT} texts ({NPOS} positions)")

L = np.load(f"{K.HERE}/collapse_top/light_top.npz"); g = np.flatnonzero(L["src"] == 0); same_saved = 0; off = 0
for t, ids in enumerate(TEXTS):
    r = K.REFS[t]; rows = off + np.arange(len(r["ids"]) - 1, len(r["ids"]) + 15); gt = g[L["win"][g] == t]
    same_saved += int(np.array_equal(X[rows], L["x16"][gt]) and np.array_equal(EX[rows], L["ex"][gt]) and np.array_equal(LS[rows], L["LS"][gt]))
    off += len(ids)
log(f"the light at the top = the plain machine's saved light (collapse_top/light_top.npz): {same_saved}/{NT} texts")

T = K.top_read(X, EX, LS, C, log=log)
log(f"\nTOP READ at all {NPOS} positions: bound broken {T['bad']['bound']} times; words read to the end != plain read "
    f"{T['bad']['final_vs_plain']}; stages' pairs != plain meetings at {T['bad']['pairs_sum']} positions")
log(f"  top 5 (set and order) = the plain machine's: {int((T['top5'] == PT5).all(1).sum())}/{NPOS}; "
    f"race winner the same: {int((T['win'] == PW).sum())}/{NPOS}")
for s, k in enumerate((4, 8, 12)):
    v = np.sort(T["surv"][:, s]); log(f"  words still in after the light's top {k:2d} places: median {v[len(v) // 2]}, "
                                      f"mean {K.per_token(int(v.sum()), len(v))}, least {v[0]}, most {v[-1]}")

# ---------- the work per token by part ----------
PARTS = ["entry", "banks", "arrival", "top read", "extra", "rest"]
NAMES = {"entry": "entry (layer 0's qkv.0, proj.0, fc.0)", "banks": "banks (the other 45)", "arrival": "arrival",
         "top read": "top read (word stamps' meetings)", "extra": "extra adds and bounds",
         "rest": "rest (water levels, bank divides, biases, bends, stream adds, top's divide and standing amounts)"}
def tot(who, part=None):
    return sum(v for k, v in C.items() if len(k.split("|")) == 3 and k.split("|")[0] in who and (part is None or k.split("|")[1] == part))
plain, comb = tot(("both", "plain")), tot(("both", "comb"))
log(f"\nWORK PER TOKEN (photons; mean over the {NPOS} positions of the 30 refs texts)")
log(f"  plain machine {K.per_token(plain, NPOS):,}  (extra/extra_layer0.py: 3,666,596,259)")
log(f"  combined      {K.per_token(comb, NPOS):,}  = {K.pct(comb, plain)}% of plain")
log(f"\n  {'part':100s} {'plain':>15s} {'combined':>15s}  share")
for part in PARTS:
    p, c = tot(("both", "plain"), part), tot(("both", "comb"), part)
    log(f"  {NAMES[part]:100s} {K.per_token(p, NPOS):>15,} {K.per_token(c, NPOS):>15,}  {K.pct(c, p) if p else '--'}{'%' if p else ''}")
    for k in sorted(C, key=lambda k: -C[k]):
        f = k.split("|")
        if len(f) < 3: continue
        who, pa, name = f
        if pa == part and who in ("both", "plain", "comb"):
            col = f"{K.per_token(C[k], NPOS):>15,}"
            log(f"      {('[plain] ' if who == 'plain' else '[combined] ' if who == 'comb' else '') + name:94s} "
                f"{col if who != 'comb' else '':>15s} {col if who != 'plain' else '':>15s}")
log("\n  stated apart (not in the totals):")
for k in sorted(C):
    f = k.split("|")
    if f[0] == "side": log(f"      {'[not used] the qkv.0 carry through the line-up: ' + f[2]:94s} {K.per_token(C[k], NPOS):>15,}")
    if f[0] == "comp": log(f"      {f[1]:94s} {K.per_token(C[k], NPOS):>15,}")
mt = lambda who: tot(who, "entry") + tot(who, "banks") + tot(who, "top read")
lu_meet = sum(v for k, v in C.items() if k.startswith("comb|banks|"))
lu_plain = sum(v for k, v in C.items() if k.startswith("plain|banks|"))
pour = sum(v for k, v in C.items() if k.startswith("comb|extra|line-up: light poured"))
wr = sum(v for k, v in C.items() if k.startswith("comb|extra|line-up: running sums written"))
log(f"\n  the 45 banks: line-up meetings {K.pct(lu_meet, lu_plain)}% of their plain meetings; with the pours "
    f"{K.pct(lu_meet + pour, lu_plain)}%; with the running sums written in single waves {K.pct(lu_meet + pour + wr, lu_plain)}% "
    f"(pours and writing here include proj.0's and fc.0's line-ups)")
log(f"  meetings only (entry + banks + top read): plain {K.per_token(mt(('plain',)), NPOS):,}, combined "
    f"{K.per_token(mt(('comb',)), NPOS):,} = {K.pct(mt(('comb',)), mt(('plain',)))}%")
if K.ALT_COUNT:
    al = sum(v for k, v in C.items() if k.startswith("alt|"))
    lu_all = sum(v for k, v in C.items() if k.startswith("comb|") and "meetings, line-up" in k)
    alt_total = comb - lu_all - wr + al
    log(f"\n  RUNNING SUMS MET AS THEY ARE (gaps in single waves; the running sums not rewritten): line-up meetings "
        f"{K.per_token(al, NPOS):,} instead of {K.per_token(lu_all, NPOS):,} + writing {K.per_token(wr, NPOS):,}")
    log(f"  the 45 banks: {K.pct(sum(v for k, v in C.items() if k.startswith('alt|banks|')), lu_plain)}% of their plain meetings; "
        f"with the pours {K.pct(sum(v for k, v in C.items() if k.startswith('alt|banks|')) + pour, lu_plain)}%")
    log(f"  whole machine: {K.per_token(alt_total, NPOS):,} = {K.pct(alt_total, plain)}% of plain")
json.dump(dict(npos=NPOS, total=dict(C), per_text=per_text, bad_texts={str(k): v for k, v in bad_texts.items()},
               top=dict(bad=T["bad"], same_top5=int((T["top5"] == PT5).all(1).sum()), same_win=int((T["win"] == PW).sum()),
                        surv=T["surv"].tolist(), enter=T["enter"].tolist(), work=T["work"].tolist())),
          open(f"{K.HERE}/combined/refs_counts{'' if NT == 30 else '_try'}{'_alt' if K.ALT_COUNT else ''}.json", "w"))
log(f"\nsaved combined/refs_counts{'_alt' if K.ALT_COUNT else ''}.json ({time.time() - t0:.0f}s, max memory {K.rss()} MB)")
