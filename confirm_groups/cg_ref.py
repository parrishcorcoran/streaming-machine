"""confirm_groups/cg_ref.py -- A FRESH RUN OF qwen/qwen_base2.py's OWN STEPS (the reference machine), by the independent
checker, on the 30 refs texts (capture's Qwen words of refs_many.json, 613 positions) and on the 30 greedy texts
(qwen_fringes/words_ref.json's prompt + greedy words, 619 positions, read at 480 places).

HOW: qwen_base2.py's source is executed with four lines changed, none of which changes a number:
  - the loader: M is the npz opened lazily, and W gives each plate as int64 when its layer comes (the file's own
    W dict would hold every plate as int64, 4.8 GB, over the 3 GB rule); the word rows are read in place;
  - the plate product stays qwen_base2's own `x16 @ w` (int64), only recorded (its sha256 per text);
  - the forward is run for every text layer by layer together (every step is row by row and the arrival reads only its
    own text's positions), and the read at the top is qwen_base2's own gauged_slab(R, "head") at every position.
The gauge's math.log2 is left as qwen_base2.py has it (this is the reference).

RECORDED (confirm_groups/out/ref_<set>.json): per text the sha256 of every slab's pile (the int64 product), every
arrival output, the stream after each half layer, the top light (x16, its power, the water level's log); per position
the score row's sha256, the top 5 (by score, ties by word number) and the race winner (qwen_base2.race).

PREDICTION (written before this file ran): every digest equal to capture's record (qwen06b/light/meta.json: 3,360 pile
digests, 613 score rows, top 5 and winners) and to qwen_fringes/ref.json (5,910 digests); on the greedy texts every
digest equal to words_ref.json's and the race winner at each of the 480 read places = the next greedy word.

    python3 confirm_groups/cg_ref.py set=refs|words
"""
import os, sys, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cg_common import HERE, QD, NPZ, CG, member, digest, rss
import hashlib
A = dict(a.split("=") for a in sys.argv[1:]); SET = A.get("set", "refs")
os.makedirs(f"{CG}/out", exist_ok=True)
LOG = open(f"{CG}/out/ref_{SET}.log", "w")
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

class LazyW(dict):
    def __getitem__(self, name):
        if name not in self.keys():
            for k in [k for k in self.keys() if k != "head"]: dict.__delitem__(self, k)      # one slab plate at a time
            dict.__setitem__(self, name, (np.asarray(member(name + ".w")).astype(np.int64), np.asarray(member(name + ".e")).astype(np.int64)))
        return dict.__getitem__(self, name)
REC = dict(rows=None, dig={})
def rec(name, pile):
    for t, (a, b) in enumerate(REC["rows"]): REC["dig"][f"pile|{name}|{t}"] = digest(pile[a:b])
    return pile
src = open(f"{QD}/qwen_base2.py").read(); src = src[:src.index("from transformers import AutoTokenizer")]
for a, b in [('M = np.load("qwen_base2.npz")', 'M = _M'),
             ('W = {k[:-2]: (M[k].astype(np.int64), M[k[:-2] + ".e"]) for k in M.files if k.endswith(".w")}', 'W = _W'),
             ("x16, ex = to16(R); pile = x16 @ w", "x16, ex = to16(R); pile = _rec(name, x16 @ w)"),
             ("x16, ex = to16(X); return rshift(x16 @ w,", "x16, ex = to16(X); return rshift(_rec(name, x16 @ w),")]:
    assert src.count(a) == 1, a; src = src.replace(a, b)
Q = dict(_M=np.load(NPZ), _W=LazyW(), _rec=rec); _argv = sys.argv; sys.argv = ["x"]; exec(src, Q); sys.argv = _argv
assert (Q["GAUGE"], Q["TURN"], Q["BEND"], Q["ARRIVAL"]) == ("int",) * 4

def top5(s): return np.lexsort((np.arange(len(s)), -s))[:5].tolist()

def run(texts):
    t0 = time.time(); lens = [len(t) for t in texts]; off = np.concatenate([[0], np.cumsum(lens)]).astype(int)
    rows = [(int(off[i]), int(off[i + 1])) for i in range(len(texts))]; REC.update(rows=rows, dig={})
    EMB = member("embed"); R = np.asarray(EMB[np.concatenate([np.asarray(t, np.int64) for t in texts])]).astype(np.int64)
    gs, ps, att, bend, to16, gauge, race = (Q[k] for k in ("gauged_slab", "plain_slab", "attention", "bend", "to16", "gauge", "race"))
    D = REC["dig"]
    for l in range(Q["L"]):
        QKV = gs(R, f"qkv.{l}")
        o = np.concatenate([att(QKV[a:b], l) for a, b in rows]); del QKV
        for t, (a, b) in enumerate(rows): D[f"arrival|{l}|{t}"] = digest(o[a:b])
        R = R + ps(o, f"o.{l}"); del o
        for t, (a, b) in enumerate(rows): D[f"stream|{l}.0|{t}"] = digest(R[a:b])
        GU = gs(R, f"gu.{l}"); R = R + ps(bend(GU[:, :3072], GU[:, 3072:]), f"down.{l}"); del GU
        for t, (a, b) in enumerate(rows): D[f"stream|{l}.1|{t}"] = digest(R[a:b])
        if l % 7 == 6: log(f"  layer {l} done ({time.time() - t0:.0f}s, {rss()} MB)")
    x16, ex = to16(R); LS = np.asarray(gauge(x16, ex, Q["D"]), np.int64); ex = np.asarray(ex, np.int64)
    for t, (a, b) in enumerate(rows): D[f"top light|{t}"] = digest(np.concatenate([x16[a:b], ex[a:b, None], LS[a:b, None]], 1))
    T5, WIN, SHA = [], [], []
    for a, b in rows:                                                          # qwen_base2's own read, every position
        S = gs(R[a:b], "head")
        T5 += [top5(s) for s in S]; WIN += [int(race(s)) for s in S]; SHA += [hashlib.sha256(np.ascontiguousarray(s, np.int64).tobytes()).hexdigest() for s in S]
        del S
    log(f"  the read done ({time.time() - t0:.0f}s, {rss()} MB)")
    return dict(rows=rows, dig=dict(D), top5=T5, win=WIN, sha=SHA, x16=x16, ex=ex, LS=LS)

if SET == "refs":
    M_ = json.load(open(f"{HERE}/qwen06b/light/meta.json")); texts = M_["texts"]
    log(f"REFERENCE RUN of qwen_base2.py's own steps on the 30 refs texts ({sum(map(len, texts))} positions)")
    r = run(texts)
    RF = json.load(open(f"{HERE}/qwen_fringes/ref.json"))
    pd = sum(r["dig"][f"pile|{k}|{t}"] != v for k, vs in M_["dig"].items() for t, v in enumerate(vs))
    log(f"  piles vs capture's record: {pd} of {sum(len(v) for v in M_['dig'].values())} differ")
    log(f"  score rows vs capture's: {sum(a != b for a, b in zip(r['sha'], M_['sha']))} of {len(M_['sha'])} differ; top 5 "
        f"{sum(a != b for a, b in zip(r['top5'], M_['top5']))} differ; race winner {sum(a != b for a, b in zip(r['win'], M_['win']))} differ")
    kinds = {}
    for k, v in RF["dig"].items():
        kk = k.split("|")[0]; kinds.setdefault(kk, [0, 0]); kinds[kk][1] += 1; kinds[kk][0] += r["dig"].get(k) != v
    log("  vs qwen_fringes/ref.json: " + ", ".join(f"{k} {a} of {b} differ" for k, (a, b) in kinds.items()))
    log(f"  read vs ref.json: score rows {sum(a != b for a, b in zip(r['sha'], RF['read']['sha']))} differ, top 5 "
        f"{sum(a != b for a, b in zip(r['top5'], RF['read']['top5']))} differ, winners {sum(a != b for a, b in zip(r['win'], RF['read']['win']))} differ")
    cap_top = np.load(f"{HERE}/qwen06b/light/top_light.npy").astype(np.int64)
    log(f"  top light vs capture's top_light.npy: {(cap_top != r['x16']).any(1).sum()} rows differ; ex "
        f"{(np.load(f'{HERE}/qwen06b/light/top_ex.npy') != r['ex']).sum()}; LS {(np.load(f'{HERE}/qwen06b/light/top_LS.npy') != r['LS']).sum()}")
else:
    W = json.load(open(f"{HERE}/qwen_fringes/words_ref.json")); texts = W["texts"]
    log(f"REFERENCE RUN of qwen_base2.py's own steps on the 30 greedy texts ({sum(map(len, texts))} positions, {len(W['read_rows'])} read places)")
    r = run(texts)
    kinds = {}
    for k, v in W["dig"].items():
        kk = k.split("|")[0]; kinds.setdefault(kk, [0, 0]); kinds[kk][1] += 1; kinds[kk][0] += r["dig"].get(k) != v
    log("  vs words_ref.json: " + ", ".join(f"{k} {a} of {b} differ" for k, (a, b) in kinds.items()))
    rr = W["read_rows"]
    log(f"  the race winner at the read places = the next greedy word: {sum(r['win'][q] == n for q, n in zip(rr, W['next']))} of {len(rr)}")
    log(f"  vs words_ref's read: score rows {sum(r['sha'][q] != s for q, s in zip(rr, W['sha']))} differ, top 5 "
        f"{sum(r['top5'][q] != t for q, t in zip(rr, W['top5']))} differ")
    RQ = json.load(open(f"{QD}/refs_qwen.json"))
    gen = [[r["win"][q] for q in rr[16 * i:16 * i + 16]] for i in range(30)]
    log(f"  prompts = refs_qwen.json's: {sum(t[:len(q['ids'])] == q['ids'] for t, q in zip(texts, RQ))} of 30; the 16 words generated "
        f"= words_ref's words {sum(g == w for g, w in zip(gen, W['words']))} of 30; = fp32's own words {sum(g == q['ref'] for g, q in zip(gen, RQ))} of 30")
json.dump({k: r[k] for k in ("rows", "dig", "top5", "win", "sha")}, open(f"{CG}/out/ref_{SET}.json", "w"))
np.savez(f"{CG}/out/ref_{SET}_top_light.npz", x16=r["x16"], ex=r["ex"], LS=r["LS"])
log(f"done, max memory {rss()} MB")
