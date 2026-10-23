"""THE TOP-5 JUDGE. 300 windows of real text (wikitext-2 test, 24 tokens each, judge5_texts.json), every position
teacher-forced: at positions 3..22 the machine's next word is checked against the full-resolution base-2 machine's
TOP 5 there (6,000 judged positions). A changed machine passes a position when its word is one of the reference's top 5.
    from judge5 import judge, scores_all
    judge(entry=None, top=None, plates=None, forward=None, n=300) -> (in_top5, top1_same, judged)
  forward(ids) -> (positions x 50257) whole-number scores, if you change the machine's arithmetic itself.
  The reference top-5 is built once into judge5_ref.npz (python3 judge5.py).
"""
import sys, os, json, time, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"
_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
_cwd = os.getcwd(); os.chdir(HERE); _argv = sys.argv; sys.argv = ["x"]; exec(_src); sys.argv = _argv; os.chdir(_cwd)
WINDOWS = json.load(open(f"{HERE}/judge5_texts.json"))
FIRST, LAST = 3, 23                                     # judged positions: next word after positions 3..22
def scores_all(ids, wte=None):
    """the base-2 machine on every position at once; whole-number scores at the top for every judged position"""
    wte = M["wte"] if wte is None else wte
    P = len(ids); R = wte[ids] + M["wpe"][:P]
    for l in range(12):
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        R = R + plain_bank(edge(gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
    w, we, b = W["head"]; Rj = R[FIRST:LAST]; x16, ex, LS = gauge_log(Rj)
    return power(product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b
def top5(S):
    idx = np.argpartition(-S, 5, axis=1)[:, :5]; order = np.argsort(-np.take_along_axis(S, idx, 1), axis=1)
    return np.take_along_axis(idx, order, 1)
_REF = None
def _ref():
    global _REF
    if _REF is None: _REF = np.load(f"{HERE}/judge5_ref.npz")["top5"]
    return _REF
def judge(entry=None, top=None, plates=None, forward=None, n=300):
    ref = _ref(); saved = {}; inn = same = tot = 0
    try:
        if top is not None: saved["head"] = W["head"]; W["head"] = (np.asarray(top).astype(np.int64).T.astype(np.float64), W["head"][1], W["head"][2])
        for k, v in (plates or {}).items(): saved[k] = W[k]; W[k] = (np.asarray(v).astype(np.float64), W[k][1], W[k][2])
        for i, win in enumerate(WINDOWS[:n]):
            S = forward(win["ids"]) if forward is not None else scores_all(win["ids"], entry)
            pick = S.argmax(1) if S.ndim == 2 else S
            inn += int((ref[i] == pick[:, None]).any(1).sum()); same += int((ref[i][:, 0] == pick).sum()); tot += len(pick)
        return inn, same, tot
    finally:
        for k, v in saved.items(): W[k] = v
if __name__ == "__main__":
    t0 = time.time(); T5 = []
    for i, win in enumerate(WINDOWS):
        T5.append(top5(scores_all(win["ids"])))
        if i % 50 == 49: print(f"  {i + 1} windows ({time.time() - t0:.0f}s)", flush=True)
    np.savez(f"{HERE}/judge5_ref.npz", top5=np.array(T5))
    inn, same, tot = judge()
    print(f"reference built: {tot} judged positions; the unchanged machine: in top 5 {inn}/{tot}, same top word {same}/{tot}  ({time.time() - t0:.0f}s)")
