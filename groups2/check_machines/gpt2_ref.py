"""CHECKER (independent of base2_groups.py): base2.py's OWN steps, run fresh, on the checker's random draw.
Executes base2.py's source (up to the tokenizer) unchanged and records, text by text, every pile, the fc outputs and
shifts, every stream, the light at the top and the top scores' top 5 (judge5's rule) and race winner (base2.race).
Also the 50 drawn judge5 windows: the top 5 and race winner at positions 3..22, and the light at the top there.
    python3 gpt2_ref.py   -> gpt2_ref.npz
Draw: numpy default_rng(20261007): texts choice(30, 5) = [5, 12, 19, 20, 28]; windows choice(300, 50) next.
"""
import os, sys, json, time, resource, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/groups2/check_machines"
src = open(f"{HERE}/base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]
src = src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np").replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
B = {}; cwd = os.getcwd(); os.chdir(HERE); argv = sys.argv; sys.argv = ["x"]; exec(src, B); sys.argv = argv; os.chdir(cwd)
M, W, ES = B["M"], B["W"], B["ES"]
gauge_log, product, power, rshift, to16, edge, attention, race = (B[k] for k in ("gauge_log", "product", "power", "rshift", "to16", "edge", "attention", "race"))
def top5(S):                                                  # judge5.top5, its two lines
    idx = np.argpartition(-S, 5, axis=1)[:, :5]; order = np.argsort(-np.take_along_axis(S, idx, 1), axis=1)
    return np.take_along_axis(idx, order, 1)
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6

def run(ids, rec, pre, keep_qkv=False):
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        w, we, b = W[f"qkv.{l}"]; x16, ex, LS = gauge_log(R); pile = product(x16, w)
        rec[f"{pre}qkv_light_{l}"] = x16; rec[f"{pre}qkv_pile_{l}"] = pile
        QKV = power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b
        if keep_qkv: rec[f"{pre}QKV_{l}"] = QKV
        o = attention(QKV); rec[f"{pre}arrival_{l}"] = o
        w, we, b = W[f"proj.{l}"]; o16, eo = to16(o, -ES); pile = product(o16, w)
        rec[f"{pre}proj_light_{l}"] = o16; rec[f"{pre}proj_pile_{l}"] = pile
        R = R + rshift(pile, -(we[None, :] + eo[:, None] + ES)) + b; rec[f"{pre}stream_a_{l}"] = R
        w, we, b = W[f"fc.{l}"]; x16, ex, LS = gauge_log(R); pile = product(x16, w)
        rec[f"{pre}fc_light_{l}"] = x16; rec[f"{pre}fc_pile_{l}"] = pile
        H = power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b
        out16, eg = to16(edge(H), -ES); rec[f"{pre}fc_out16_{l}"] = out16; rec[f"{pre}fc_sh_{l}"] = eg + ES
        w, we, b = W[f"mproj.{l}"]; pile = product(out16, w); rec[f"{pre}mproj_pile_{l}"] = pile
        R = R + rshift(pile, -(we[None, :] + eg[:, None] + ES)) + b; rec[f"{pre}stream_{l}"] = R
    return R

def top(R):
    w, we, b = W["head"]; x16, ex, LS = gauge_log(R)
    S = power(product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b
    return x16, ex, LS, top5(S), np.array([race(s) for s in S]), S

if __name__ == "__main__":
    t0 = time.time(); rng = np.random.default_rng(20261007)
    TS = np.sort(rng.choice(30, 5, replace=False)); WS = np.sort(rng.choice(300, 50, replace=False))
    REFS = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in REFS]
    WIN = json.load(open(f"{HERE}/judge5_texts.json"))
    rec = dict(texts=TS, windows=WS)
    for t in TS:
        R = run(TEXTS[t], rec, f"t{t}_", keep_qkv=True)
        x16, ex, LS, t5, win, S = top(R)
        rec.update({f"t{t}_top_x16": x16, f"t{t}_top_ex": ex, f"t{t}_top_LS": LS, f"t{t}_top5": t5, f"t{t}_win": win,
                    f"t{t}_s5": np.sort(S, 1)[:, ::-1][:, :6]})
        print(f"text {t}: {len(TEXTS[t])} positions ({time.time() - t0:.0f}s, {rss()} MB)", flush=True)
    WX, WEX, WLS, W5, WW = [], [], [], [], []
    for i in WS:
        ids = WIN[i]["ids"]; P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
        for l in range(12):
            R = R + B["plain_bank"](attention(B["gauged_bank"](R, f"qkv.{l}")), f"proj.{l}")
            R = R + B["plain_bank"](edge(B["gauged_bank"](R, f"fc.{l}")), f"mproj.{l}")
        x16, ex, LS, t5, win, S = top(R[3:23])
        WX.append(x16); WEX.append(ex); WLS.append(LS); W5.append(t5); WW.append(win)
    rec.update(w_x16=np.concatenate(WX), w_ex=np.concatenate(WEX), w_LS=np.concatenate(WLS), w_top5=np.concatenate(W5), w_win=np.concatenate(WW))
    J5 = np.load(f"{HERE}/judge5_ref.npz")["top5"][WS].reshape(-1, 5)
    print(f"50 windows: base2's fresh top 5 = judge5_ref.npz {int((rec['w_top5'] == J5).all(1).sum())}/{len(J5)} ({time.time() - t0:.0f}s, {rss()} MB)")
    np.savez(f"{OUT}/gpt2_ref.npz", **rec)
    print(f"saved ({time.time() - t0:.0f}s, max memory {rss()} MB)")
