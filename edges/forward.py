"""WHAT MOVES THE EDGES, step 0: the machine's own column on the 30 refs texts, every write kept.

base2.py executed from its own source, never changed. For each text and each depth l the two writes the stream gets:
    A_l = the arrival write (proj.l's output, its bias included), M_l = the middle write (mproj.l's output, bias included)
and the value each knife edge sees at fc.l, H_l = the machine's own gauged_bank(Rmid_l, "fc.l") (whole numbers, 2^-16).
Rmid_l = word row + position row + sum_{m<l} (A_m + M_m) + A_l, exactly (whole-number adds).
Checks: the last word of every text equals base2.py's forward(); the words equal the refs words where the refs say.
    python3 edges/forward.py        -> edges/forward.npz
"""
import sys, os, json, time, numpy as np
np.seterr(all="ignore")
HERE = "/Users/abundancemachine/streaming-machine"; os.chdir(HERE)
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]
src = src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
src = src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
exec(src)                                                            # the machine, untouched
refs = json.load(open("refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in refs]

def column(ids):
    """forward() of base2.py line for line, keeping both writes of every depth and the knife edges' values"""
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]; A, MW, H = [], [], []
    for l in range(12):
        a = plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        R = R + a
        h = gauged_bank(R, f"fc.{l}")
        m = plain_bank(edge(h), f"mproj.{l}")
        R = R + m
        A.append(a); MW.append(m); H.append(h)
    w, we, b = W["head"]; x16, ex, LS = gauge_log(R)
    top = power(product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b
    return np.array(A), np.array(MW), np.array(H), [race(top[p]) for p in range(P)]

t0 = time.time(); out = {}; same_last = 0; wsame = wn = 0
for t, seq in enumerate(TEXTS):
    A, MW, H, words = column(seq)
    for X in (A, MW, H): assert np.abs(X).max() < 2 ** 31
    same_last += words[-1] == forward(seq)                           # base2.py's own forward
    n0 = len(refs[t]["ids"])
    for p in range(n0 - 1, len(seq) - 1): wsame += words[p] == seq[p + 1]; wn += 1
    out[f"A{t}"], out[f"M{t}"], out[f"H{t}"] = A.astype(np.int32), MW.astype(np.int32), H.astype(np.int32)
    out[f"ids{t}"] = np.array(seq); out[f"words{t}"] = np.array(words)
    print(f"  text {t:2d}: {len(seq)} positions ({time.time() - t0:.0f}s)", flush=True)
np.savez("edges/forward.npz", **out)
print(f"last word == base2.forward: {same_last}/30;  words == refs words: {wsame}/{wn}")
import resource; print(f"largest memory held: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB")
