"""Which places of the light and the plate are copies of the top place (the sign)? In the offset writing (offset 2^15,
16 places), a value far below the top has every place between its own top 1 and place 15 set opposite to place 15:
those places are the sign again. Per bank: the share of pixels at each place that are such a copy (equal to NOT place
15), for the light (the 30 refs texts, 150 positions; the top read: 30 last positions) and for the plate.
A place with no relation to the sign shows about 50%.
    python3 fringes/light_places.py
"""
import sys, os, json, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; os.chdir(HERE)
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]
src = src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")
src = src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
exec(src)
refs = json.load(open("refs_many.json")); inputs = [(r["ids"] + r["ref"])[:5] for r in refs]
plain_product = product
BANKS = {id(W[k][0]): k for k in W}
def copies(V):                                           # per place 0..14: pixels equal to NOT place 15, and the pixel count
    U = V.astype(np.int64) + (1 << 15); assert U.min() >= 0 and U.max() < (1 << 16)
    top = (U >> 15) & 1
    return np.array([int((((U >> a) & 1) != top).sum()) for a in range(15)]), U.size
CNT = {}
def product(x16, w16):
    key = BANKS.get(id(w16))
    if key is not None:
        c, t = copies(x16); C = CNT.setdefault(key, [np.zeros(15, np.int64), 0]); C[0] += c; C[1] += t
    return plain_product(x16, w16)
for ids in inputs:
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        R = R + plain_bank(edge(gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
    w, we, b = W["head"]; x16, ex, LS = gauge_log(R[-1:]); product(x16, w)
print("share of pixels that copy the sign, places 14 13 12 ... 4   (light | plate)")
for k in [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]:
    c, t = CNT[k]; pc, pt = copies(M[k + ".w"])
    L = " ".join(f"{100 * c[a] // t:3d}" for a in range(14, 3, -1)); Pl = " ".join(f"{100 * pc[a] // pt:3d}" for a in range(14, 3, -1))
    print(f"{k:8s} light {L}  | plate {Pl}")
