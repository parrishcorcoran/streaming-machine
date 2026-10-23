"""Probe: how many places n each meeting uses (n = top place of light or plate + 1, as updown_machine sets it), at every
bank, on the 30 refs texts (5 tokens each). No changes to the machine."""
import sys, json, time, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"
src = open(f"{HERE}/base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]
import os; os.chdir(HERE); exec(src)
refs = json.load(open(f"{HERE}/refs_many.json")); inputs = [(r["ids"] + r["ref"])[:5] for r in refs]
plain_product = product
BANKS = {id(W[k][0]): k for k in W}
NS = {}
def product(x16, w16):
    k = BANKS.get(id(w16), "arrival")
    x = x16.astype(np.int64); w = np.asarray(w16).astype(np.int64)
    n = max(int(np.abs(x).max()).bit_length(), int(np.abs(w).max()).bit_length()) + 1
    NS.setdefault(k.split(".")[0], {}).setdefault(n, 0); NS[k.split(".")[0]][n] += 1
    return plain_product(x16, w16)
t0 = time.time()
for ids in inputs:
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        R = R + plain_bank(edge(gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
    w, we, b = W["head"]; x16, ex, LS = gauge_log(R[-1:]); product(x16, w)
print(NS, f"{time.time()-t0:.1f}s")
