"""The top as a waveform that collapses: read every word with the top 8 binary digits of its stamp, bound what the
bottom 8 could add, keep only words that could still win, read those exactly, race. Same word as the full read?
    python3 collapse.py
"""
import sys, json, numpy as np
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
refs = json.load(open("refs_many.json"))
w, we, b = W["head"]; wi = w.astype(np.int64)
whi = (wi >> 8) << 8; wlo = wi - whi                                   # top 8 digits | bottom 8 digits (0..255)
def tops(ids):
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        R = R + plain_bank(edge(gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
    return gauge_log(R)
surv, same, n = [], 0, 0
for r in refs:
    ids = r["ids"] + r["ref"][:16]
    x16, ex, LS = tops(ids)
    for p in range(len(r["ids"]) - 1, len(ids) - 1):
        x = x16[p].astype(np.int64)
        coarse = x @ whi                                                # the coarse waveform, all words
        pos, neg = np.clip(x, 0, None) @ wlo, np.clip(-x, 0, None) @ wlo   # what the bottom digits could add (exactly bounded: wlo in 0..255)
        F = (we + ex[p] + ES) * 65536 - LS[p]
        lo = power((coarse - neg)[None, :], F[None, :])[0] + b          # value bounds per word (power is increasing)
        hi = power((coarse + pos)[None, :], F[None, :])[0] + b
        cand = np.flatnonzero(hi >= lo.max())                           # words that could still win
        exact = power(((x @ wi[:, cand]))[None, :], F[None, cand])[0] + b[cand]
        win = int(cand[race(exact)])
        full = race(power((x @ wi)[None, :], F[None, :])[0] + b)
        surv.append(len(cand)); same += win == full; n += 1
print(f"{n} pieces: survivors after the coarse read  median {int(np.median(surv))}, mean {np.mean(surv):.1f}, max {max(surv)} (of 50,257)")
print(f"same word as the full read: {same}/{n}")
print(f"work at the top: 50,257 coarse (8 digits) + {np.mean(surv):.0f} exact, instead of 50,257 exact")
