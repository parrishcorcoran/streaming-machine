"""One new piece's grains, depth by depth: poured -> left after the adding -> past the edge -> kept at the 16-digit cut.
Shares only (each depth counts in its own size of grain). Averaged over the 16 pieces after a prompt, the machine hot.
    python3 grain_budget.py
"""
import sys, json, numpy as np
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
refs = json.load(open("refs_many.json")); r = refs[0]; ids = r["ids"] + r["ref"][:16]; S = len(r["ids"]); P = len(ids)
rows = slice(S, P)
def pour(x16, w):                     # all ones poured (sizes) and what's left (net), per pile, for the new pieces
    xa = np.abs(x16[rows]).astype(np.float64); return (xa @ np.abs(w)).sum(), np.abs(x16[rows].astype(np.float64) @ w).sum()
def cutkeep(X):                       # share of the size kept when each piece's values are cut back to 16 digits
    x16, ex = to16(X, -ES); kept = np.abs(x16[rows].astype(np.float64) * 2.0 ** (ex[rows, None] + ES)).sum(); return kept / np.abs(X[rows]).astype(np.float64).sum()
print("depth   slab      left after adding   past the edge   kept at the 16-digit cut")
R = M["wte"][ids] + M["wpe"][:P]
for l in range(12):
    w = W[f"qkv.{l}"][0]; x16, _, _ = gauge_log(R); a, b = pour(x16, w); c1 = cutkeep(R)
    A = attention(gauged_bank(R, f"qkv.{l}"))
    w2 = W[f"proj.{l}"][0]; x2, _ = to16(A, -ES); a2, b2 = pour(x2, w2); c2 = cutkeep(A)
    R = R + plain_bank(A, f"proj.{l}")
    w3 = W[f"fc.{l}"][0]; x3, _, _ = gauge_log(R); a3, b3 = pour(x3, w3)
    Hb = gauged_bank(R, f"fc.{l}"); H = edge(Hb); edge_share = np.abs(H[rows]).astype(np.float64).sum() / np.abs(Hb[rows]).astype(np.float64).sum()
    w4 = W[f"mproj.{l}"][0]; x4, _ = to16(H, -ES); a4, b4 = pour(x4, w4); c4 = cutkeep(H)
    R = R + plain_bank(H, f"mproj.{l}")
    print(f"  {l:2d}    first     {100*b/a:6.2f}%                            {100*c1:8.4f}%")
    print(f"        second    {100*b2/a2:6.2f}%                            {100*c2:8.4f}%")
    print(f"        third     {100*b3/a3:6.2f}%          {100*edge_share:6.2f}%")
    print(f"        fourth    {100*b4/a4:6.2f}%                            {100*c4:8.4f}%")
