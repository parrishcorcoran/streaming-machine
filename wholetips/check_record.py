"""Follow-up to check_independent.py: the record's judge as wholetips.py and shrink/ define it -- the next word after
(prompt + refs words)[:5] -- for tips k = 1, k = 2 and the 4-bit grid, against the uncut fp32 machine's own word there.
Also the share of numbers put dark and each number's error as a share of itself, for the word table and two plates.
Reuses check_independent.py's definitions (exec of its source up to the words run), nothing from wholetips.py."""
src = open("/Users/abundancemachine/streaming-machine/wholetips/check_independent.py").read()
src = src.replace("x = A.ravel().astype(np.float64)", "x = A.ravel()").replace("xx = x[i0:i0 + CH]; neg", "xx = x[i0:i0 + CH].astype(np.float64); neg")
src = src.replace("CH = 1 << 19", "CH = 1 << 18")
assert "xx = x[i0:i0 + CH].astype(np.float64)" in src
exec(src[:src.index("REF = words()")])
INP = [(r["ids"] + r["ref"])[:5] for r in REFS]
def rec(): return [int(MODEL(torch.tensor([x])).logits[0, -1].argmax()) for x in INP]
R0 = rec()
for ver in [("tips", 1), ("tips", 2), ("grid", 4)]:
    CUT, BITS, _ = build(ver); install(CUT); r = rec(); install(S)
    say(f"{ver}: record's judge {sum(int(a == b) for a, b in zip(r, R0))}/30")
    for k in ["wte.weight", "h.5.mlp.c_fc.weight", "h.5.attn.c_attn.weight"]:
        o = rows(k, S[k])[:8192].astype(np.float64); c = rows(k, CUT[k])[:8192].astype(np.float64); nz = o != 0
        ratio = np.abs(o).max(1) / np.median(np.abs(o), 1)
        say(f"   {k}: dark {100 * (c == 0).mean():.2f}%  median error/own size {np.median(np.abs(c - o)[nz] / np.abs(o)[nz]):.4f}  "
            f"row's largest / middle {np.median(ratio):.2f}")
    del CUT
