"""THE UP/DOWN MACHINE. GPT-2 in base 2 with every meeting written without zeros: each value is n places, every place a
photon going up (+1) or a hole (-1) (D = 2v + 1). A meeting of a token's lines with a neuron's lines is
    pile = 1/4 [ sum over place pairs (a, b) of 2^(a+b) (lines - 2 x lines where the two disagree)
                 - token stack - neuron stack + lines ]
where "lines where they disagree" is one count over the whole row at once, the token stack is one number a token
(shared by every neuron) and the neuron stack one number a neuron (recorded once). Every bank, the arrival's two
products, and the word read. Words out of 30 against the plain machine.
    python3 updown_machine.py
"""
import sys, json, time, numpy as np
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
refs = json.load(open("refs_many.json")); inputs = [(r["ids"] + r["ref"])[:5] for r in refs]
plain_product = product; COUNTS = [0]
def rows(B, n):                                        # each place's row of up/hole pixels, packed 64 lines a word: (n, rows, words)
    B = np.ascontiguousarray(B.astype(np.uint64)); N = B.shape[-1]; pad = (-N) % 64
    if pad: B = np.concatenate([B, np.zeros(B.shape[:-1] + (pad,), np.uint64)], -1)
    planes = ((B[None] >> np.arange(n, dtype=np.uint64)[:, None, None]) & np.uint64(1)).astype(np.uint8)
    return np.packbits(planes, axis=-1, bitorder="little").view(np.uint64)
PLATE = {}
def updown(x, w, key=None):
    x = x.astype(np.int64); w = np.asarray(w).astype(np.int64); N = x.shape[1]
    n = max(int(np.abs(x).max()).bit_length(), int(np.abs(w).max()).bit_length()) + 1; off = 1 << (n - 1)
    if key is not None and (key, n) in PLATE: Wr, ws = PLATE[(key, n)]
    else:
        Wr = rows((w + off).T, n); ws = 2 * w.sum(0) + N                     # neuron rows; neuron stack (recorded once)
        if key is not None: PLATE[(key, n)] = (Wr, ws)
    Xr = rows(x + off, n); ts = 2 * x.sum(1) + N                              # token rows; token stack (one number a token)
    big = 2 * n + N.bit_length() + 2 > 62
    S = np.zeros((x.shape[0], w.shape[1]), dtype=object if big else np.int64)
    for a in range(n):
        dis = np.bitwise_count(Xr[a][:, None, None, :] ^ Wr[None, :, :, :]).sum(-1, dtype=np.int64)   # (rows, n places, neurons)
        agree = N - 2 * dis; COUNTS[0] += dis.size
        for b in range(n): S = S + (agree[:, b, :].astype(object) if big else agree[:, b, :]) * (1 << (a + b))
    return ((S - ts[:, None] - ws[None, :] + N) // 4).astype(np.int64)
BANKS = {id(W[k][0]): k for k in W}
def product(x16, w16):
    return updown(x16, w16, BANKS.get(id(w16)))
def run(ids, prod):
    global product
    product = prod; P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        R = R + plain_bank(edge(gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
    w, we, b = W["head"]; x16, ex, LS = gauge_log(R[-1:])
    return race(power(product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None])[0] + b)
updown_product = product
# check one bank first: identical piles
x = np.random.default_rng(0).integers(-2**15 + 1, 2**15, (5, 768)); w = W["fc.0"][0]
print(f"  one bank, identical piles: {np.array_equal(updown(x, w), plain_product(x, w))}", flush=True)
t0 = time.time(); right = 0
for i, ids in enumerate(inputs):
    a, b = run(ids, plain_product), run(ids, updown_product); right += a == b
    if i % 5 == 4: print(f"  {i + 1:2d} pieces: up/down machine gives the same word {right} / {i + 1}   ({time.time() - t0:.0f}s)", flush=True)
print(f"  counts taken (each over a whole row of lines): {COUNTS[0]:,}")
