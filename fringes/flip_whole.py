"""THE FLIP, EXACT, WHOLE MACHINE (the up/down form of updown_machine.py, same offset, same banks, same light).
Every place of a 16-bit value is up (+1) or a hole (-1), D = 2v + 1. A meeting of light place a with plate place b over
the N lines is a count d_ab of lines where they disagree; tip = (N - 2 d_ab) / 2;
    pile = 1/4 [ S - token stack - neuron stack + N ],   S = sum_{a,b} 2^(a+b) (N - 2 d_ab),
    token stack = 2 x (sum of the light's values) + N,  neuron stack = 2 x (sum of the plate's values) + N.
Flipping every place of a value (1 -> hole, hole -> 1) turns v into -v - 1.

Three machines, run on every text (30 refs texts, 5 tokens each, as updown_machine.py):
  plain  base2.py's product (whole-number products) -> the final scores at the top (50,257 whole numbers)
  BOTH   every place of the light AND of the plate flipped at every meeting (every bank, the arrival's two products,
         the top read). Checked at every meeting: every count d_ab equal to the unflipped one; the pile from the flipped
         counts with the machine's stacks equal to base2's product (no correction). Also recorded: the pile you get if
         the stacks are taken from the flipped light and plate themselves.
  PLATE  only the plate flipped. Checked against the unflipped counts (recomputed in the same meeting): every count is
         N - d_ab (every tip changes sign), and the machine's pile is
         recovered from the flipped meeting S' by
             pile = 1/4 [ -S' - token stack - neuron stack + N ]
         equivalently, with the flipped plate's own pile p' = 1/4 [ S' - token stack - flipped neuron stack + N ]
             pile = -p' + (N - token stack) / 2 = -p' - (sum of the light's values)
  BOTH and PLATE each run the whole machine on the pile they produce; their final scores are compared to plain's.
    python3 fringes/flip_whole.py FIRST LAST        (texts FIRST..LAST-1)
"""
import sys, os, json, time, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; os.chdir(HERE)
FIRST, LAST = int(sys.argv[1]), int(sys.argv[2])
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]
src = src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")             # the machine itself never
src = src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")                    # uses these (memory)
exec(src)
refs = json.load(open("refs_many.json")); inputs = [(r["ids"] + r["ref"])[:5] for r in refs]
plain_product = product
BANKS = {id(W[k][0]): k for k in W}
INT16 = {}
def plate_values(key, w16):                              # the plate as stored (int16 for banks), no float->int64 copy of the top read
    if key is None: return np.asarray(w16).astype(np.int64)
    if key not in INT16: INT16[key] = M[key + ".w"]
    return INT16[key]
def places_needed(x, w): return max(int(np.abs(x).max()).bit_length(), max(int(w.max()), -int(w.min())).bit_length()) + 1

def place_rows(V, n, flip):                              # V: (rows, lines) values; offset 2^(n-1) as updown_machine
    R, N = V.shape; words = (N + 63) // 64               # each place's row of pixels, packed 64 lines a word: (n, rows, words)
    out = np.empty((n, R, words), np.uint64)
    for r0 in range(0, R, 4096):                         # in slices of rows (the top read has 50,257 neurons)
        U = V[r0:r0 + 4096].astype(np.int64) + (1 << (n - 1))
        assert U.min() >= 0 and U.max() < (1 << n)
        if flip: U = ((1 << n) - 1) - U                 # every 1 a hole, every hole a 1
        U = np.ascontiguousarray(U.astype(np.uint64))
        if words * 64 > N: U = np.concatenate([U, np.zeros((U.shape[0], words * 64 - N), np.uint64)], 1)  # pad lines 0 on both sides
        for a in range(n):
            pl = ((U >> np.uint64(a)) & np.uint64(1)).astype(np.uint8)
            out[a, r0:r0 + 4096] = np.packbits(pl, axis=-1, bitorder="little").view(np.uint64)
    return out

def disagree(Xr, Wr):                                    # d[a, row, b, neuron]: lines where light place a and plate place b disagree
    d = np.empty((Xr.shape[0], Xr.shape[1], Wr.shape[0], Wr.shape[1]), np.int64)
    for a in range(Xr.shape[0]):
        d[a] = np.bitwise_count(Xr[a][:, None, None, :] ^ Wr[None, :, :, :]).sum(-1, dtype=np.int64)
    return d

def meeting(d, N, n):                                    # S = sum_{a,b} 2^(a+b) (N - 2 d_ab)
    big = 2 * n + N.bit_length() + 2 > 62
    S = np.zeros((d.shape[1], d.shape[3]), dtype=object if big else np.int64)
    for a in range(n):
        for b in range(n):
            t = N - 2 * d[a, :, b, :]
            S = S + (t.astype(object) if big else t) * (1 << (a + b))
    return S

def quarter(V):
    assert (np.asarray(V % 4) == 0).all(); return (V // 4).astype(np.int64)

PLATE = {}
def plate_rows(key, w, n, flip):
    if key is None: return place_rows(w.T, n, flip)
    if (key, n, flip) not in PLATE: PLATE[(key, n, flip)] = place_rows(w.T, n, flip)
    return PLATE[(key, n, flip)]

STATS = {}                                               # per kind: calls, meetings, and every check's failures
def tally(kind, field, v):
    s = STATS.setdefault(kind, {}); s[field] = s.get(field, 0) + int(v)
def kind_of(key): return "arrival" if key is None else key.split(".")[0]

def product_both(x16, w16):
    key = BANKS.get(id(w16)); k = kind_of(key)
    x = x16.astype(np.int64); w = plate_values(key, w16); N = x.shape[1]; n = places_needed(x, w)
    ts = 2 * x.sum(1) + N; ws = 2 * w.sum(0, dtype=np.int64) + N
    d0 = disagree(place_rows(x, n, False), plate_rows(key, w, n, False))
    d1 = disagree(place_rows(x, n, True), plate_rows(key, w, n, True))
    S0, S1 = meeting(d0, N, n), meeting(d1, N, n)
    plain = plain_product(x16, w16)
    pile = quarter(S1 - ts[:, None] - ws[None, :] + N)                      # the machine's stacks: no correction
    own = quarter(S1 + ts[:, None] + ws[None, :] + N)                       # stacks taken from the flipped light and plate
    tally(k, "calls", 1); tally(k, "meetings", d0.size)
    tally(k, "counts_changed", (d1 != d0).sum())
    tally(k, "S_changed", (np.asarray(S1 != S0)).sum())
    tally(k, "unflipped_pile_wrong", (quarter(S0 - ts[:, None] - ws[None, :] + N) != plain).sum())
    tally(k, "pile_wrong", (pile != plain).sum()); tally(k, "piles", pile.size)
    tally(k, "ownstack_pile_equal", (own == plain).sum())
    tally(k, "ownstack_diff_not_(ts+ws)/2", ((own - plain) != (ts[:, None] + ws[None, :]) // 2).sum())
    return pile

def product_plate(x16, w16):
    key = BANKS.get(id(w16)); k = kind_of(key)
    x = x16.astype(np.int64); w = plate_values(key, w16); N = x.shape[1]; n = places_needed(x, w)
    ts = 2 * x.sum(1) + N; ws = 2 * w.sum(0, dtype=np.int64) + N
    wsf = 2 * (-w.sum(0, dtype=np.int64) - w.shape[0]) + N                       # the flipped plate's own stack: values -w-1
    dp = disagree(place_rows(x, n, False), plate_rows(key, w, n, True))
    d0 = disagree(place_rows(x, n, False), plate_rows(key, w, n, False)); S0 = meeting(d0, N, n)   # the unflipped meeting
    Sp = meeting(dp, N, n)
    plain = plain_product(x16, w16)
    pile = quarter(-Sp - ts[:, None] - ws[None, :] + N)                      # recovered: flipped meeting + stacks + N
    own = quarter(Sp - ts[:, None] - wsf[None, :] + N)                       # the flipped plate's own pile
    rec2 = -own + (N - ts[:, None]) // 2
    tally(k + "|plate", "calls", 1); tally(k + "|plate", "meetings", dp.size)
    tally(k + "|plate", "tips_not_negated", (dp != N - d0).sum())
    tally(k + "|plate", "S_not_negated", (np.asarray(Sp != -S0)).sum())
    tally(k + "|plate", "neg_stack_not_minus", (wsf != -ws).sum())
    tally(k + "|plate", "own_not_negative_plate_product", sum(int((own[:, c:c + 4096] != plain_product(x16, -np.asarray(w16)[:, c:c + 4096] - 1.0)).sum()) for c in range(0, own.shape[1], 4096)))
    tally(k + "|plate", "pile_wrong", (pile != plain).sum()); tally(k + "|plate", "piles", pile.size)
    tally(k + "|plate", "rec2_wrong", (rec2 != plain).sum())
    return pile

def run(ids, prod):
    global product
    product = prod; P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        R = R + plain_bank(edge(gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
    w, we, b = W["head"]; x16, ex, LS = gauge_log(R[-1:])
    return power(product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None])[0] + b

t0 = time.time(); scores_same = {"both": 0, "plate": 0}; words_same = {"both": 0, "plate": 0}
for i in range(FIRST, LAST):
    ids = inputs[i]
    s_plain = run(ids, plain_product)
    s_both = run(ids, product_both)
    s_plate = run(ids, product_plate)
    scores_same["both"] += np.array_equal(s_both, s_plain); scores_same["plate"] += np.array_equal(s_plate, s_plain)
    words_same["both"] += race(s_both) == race(s_plain); words_same["plate"] += race(s_plate) == race(s_plain)
    print(f"  text {i:2d}: final scores identical  both-flipped {np.array_equal(s_both, s_plain)}  "
          f"plate-flipped {np.array_equal(s_plate, s_plain)}   ({time.time() - t0:.0f}s)", flush=True)
out = {"texts": [FIRST, LAST], "scores_same": scores_same, "words_same": words_same, "stats": STATS}
json.dump(out, open(f"{HERE}/fringes/flip_{FIRST:02d}_{LAST:02d}.json", "w"), indent=1)
for k, v in STATS.items(): print(f"  {k:14s} {v}")
print(f"  final scores identical: {scores_same}  same word: {words_same}  of {LAST - FIRST}")
