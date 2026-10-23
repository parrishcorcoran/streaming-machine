"""THE FRINGE MAP. Every bank of all 12 layers (qkv, proj, fc, mproj), and the top read kept apart, on the 30 refs
texts (5 tokens each, as updown_machine.py: every position of every text, 150 positions a bank; the top read meets
only the last position, 30 positions). Up/down form as updown_machine.py: offset 2^(n-1), D = 2v + 1, n = 16 places.

For every meeting (position x neuron x light place a x plate place b): d = lines where they disagree,
    tip = (N - 2d) / 2,   straight-through = the pixels that pair off = N - 2|tip|,   fringe = 2|tip|.
Recorded: the count of meetings at every |tip| value, for every place pair (a, b), per bank.
CONTROL: the same meetings with each position's light lines put in their own random order (same values, pattern
broken; plate untouched).
Also checked at every bank: the pile from the real counts equals base2.py's product (the counts are the machine's).

PREDICTIONS (written before running, as given):
  - fringe share about 2-3% of each meeting in every bank;
  - above-chance fringes only in the top corner (light's top 1-2 places x weight's top 3-4 places) in every layer;
  - chance-size tips everywhere else.
    python3 fringes/fringe_map.py          -> fringes/fringe_hists.npz   (report: fringes/fringe_report.py)
    python3 fringes/fringe_map.py 2        -> fringes/fringe_hists_control2.npz: a second, independent control only
                                              (another random order), to see how far two controls differ from each other
"""
import sys, os, json, time, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; os.chdir(HERE)
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv_saved = sys.argv; sys.argv = ["x"]
src = src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")             # the machine itself never
src = src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")                     # uses these (memory)
exec(src)
SALT = int(sys.argv_saved[1]) if len(sys.argv_saved) > 1 else 0
refs = json.load(open("refs_many.json")); inputs = [(r["ids"] + r["ref"])[:5] for r in refs]
plain_product = product
BANKS = {id(W[k][0]): k for k in W}
KINDS = ["qkv", "proj", "fc", "mproj", "head"]

def place_rows(V, n):                                    # V: (rows, lines) values; offset 2^(n-1); (n, rows, words) packed
    R, N = V.shape; words = (N + 63) // 64
    out = np.empty((n, R, words), np.uint64)
    for r0 in range(0, R, 4096):
        U = V[r0:r0 + 4096].astype(np.int64) + (1 << (n - 1))
        assert U.min() >= 0 and U.max() < (1 << n)
        U = np.ascontiguousarray(U.astype(np.uint64))
        if words * 64 > N: U = np.concatenate([U, np.zeros((U.shape[0], words * 64 - N), np.uint64)], 1)
        for a in range(n):
            pl = ((U >> np.uint64(a)) & np.uint64(1)).astype(np.uint8)
            out[a, r0:r0 + 4096] = np.packbits(pl, axis=-1, bitorder="little").view(np.uint64)
    return out

def disagree(Xr, Wr):                                    # d[a, row, b, neuron]
    d = np.empty((Xr.shape[0], Xr.shape[1], Wr.shape[0], Wr.shape[1]), np.int64)
    for a in range(Xr.shape[0]):
        d[a] = np.bitwise_count(Xr[a][:, None, None, :] ^ Wr[None, :, :, :]).sum(-1, dtype=np.int64)
    return d

def meeting(d, N, n):
    S = np.zeros((d.shape[1], d.shape[3]), np.int64)
    for a in range(n):
        for b in range(n): S += (N - 2 * d[a, :, b, :]) * (1 << (a + b))
    return S

PLATE, HIST, INFO = {}, {}, {}
def hist_add(name, d, N):                                # count of meetings at each |tip|, per place pair
    n = d.shape[0]; half = N // 2
    if name not in HIST: HIST[name] = np.zeros((n, n, half + 1), np.int64)
    idx = (np.arange(n)[:, None, None, None] * n + np.arange(n)[None, None, :, None]) * (half + 1) + np.abs(half - d)
    HIST[name] += np.bincount(idx.ravel(), minlength=HIST[name].size).reshape(HIST[name].shape)

TEXT = [0]; BAD = [0]
def product(x16, w16):
    pile = plain_product(x16, w16)
    key = BANKS.get(id(w16))
    if key is None: return pile                          # the arrival's two products: not banks
    x = x16.astype(np.int64)
    if key not in PLATE: PLATE[key] = M[key + ".w"]
    w = PLATE[key]; N = x.shape[1]
    n = max(int(np.abs(x).max()).bit_length(), max(int(w.max()), -int(w.min())).bit_length()) + 1
    assert n == 16, (key, n)
    if (key, "rows") not in PLATE: PLATE[(key, "rows")] = place_rows(w.T, n)
    Wr = PLATE[(key, "rows")]
    if SALT == 0:
        d = disagree(place_rows(x, n), Wr)
        S = meeting(d, N, n); ts = 2 * x.sum(1) + N; ws = 2 * w.sum(0, dtype=np.int64) + N
        V = S - ts[:, None] - ws[None, :] + N; BAD[0] += int((V % 4 != 0).sum() + (V // 4 != pile).sum())
        hist_add(key, d, N); del d
    l, kind = (int(key.split(".")[1]), KINDS.index(key.split(".")[0])) if key != "head" else (12, 4)
    rng = np.random.default_rng([TEXT[0], l, kind] + ([SALT] if SALT else []))
    xc = rng.permuted(x, axis=1)                         # each position's lines in their own random order
    assert (np.sort(xc, 1) == np.sort(x, 1)).all()
    hist_add(key + "|control", disagree(place_rows(xc, n), Wr), N)
    INFO[key] = INFO.get(key, 0) + x.shape[0]
    return pile

t0 = time.time()
for i, ids in enumerate(inputs):
    TEXT[0] = i; P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        R = R + plain_bank(edge(gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
    w, we, b = W["head"]; x16, ex, LS = gauge_log(R[-1:]); product(x16, w)
    if i % 5 == 4: print(f"  {i + 1} texts ({time.time() - t0:.0f}s)  piles from the counts not equal to the machine's: {BAD[0]}", flush=True)
np.savez(f"{HERE}/fringes/fringe_hists" + (f"_control{SALT}" if SALT else "") + ".npz", **{k.replace("|", "__"): v for k, v in HIST.items()},
         positions=json.dumps(INFO), bad=BAD[0])
print(f"  saved fringes/fringe_hists" + (f"_control{SALT}" if SALT else "") + f".npz  positions per bank: {sorted(set(INFO.values()))}  ({time.time() - t0:.0f}s)")
