"""THE PLATE MACHINE, PHASE 1 (spec approved by Parrish 2026-10-08; his corrections: no old repos, no quantization
comparisons). GPT-2 as plates and light, simulated exactly in whole numbers. base2.py is never edited.

 1a THE PLATES: every bank's whole numbers from base2.py (48 banks + the top read's word plate), each in up/hole form
    (signed digits, no two lit side by side), one sheet per place. Nothing dropped. Reported: stored size, as is.
 1b THE LIGHT: at each bank, each lane written exactly as base2.py writes it (16 places at its own power, to16), in
    up/hole form. Each lit place is one photon sent; every photon counted.
 1c THE DETECTORS: at each pile, for each pair (light place a, plate place b), the detector reads the net:
    up-meets-up + hole-meets-hole - up-meets-hole - hole-meets-up. One whole number, computed exactly here; the
    meetings themselves are not counted (a plate makes them by interference). Counted: one detection per pile per
    (a, b) with light at a (this position, any lane) and plate at b (this pile, any line).
 1d THE PILE: sum of 2^(a+b) x each net; one add per detection.
 1e EVERYTHING ELSE exactly as base2.py (water levels, bends, the arrivals over the record with each position computed
    once, biases, stream adds, the top read's race), counted as base2_fringes2.py counts them: arrival 8,020,119 and
    rest 4,558,811 a token (flat_build/1/machine/gpt2_report.log, the same 648 refs positions). The race's compares
    are a count, kept apart.
 1f CHECK, pass or fail: every pile identical to base2.py's own product (asserted at every bank and the top read);
    every word identical: the 30 refs texts and judge5's 6,000 positions (reference top 5 = base2.py's own).
 1g REPORT: a token (the 648 refs positions): photons sent, detections, adds, the rest, total; against plain (rebuilt
    here: lit light x lit plate in binary 1s; checked against 3,666,596,259), the counting machine (18.86%) and the
    extra (6.24%).

PREDICTION (written before any run): every pile and word identical; about 200-250 detections a pile; about 133,000
piles a token (82,944 in the banks + 50,257 at the top read); about 30M detections, 30M adds, 13M the rest: about 75M
a token, about 2% of plain.
    python3 plate_machine.py refs | windows a b | report      (run from ~/streaming-machine/platemachine)
"""
import sys, os, json, time, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")   # the Mac library's spurious float warnings; every pile is asserted exact
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/platemachine/out"; os.makedirs(OUT, exist_ok=True)
_src = open(f"{HERE}/base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
_src = _src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np").replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer", "")   # unused by the functions here
_cwd = os.getcwd(); os.chdir(HERE); _argv = sys.argv; sys.argv = ["x"]; NS = {}; exec(_src, NS); sys.argv = _argv; os.chdir(_cwd)
M, W, ES = NS["M"], NS["W"], NS["ES"]
for _k in list(W): W[_k] = (M[_k + ".w"], W[_k][1], W[_k][2])   # the same plate values, held as their 16-place whole numbers (memory only)
ORIG = NS["product"]
BANK = {id(W[k][0]): k for k in W}                    # which plate a product is reading
NPL = 17                                              # up/hole places: 16 whole-number places can need one more
def updown(A):
    """signed digits, place by place: +1 up, -1 hole, 0 dark (no two lit side by side); sheets (NPL, ...) int8.
    Done in slices of columns so the word plate stays small in memory."""
    out = np.zeros((NPL,) + A.shape, np.int8); step = 4096
    for c0 in range(0, A.shape[-1], step):
        a = np.abs(A[..., c0:c0 + step]).astype(np.int32); s = np.sign(A[..., c0:c0 + step]).astype(np.int8)  # values < 2^16
        for p in range(NPL):
            d = np.where((a & 1) == 1, 2 - (a & 3), 0); out[p][..., c0:c0 + step] = d * s; a = (a - d) >> 1
        assert not a.any()
    return out
def ones(A): return np.bitwise_count(np.abs(np.asarray(A, np.int64))).astype(np.int64)
C = {}                                                # counts, by part
def add(part, k, v): C.setdefault(part, {}); C[part][k] = C[part].get(k, 0) + int(v)
HEAD = {}
def sheets_of(name, w):
    if name == "head":
        if not HEAD:
            Ws = updown(w.astype(np.int32)); HEAD["s"] = Ws; HEAD["B"] = (Ws != 0).any(1).sum(0)
            HEAD["o"] = np.bitwise_count(np.abs(w.astype(np.int32))).astype(np.int64).sum(1)   # each line's 1s over all words
        return HEAD["s"], HEAD["B"], HEAD["o"]
    Ws = updown(w.astype(np.int64)); return Ws, (Ws != 0).any(1).sum(0), ones(w).sum(1)    # each line's 1s over all piles
CHK = {"piles": 0, "differing": 0}
def plate_product(x16, w):
    name = BANK.get(id(w))
    if name is None: return ORIG(x16, w)              # the arrival's own products (1e)
    ref = ORIG(x16, w)
    X = np.asarray(x16, np.int64); P = X.shape[0]
    Ls = updown(X); Ws, B, wo = sheets_of(name, w)
    Lf = Ls.reshape(NPL * P, -1).astype(np.float32); pile = np.zeros(ref.shape, np.int64)
    sh = np.arange(NPL, dtype=np.int64)[:, None, None]
    for c0 in range(0, ref.shape[1], 8192):                   # the piles in slices (memory only)
        c1 = min(c0 + 8192, ref.shape[1])
        for b in range(NPL):
            Wb = Ws[b][:, c0:c1]
            if not Wb.any(): continue
            net = np.rint(Lf @ Wb.astype(np.float32)).astype(np.int64).reshape(NPL, P, -1)   # every detector at plate place b
            pile[:, c0:c1] += np.left_shift(np.left_shift(net, sh).sum(0), b)
    CHK["piles"] += pile.size; CHK["differing"] += int((pile != ref).sum())
    assert np.array_equal(pile, ref), name
    part = "top" if name == "head" else name.split(".")[0]
    A = (Ls != 0).any(2).sum(0)                       # per position: light places lit
    det = int(A.sum()) * int(B.sum())
    add(part, "photons", (Ls != 0).sum()); add(part, "detections", det); add(part, "adds", det)
    add(part, "piles", P * ref.shape[1]); add(part, "plain", int((ones(X) @ wo).sum()))
    return ref
NS["product"] = plate_product
def race_count(vals):                                 # base2.race's compares (count only)
    u = vals.astype(np.int64).view(np.uint64) ^ np.uint64(1 << 63); alive = np.ones(len(u), bool); n = 0
    for bit in range(63, -1, -1):
        n += int(alive.sum()); has = alive & (((u >> np.uint64(bit)) & np.uint64(1)) == 1)
        if has.any(): alive = has
    return n, int(np.flatnonzero(alive)[0])
def forward_all(ids, first=0, last=None):
    """base2.forward's own steps, the top read at every position in [first, last)"""
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        R = R + NS["plain_bank"](NS["attention"](NS["gauged_bank"](R, f"qkv.{l}")), f"proj.{l}")
        R = R + NS["plain_bank"](NS["edge"](NS["gauged_bank"](R, f"fc.{l}")), f"mproj.{l}")
    w, we, b = W["head"]; Rj = R[first:last]
    x16, ex, LS = NS["gauge_log"](Rj)
    pile = NS["product"](x16, w)
    return NS["power"](pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b
if __name__ == "__main__":
    mode = sys.argv[1]; t0 = time.time()
    if mode == "refs":
        refs = json.load(open(f"{HERE}/refs_many.json")); npos = 0; races = 0; words = []
        for i, r in enumerate(refs):
            ids = r["ids"] + r["ref"]; S = forward_all(ids); npos += len(ids)
            for row in S: n, wd = race_count(row); races += n; words.append(wd)
            print(f"  text {i+1}/30 ({time.time()-t0:.0f}s)", flush=True)
        json.dump(dict(C=C, CHK=CHK, npos=npos, races=races, words=words), open(f"{OUT}/refs.json", "w"))
    elif mode == "windows":
        a, b = int(sys.argv[2]), int(sys.argv[3]); ref5 = np.load(f"{HERE}/judge5_ref.npz")["top5"]; inn = same = tot = 0
        WIN = json.load(open(f"{HERE}/judge5_texts.json"))
        for i in range(a, b):
            win = WIN[i]
            S = forward_all(win["ids"], 3, 23); pick = np.array([race_count(row)[1] for row in S])
            inn += int((ref5[i] == pick[:, None]).any(1).sum()); same += int((ref5[i][:, 0] == pick).sum()); tot += len(pick)
            if (i - a) % 25 == 24: print(f"  window {i+1} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(dict(inn=inn, same=same, tot=tot, CHK=CHK), open(f"{OUT}/windows_{a}_{b}.json", "w"))
    print(f"{mode} done ({time.time()-t0:.0f}s); piles checked {CHK['piles']:,}, differing {CHK['differing']}")
