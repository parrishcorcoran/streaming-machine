"""BASE2 WITH THE STORED ENTRY. base2.py's machine with one change: at layer 0's qkv bank the light never meets the
plate whole. It runs from the files in entry/ (built by entry/build.py for all 50,257 words and all 1,024 positions;
manifest in entry/manifest.json).

At qkv.0 a token's light is its word's row + its position's row, R = wte[w] + wpe[p]. The water level (base2's own
gauge_log, unchanged) puts it on 16 bits with shift s: x16 = rshift(R, s). base2.py meets x16 with the plate. Here:
  s = 4  (only ever at position 0):      the pile is read, ZP0[w]
  s = 0  (only ever at position 1023):   the pile is read, Z1023[w]
  s = 1 or 2 (every other case):         pile = T_s[w] + P_s[p] + the carry met with the plate, where
         T_s[w] = rshift(wte[w], s) @ plate and P_s[p] = rshift(wpe[p], s) @ plate are read from the files, and
         carry = x16 - rshift(wte[w], s) - rshift(wpe[p], s) is -1, 0 or +1 on each line. Each lit line pours its
         plate row, up for +1, down for -1. Nothing else meets the plate.
  s = 3 never occurs (entry/survey_s.py, all 51,463,168 pairs); the machine stops if it ever sees it.
Everything else (the water level, the gain, the bias, the arrival, the other 47 banks, the bend, the read at the top)
is base2.py's own code: loaded the way judge5.py loads it (base2.py's source up to the tokenizer), and called from
there.
    from base2_entry import forward, scores_all, entry_bank, entry_pile
    python3 base2_entry.py          # base2.py's 3 prompts: words against fp32 GPT-2 and against base2.py
"""
import sys, os, json, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")   # Accelerate's spurious flags; exact by the bounds below
HERE = "/Users/abundancemachine/streaming-machine"
if HERE not in sys.path: sys.path.insert(0, HERE)
import judge5 as J                                    # base2.py's own machine (its source, exec'd by judge5.py)
M, W, ES = J.M, J.W, J.ES
ENT = f"{HERE}/entry"; MAN = json.load(open(f"{ENT}/manifest.json"))
NEUR = MAN["neurons"]

class Table:                                          # one stored table, memory-mapped: rows read by word or position
    def __init__(self, name):
        t = MAN["tables"][name]; self.name, self.kind = name, t["kind"]
        ids = np.load(f"{ENT}/{t['ids']}").astype(np.int64)
        self.inv = np.full(int(ids.max()) + 1, -1, np.int64); self.inv[ids] = np.arange(len(ids))
        mm = lambda f: np.asarray(np.load(f"{ENT}/{f}.npy", mmap_mode="r"))       # a plain view of the mapped file
        if self.kind == "split40": self.lo, self.hi = mm(f"{name}_lo"), mm(f"{name}_hi")
        else: self.v = mm(name)
    def read(self, keys):                             # whole numbers: value = hi x 2^32 + lo
        keys = np.asarray(keys, np.int64); assert keys.max() < len(self.inv)
        r = self.inv[keys]; assert (r >= 0).all(), f"{self.name}: a row that was never stored"
        if self.kind == "split40": return self.hi[r].astype(np.int64) * (1 << 32) + self.lo[r]
        return self.v[r].astype(np.int64)
ZP0, Z1023 = Table("ZP0"), Table("Z1023")
T = {1: Table("T1"), 2: Table("T2")}; P = {1: Table("P1"), 2: Table("P2")}

PLATE16 = M["qkv.0.w"]                                # (768 lines, 2304 neurons), int16, |weight| <= 32,755 < 2^15
PLATE32F = PLATE16.astype(np.float32)                 # the same whole numbers, for the narrow meeting
PLATE64F = PLATE16.astype(np.float64)                 # the same whole numbers, for a position with more than 512 lit lines
LIT_MAX = 512                                         # 512 lit lines x 2^15 = 2^24: every partial sum exact in float32
MEET = "narrow"                                       # "narrow" (default) or "lit"; both meet only the carry, both exact
def carry_meet(c, how=None):
    """the carry met with the plate. Each product is -1, 0 or +1 times a plate weight. With at most 512 lit lines on a
    position, every partial sum stays within 512 x 2^15 = 2^24, so float32 adds them exactly in any order.
      narrow: the carry (all 768 lines, dark ones 0) times the plate as one float32 product. Fastest here.
      lit:    only the lit lines' plate rows are read, each poured with its sign (the literal pour).
    A position with more than 512 lit lines (none in the 30 texts) is met in float64, still only the carry."""
    c = np.asarray(c); how = how or MEET; n = len(c); out = np.empty((n, NEUR), np.int64)
    lit = np.abs(c).sum(1); ok = lit <= LIT_MAX
    if how == "narrow":
        if ok.all(): return (c.astype(np.float32) @ PLATE32F).astype(np.int64)
        out[ok] = (c[ok].astype(np.float32) @ PLATE32F).astype(np.int64)
    else:
        for i in np.flatnonzero(ok):
            idx = np.flatnonzero(c[i]); out[i] = (c[i, idx].astype(np.float32) @ PLATE32F[idx]).astype(np.int64)
    if not ok.all():
        out[~ok] = (c[~ok].astype(np.float64) @ PLATE64F).astype(np.int64)        # below 768 x 2^15 < 2^53: exact
    return out

ON_PILE = None                                        # a checker may watch the qkv.0 pile (verify.py)
def entry_pile(x16, ex, A, B, words, positions):
    """qkv.0's pile from the stored files: x16, ex from the water level; A, B the word and position rows (2^16)."""
    s = ex + ES; words = np.asarray(words, np.int64); positions = np.asarray(positions, np.int64)
    pile = np.empty((len(s), NEUR), np.int64)
    for sv in np.unique(s):
        m = np.flatnonzero(s == sv)
        if sv == 4:
            assert (positions[m] == 0).all(); pile[m] = ZP0.read(words[m])
        elif sv == 0:
            assert (positions[m] == 1023).all(); pile[m] = Z1023.read(words[m])
        elif sv in (1, 2):
            c = x16[m] - J.rshift(A[m], sv) - J.rshift(B[m], sv); assert np.abs(c).max() <= 1
            pile[m] = T[sv].read(words[m]) + P[sv].read(positions[m]) + carry_meet(c)
        else:
            raise ValueError(f"shift {sv} at qkv.0: entry/survey_s.py found it never occurs")
    if ON_PILE is not None: ON_PILE(pile)
    return pile

def entry_bank(R, A, B, words, positions):            # base2's gauged_bank for qkv.0, the product replaced by the files
    w, we, b = W["qkv.0"]                             # only the gain's powers and the bias are used
    x16, ex, LS = J.gauge_log(R)
    pile = entry_pile(x16, ex, A, B, words, positions)
    F = (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]
    return J.power(pile, F) + b

def stream(ids, wte=None, rec=None):                  # base2's layers, layer 0's qkv from the files
    ids = np.asarray(ids, np.int64); n = len(ids)
    A = (M["wte"] if wte is None else wte)[ids]; B = M["wpe"][:n]; R = A + B
    for l in range(12):
        Q = entry_bank(R, A, B, ids, np.arange(n)) if l == 0 else J.gauged_bank(R, f"qkv.{l}")
        R = R + J.plain_bank(J.attention(Q), f"proj.{l}")
        if rec is not None: rec.append(R)
        R = R + J.plain_bank(J.edge(J.gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
        if rec is not None: rec.append(R)
    return R

def forward(ids):                                     # base2.py's forward: the word after the last position
    R = stream(ids); w, we, b = W["head"]
    x16, ex, LS = J.gauge_log(R[-1:]); pile = J.product(x16, w)
    return J.race(J.power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None])[0] + b)

def scores_all(ids, wte=None):                        # judge5.scores_all with the stored entry
    R = stream(ids, wte); w, we, b = W["head"]; Rj = R[J.FIRST:J.LAST]; x16, ex, LS = J.gauge_log(Rj)
    return J.power(J.product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b

if __name__ == "__main__":
    import torch
    tok = J.GPT2Tokenizer.from_pretrained("gpt2"); ref = J.GPT2LMHeadModel.from_pretrained("gpt2").eval()
    WORDS = 8; same_all = True
    for prompt in ["The capital of France is", "Once upon a time", "In 1492, Columbus"]:
        ids = tok(prompt).input_ids
        want = ref.generate(torch.tensor([ids]), max_new_tokens=WORDS, do_sample=False, pad_token_id=50256)[0, len(ids):].tolist()
        got, base = [], []
        for _ in range(WORDS): got.append(forward(ids + got))
        for _ in range(WORDS): base.append(J.forward(ids + base))
        same = got == want == base; same_all &= same
        print(f"  {prompt!r:28s} {'IDENTICAL' if same else 'DIFFERENT'}  {tok.decode(got)!r}  (base2.py: {tok.decode(base)!r})")
    print("all three identical to fp32 GPT-2 and to base2.py" if same_all else "not all identical")
