"""BASE2 WITH LAYER 0 READ FROM STORAGE WHEREVER IT IS FIXED BY (WORD, POSITION). base2_entry.py's machine (step 1:
qkv.0 from entry/, only the -1/0/+1 carry meets that plate) with layer 0's other plates changed the same way. Tables in
layer0/ (built by layer0/build.py; manifest in layer0/manifest.json). Everything else is base2.py's own code, called
through judge5 (base2.py's source up to the tokenizer).

At layer 0, per token (word w at position p):
  position 0      nothing arrives but the token itself, so the whole of layer 0 is fixed by the word: the stream
                  after layer 0 is READ, Z0[w]. (q, k, v at position 0 still come from step 1's ZP0, for the later
                  tokens' arrival.)
  positions 1 on:
    qkv.0         step 1: T_s[w] + P_s[p] read, the carry met with the plate. q, k, v are then fixed by the pair.
    the arrival   base2.py's own: scores q x k over every earlier key, the race, the shares, the mix of values, the
                  divide. This is the KV: the only part of layer 0 that depends on the earlier tokens.
    proj.0        o16 = the arrival's output after its water level (shift so). The token's own value rounded at so,
                  ref = rshift(v_own, so), is fixed by the pair; its meeting is READ, OWN_so[(w, p)], and only
                  o16 - ref (what the earlier tokens' values add) meets the plate. OWN is per pair, so it holds only
                  the pairs of the 30 refs texts and the 300 judge windows; any other pair, or a shift not stored,
                  meets the whole o16 as base2.py does.
    fc.0          x1 = R1 after its water level (shift s1; R1 = word + position + the arrival's write). The word's
                  and the position's parts rounded at s1 are READ, FT_s1[w] + FP_s1[p] (the position row carries
                  proj.0's bias); only x1 - those two rows (the arrival write's part, rounded at s1, plus a -1/0/+1
                  carry) meets the plate. A shift other than 5 or 6 meets the whole x1 as base2.py does.
    the bend, mproj.0   base2.py's own.
    from base2_layer0 import forward, scores_all, stream, layer0, FALLBACK
    python3 base2_layer0.py          # base2.py's 3 prompts: words against fp32 GPT-2 and against base2.py
"""
import sys, os, json, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")   # Accelerate's spurious flags; exact by the bounds
HERE = "/Users/abundancemachine/streaming-machine"
if HERE not in sys.path: sys.path.insert(0, HERE)
import base2_entry as E                               # step 1 (and, through it, base2.py's own machine)
J = E.J; M, W, ES = J.M, J.W, J.ES
L0 = f"{HERE}/layer0"; MAN0 = json.load(open(f"{L0}/manifest.json"))

class Table:                                          # one stored table, memory-mapped: rows read by key
    def __init__(self, name):
        t = MAN0["tables"][name]; self.name, self.kind = name, t["kind"]
        self.ids = np.load(f"{L0}/{t['ids']}").astype(np.int64); assert (np.diff(self.ids) > 0).all()   # sorted keys
        mm = lambda f: np.asarray(np.load(f"{L0}/{f}.npy", mmap_mode="r"))
        if self.kind == "split40": self.lo, self.hi = mm(f"{name}_lo"), mm(f"{name}_hi")
        else: self.v = mm(name)
    def rows(self, keys):                             # the row of each key, -1 where it was never stored
        keys = np.asarray(keys, np.int64); r = np.minimum(np.searchsorted(self.ids, keys), len(self.ids) - 1)
        return np.where(self.ids[r] == keys, r, -1)
    def has(self, keys): return self.rows(keys) >= 0
    def read(self, keys):                             # whole numbers (split40: value = hi x 2^32 + lo)
        r = self.rows(keys); assert (r >= 0).all(), f"{self.name}: a row that was never stored"
        if self.kind == "split40": return self.hi[r].astype(np.int64) * (1 << 32) + self.lo[r]
        return self.v[r].astype(np.int64)
Z0 = Table("Z0")
FT = {5: Table("FT5"), 6: Table("FT6")}; FP = {5: Table("FP5"), 6: Table("FP6")}
OWN = {1: Table("OWN1"), 2: Table("OWN2"), 3: Table("OWN3")}

WTE = M["wte"]; WPE = M["wpe"]                        # int64 at 2^16, read once
WP, WEP, BP = W["proj.0"]; WF, WEF, BF = W["fc.0"]
FALLBACK = {"proj pairs not stored": 0, "proj shift not stored": 0, "fc shift not stored": 0}

def proj_bank(o, v_own, keys):
    """base2's plain_bank for proj.0, the product split: the own value's part read, the rest met.
    Returns the bank's output, and (pile, o16, so) for a checker."""
    o16, eo = J.to16(o, -ES); so = eo + ES; n = len(o)
    pile = np.empty((n, 768), np.int64)
    for sv in np.unique(so):
        m = np.flatnonzero(so == sv)
        has = OWN[sv].has(keys[m]) if sv in OWN else np.zeros(len(m), bool)
        st, pl = m[has], m[~has]
        if len(st):
            ref = J.rshift(v_own[st], sv)                                   # the token's own value at o's shift
            pile[st] = OWN[sv].read(keys[st]) + J.product(o16[st] - ref, WP)
        if len(pl):                                                         # not stored: the whole o16 meets the plate
            pile[pl] = J.product(o16[pl], WP)
            FALLBACK["proj shift not stored" if sv not in OWN else "proj pairs not stored"] += len(pl)
    return J.rshift(pile, -(WEP[None, :] + eo[:, None] + ES)) + BP, (pile, o16, so)

def fc_bank(R1, words, positions):
    """base2's gauged_bank for fc.0, the product split: word and position parts read, the rest met."""
    x1, e1, LS = J.gauge_log(R1); s1 = e1 + ES; n = len(R1)
    pile = np.empty((n, 3072), np.int64); rest = np.zeros((n, 768), np.int64)
    for sv in np.unique(s1):
        m = np.flatnonzero(s1 == sv)
        if sv in FT:
            a_, b_ = J.rshift(WTE[words[m]], sv), J.rshift(WPE[positions[m]] + BP[None, :], sv)
            rest[m] = x1[m] - a_ - b_                                       # the arrival write's part + the carry
            pile[m] = FT[sv].read(words[m]) + FP[sv].read(positions[m]) + J.product(rest[m], WF)
        else:                                                               # not stored: the whole x1 meets the plate
            pile[m] = J.product(x1[m], WF); rest[m] = x1[m]; FALLBACK["fc shift not stored"] += len(m)
    F = (WEF[None, :] + e1[:, None] + ES) * 65536 - LS[:, None]
    return J.power(pile, F) + BF, (pile, x1, s1, LS, rest)

def layer0(ids, A=None, B=None, rec=None):
    """layer 0 for a whole text at once (every position, as base2.py's forward runs it). Returns the stream after it."""
    ids = np.asarray(ids, np.int64); n = len(ids); pos = np.arange(n)
    A = WTE[ids] if A is None else A; B = WPE[:n] if B is None else B; R = A + B
    QKV = E.entry_bank(R, A, B, ids, pos)                                   # step 1
    R2 = np.empty_like(R)
    first = pos == 0
    if first.any(): R2[first] = Z0.read(ids[first])                         # position 0: read whole
    if rec is not None: rec.update(QKV=QKV, R=R)
    r = np.flatnonzero(~first)
    if len(r):
        o = J.attention(QKV)[r]                                             # the arrival, base2.py's own
        att, (ppile, o16, so) = proj_bank(o, QKV[r, 1536:], ids[r] * 1024 + r)
        R1 = R[r] + att
        H, (fpile, x1, s1, LS, rest) = fc_bank(R1, ids[r], r)
        G = J.edge(H)
        R2[r] = R1 + J.plain_bank(G, "mproj.0")
        if rec is not None: rec.update(rows=r, o=o, proj_pile=ppile, o16=o16, so=so, att=att, R1=R1, fc_pile=fpile,
                                       x1=x1, s1=s1, LS=LS, rest=rest, H=H, G=G)
    if rec is not None: rec["R2"] = R2
    return R2

def stream(ids, wte=None, rec=None):                  # base2's layers, layer 0 from the stored tables
    ids = np.asarray(ids, np.int64); n = len(ids)
    A = (WTE if wte is None else wte)[ids]; B = WPE[:n]
    R = layer0(ids, A, B, rec)
    if rec is not None and "streams" in rec: rec["streams"].append(R)
    for l in range(1, 12):
        R = R + J.plain_bank(J.attention(J.gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        if rec is not None and "streams" in rec: rec["streams"].append(R)
        R = R + J.plain_bank(J.edge(J.gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
        if rec is not None and "streams" in rec: rec["streams"].append(R)
    return R

def forward(ids):                                     # base2.py's forward: the word after the last position
    R = stream(ids); w, we, b = W["head"]
    x16, ex, LS = J.gauge_log(R[-1:]); pile = J.product(x16, w)
    return J.race(J.power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None])[0] + b)

def scores_all(ids, wte=None):                        # judge5.scores_all with layer 0 from storage
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
    print(f"fallbacks (pile met whole, as base2.py): {FALLBACK}")
