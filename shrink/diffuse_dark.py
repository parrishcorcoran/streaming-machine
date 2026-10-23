"""DIFFUSE family, the dark frame. Error diffusion failed (v1-v6) because the light reading a neuron's lines changes
from line to line like noise, so any carry along the lines only adds scatter. What a halftone DOES get right is the
part of the picture every viewer sees the same way: its mean tone. In this machine that part is the STANDING light --
the light that is on every line for every word (Parrish: "the next token only barely touches everything"). A
developed plate's rounding error read by the standing light is the same offset on that neuron for every word: a
fixed-pattern error, like a camera sensor's dark frame. A camera measures it once and subtracts it. Here it is carried
into the one place every word's pile reads identically -- the neuron's own constant (its bias).

How (whole numbers only): the machine as it is (original plates) is lit by calibration text that is NOT one of the 30
judged prompts (public-domain passages below, cut into 5-word windows, the judge's positions 0-4). At every bank the
developed plate and the original plate are both lit by the SAME light; the difference of their outputs (whole numbers,
the machine's own arithmetic) is summed over all calibration positions; the sum is turned into a mean by one product
and one shift (x 3277 >> 21 = / 640 for 128 windows x 5 positions), and its negative is stored per neuron. The decoder
adds it to the neuron's constant. Nothing about the light is stored -- only one whole number per neuron.
"""
import numpy as np
from diffuse_lib import *
import table_bits as _tb

PASSAGES = [
    "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, "
    "and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing "
    "whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great "
    "battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those "
    "who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this. "
    "But, in a larger sense, we can not dedicate -- we can not consecrate -- we can not hallow -- this ground. The brave "
    "men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract.",
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it "
    "was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of "
    "Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing "
    "before us, we were all going direct to Heaven, we were all going direct the other way.",
    "Call me Ishmael. Some years ago -- never mind how long precisely -- having little or no money in my purse, and "
    "nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the "
    "world. It is a way I have of driving off the spleen and regulating the circulation. Whenever I find myself growing "
    "grim about the mouth; whenever it is a damp, drizzly November in my soul; whenever I find myself involuntarily "
    "pausing before coffin warehouses, and bringing up the rear of every funeral I meet, then I account it high time "
    "to get to sea as soon as I can.",
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a "
    "wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this "
    "truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of "
    "some one or other of their daughters.",
    "There is grandeur in this view of life, with its several powers, having been originally breathed into a few forms "
    "or into one; and that, whilst this planet has gone cycling on according to the fixed law of gravity, from so "
    "simple a beginning endless forms most beautiful and most wonderful have been, and are being, evolved.",
    "When in the Course of human events, it becomes necessary for one people to dissolve the political bands which "
    "have connected them with another, and to assume among the powers of the earth, the separate and equal station to "
    "which the Laws of Nature and of Nature's God entitle them, a decent respect to the opinions of mankind requires "
    "that they should declare the causes which impel them to the separation.",
    "The quality of mercy is not strained; it droppeth as the gentle rain from heaven upon the place beneath. It is "
    "twice blest; it blesseth him that gives and him that takes: 'tis mightiest in the mightiest; it becomes the "
    "throned monarch better than his crown.",
]
NWIN, WLEN = 128, 5
MUL, SH = 3277, 21                                          # x 3277 >> 21 = / 640 (128 windows x 5 positions)

def windows(seed=None):
    """128 windows of 5 consecutive words from the passages (in order); seed: random word ids instead (control)"""
    if seed is not None:
        return [list(map(int, np.random.default_rng(seed).integers(0, 50257, WLEN))) for _ in range(NWIN)]
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    ids = []
    for p in PASSAGES: ids += tok(" " + p).input_ids
    W = [ids[k:k + WLEN] for k in range(0, len(ids) - WLEN + 1, WLEN)][:NWIN]
    assert len(W) == NWIN, len(W)
    judged = {tuple(i) for i in _tb.inputs}
    assert not any(tuple(w) in judged for w in W)
    return W

def _swap(n, w):
    old = _tb.W[n]; _tb.W[n] = (np.asarray(w).astype(np.float64), old[1], old[2]); return old

POS = None                                                  # None: all 5 positions; 1: positions 1-4 only
def _mean_neg(S):
    if POS == 1: return -np.right_shift(S + (1 << 8), 9)   # 128 windows x 4 positions = 512: one shift
    return -np.right_shift(S * MUL + (1 << (SH - 1)), SH)
def _sum(D, O):
    a = 0 if POS is None else POS
    return sum((dk[a:] - ok[a:]).sum(0) for dk, ok in zip(D, O))

def corrections(Wdev, wins, chained=False):
    """per neuron: minus the mean over calibration positions of (developed bank output - original bank output),
    whole numbers. chained=False: both plates lit by the original machine's light. chained=True: the developed
    machine (with the constants found so far) runs its own light forward, bank by bank, and its output is compared
    with the original machine's on the original light -- so each bank's constant also takes up the standing error
    carried to it from every bank before (error carried forward along the chain of plates)."""
    orig = {n: _tb.W[n] for n in NAMES}; corr = {}
    R = [_tb.M["wte"][ids] + _tb.M["wpe"][:len(ids)] for ids in wins]; Rd = [r.copy() for r in R]
    try:
        for l in range(12):
            for bank, pb, nxt in ((f"qkv.{l}", f"proj.{l}", _tb.attention), (f"fc.{l}", f"mproj.{l}", _tb.edge)):
                o = [_tb.gauged_bank(r, bank) for r in R]
                _swap(bank, Wdev[bank]); d = [_tb.gauged_bank(r, bank) for r in (Rd if chained else R)]; _tb.W[bank] = orig[bank]
                corr[bank] = _mean_neg(_sum(d, o))
                xo = [nxt(ok) for ok in o]
                xd = [nxt(dk + corr[bank][None, :]) for dk in d] if chained else xo
                po = [_tb.plain_bank(x, pb) for x in xo]
                _swap(pb, Wdev[pb]); pd = [_tb.plain_bank(x, pb) for x in xd]; _tb.W[pb] = orig[pb]
                corr[pb] = _mean_neg(_sum(pd, po))
                R = [r + x for r, x in zip(R, po)]
                if chained: Rd = [r + x + corr[pb][None, :] for r, x in zip(Rd, pd)]
    finally:
        for n in NAMES: _tb.W[n] = orig[n]
    return corr

# ---------------------------------------------------------------- file with the per-neuron constants
HDR2 = "<BHHbhhII"

def write_file_dark(path, rule, A, develop, wins, chained=False):
    W16 = {}; V = {}; S = {}; ph = [0, 0]
    for n in NAMES:
        q = plate(n); t, off = terms(n, q, rule); s = steps(t, off, A)
        v = develop(n, q, s); V[n] = (v, t, off); W16[n] = shift_back(v, s)
        ph[0] += int(ones(v).sum()); ph[1] += int(naf(v).sum())
    C = corrections(W16, wins, chained=chained)
    with open(path, "wb") as f:
        for idx, n in enumerate(NAMES):
            v, t, off = V[n]; R_, C_ = v.shape; e = colexp(n)
            buf = encode(v); cb = encode(C[n][None, :].astype(np.int32))
            tb = int(t.min()); ob = int(off.min())
            assert (t - tb).max() < 256 and (off - ob).max() < 256 and e.min() > -128 and np.abs(C[n]).max() < (1 << 23)
            f.write(struct.pack(HDR2, idx, R_, C_, A, tb, ob, len(buf), len(cb)))
            f.write(e.astype(np.int8).tobytes()); f.write((t - tb).astype(np.uint8).tobytes())
            f.write((off - ob).astype(np.uint8).tobytes()); f.write(buf); f.write(cb)
    return os.path.getsize(path), W16, C, ph

def read_file_dark(path):
    data = open(path, "rb").read(); p = 0; hs = struct.calcsize(HDR2); out = {}
    for _ in NAMES:
        idx, R_, C_, A, tb, ob, nb, ncb = struct.unpack_from(HDR2, data, p); p += hs
        e = np.frombuffer(data, np.int8, C_, p).astype(np.int64); p += C_
        t = np.frombuffer(data, np.uint8, C_, p).astype(np.int64) + tb; p += C_
        off = np.frombuffer(data, np.uint8, R_, p).astype(np.int64) + ob; p += R_
        v = decode(data[p:p + nb], R_, C_); p += nb
        c = decode(data[p:p + ncb], 1, C_)[0].astype(np.int64); p += ncb
        n = NAMES[idx]; assert np.array_equal(e, colexp(n))
        out[n] = (shift_back(v, steps(t, off, A)), c)
    assert p == len(data)
    return out

def judge_dark(dec):
    """the decoded plates and constants go in the machine (plates replaced, constants added), words out of 30"""
    orig = {n: _tb.W[n] for n in NAMES}
    try:
        for n, (w, c) in dec.items(): _tb.W[n] = (w.astype(np.float64), orig[n][1], orig[n][2] + c)
        return int(_tb.words())
    finally:
        for n in NAMES: _tb.W[n] = orig[n]

def run_dark(label, rule, A, develop, wins, chained=False, path=None, note="", keep=False):
    if _tb._TRUTH is None: assert words() == 30
    path = path or os.path.join(SP, "diffuse_current.bin")
    t0 = time.time()
    size, Wenc, C, ph = write_file_dark(path, rule, A, develop, wins, chained)
    dec = read_file_dark(path)
    ok = all(np.array_equal(Wenc[n], dec[n][0]) and np.array_equal(C[n], dec[n][1]) for n in NAMES)
    wd = judge_dark(dec)
    cbytes = sum(len(encode(C[n][None, :].astype(np.int32))) for n in NAMES)
    rec = {"version": label, "rule": rule, "A": A, "note": note, "file": path if keep else None, "file_bytes": size,
           "total_MB": round(size / 1e6, 3), "bits_per_value": round(8 * size / NVAL, 4), "roundtrip_ok": ok,
           "constants_bytes": cbytes, "stored_photons_per_weight": round(ph[0] / NVAL, 3),
           "stored_single_wave_photons_per_weight": round(ph[1] / NVAL, 3), "words": wd, "secs": round(time.time() - t0, 1)}
    logd(rec); print(json.dumps(rec), flush=True)
    return rec
