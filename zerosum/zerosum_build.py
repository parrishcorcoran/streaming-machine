"""THE EXACT CUT -- part 2: every write line with its level taken off, written as new whole-number files.

HIS WORDS (2026-10-06): "It's a zero sum game. If one weight goes up another goes down. A 10 that becomes a 12 is a 2
/ A 10 that becomes a 8 is -2"

Write lines (each one a line of numbers across the stream's lanes):
  GPT-2:  attention-out rows (proj.L, 768 rows) and MLP-out rows (mproj.L, 3072 rows), their biases, the word rows
          (wte, 50,257) and the position rows (wpe, 1,024). The top read keeps its own copy (head.w): untouched.
  Pythia: attention-out and MLP-out rows and their biases, the input word rows (embed.w, one power of two per word).
A line's level = its average over the stream's lanes, in real units. The constant taken off = the level rounded to a
grain 2^G (half away from zero, so no tie tips a balance); after it, every line's level is within half a grain.
  - weight rows: a constant k 2^G real is k 2^(G - we_j) whole units in column j (G >= the plate's largest we_j);
  - biases, GPT-2's word and position rows: whole stream units, k 2^g;
  - Pythia's word rows: k 2^g stream units = k 2^(g + s_t) in the word's own units (s_t its shift onto the stream),
    so the row reaches the stream lowered by exactly k 2^g on every lane.

Three files a machine (zerosum_cut.py found the grains on the 30 refs texts):
  fine   the format's finest grain: weight rows 2^(largest we_j), everything else one stream unit. Takes off all but
         a few % of every line's level. Not invisible by part 1's reading: (i), (ii) and (for GPT-2) (iii) break it.
  cut    the grain that passes (i) and (ii) on these texts. At that grain no weight row and (GPT-2, Pythia-70m and
         -160m) no word row has a nonzero constant: only GPT-2 proj.11.b (-2048), Pythia-70m's two last biases (-64
         each), Pythia-14m's 37,639 word rows and four biases. Pythia-160m: nothing (no file).
  ctl    SHUFFLE CONTROL for fine: the same constants, the same size on every lane, but up on a random half of the
         lanes and down on the other half: zero-sum across the lanes, no level at all.
Also counted: the lit places (1s, and single-wave pixels) of every write line before and after; values past 16 places.

PREDICTIONS (written before the first run):
  - fine: every line's level within half a grain afterwards (asserted). Lit places of the write lines change by under
    2% (session's prediction) -- the constants are 10-100 times smaller than a typical weight; a few weights pass 16
    places where a column's largest weight sits in a row whose constant has the same sign (under 0.01%).
  - cut: the constants exactly as zerosum_cut.py listed them.
  - ctl: line sums equal to the original's to within the rounding of the constants (zero-sum pattern), lit places
    changed about as much as fine.
    python3 zerosum/zerosum_build.py machine=gpt2|14m|70m|160m   -> zerosum/zerosum_<gpt2|pythia-SIZE>[_cut|_ctl].npz
"""
import sys, os, json, time, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/zerosum"
A = dict(a.split("=") for a in sys.argv[1:]); MACH = A.get("machine", "gpt2")
TAG = "gpt2" if MACH == "gpt2" else f"pythia-{MACH}"
t0 = time.time(); LOG = open(f"{OUT}/zerosum_build_{MACH}.log", "w")
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ones(X): return np.bitwise_count(np.abs(np.asarray(X, dtype=np.int64))).astype(np.int64)
def naf(X): a = np.abs(np.asarray(X, dtype=np.int64)); return ones(a ^ (3 * a))
def rnd(S, n):                                        # whole S / whole n > 0, to nearest, half away from zero
    S = np.asarray(S, dtype=np.int64); n = np.asarray(n, dtype=np.int64)
    return np.sign(S) * ((2 * np.abs(S) + n) // (2 * n))
CUT = json.load(open(f"{OUT}/zerosum_cut_{MACH}.json"))["grains"]
if MACH == "gpt2":
    M = np.load(f"{HERE}/base2_gpt2.npz"); D, NL = 768, 12
    TABLES = {"wte": (M["wte"], None), "wpe": (M["wpe"], None)}
else:
    M = np.load(f"{HERE}/foundation0/base2_pythia-{MACH}.npz"); D, NL = int(M["d"]), int(M["L"])
    TABLES = {"embed.w": (M["embed.w"], M["embed.e"])}
BANKS = [f"{b}.{l}" for l in range(NL) for b in ("proj", "mproj")]
SIGN_RNG = np.random.default_rng(20261006)
def halves(n_lines, n_lanes):                         # each line: exactly half its lanes +1, half -1, at random
    s = np.tile(np.r_[np.ones(n_lanes // 2, np.int64), -np.ones(n_lanes - n_lanes // 2, np.int64)], (n_lines, 1))
    return SIGN_RNG.permuted(s, axis=1)

def rows_constant(w, we, G):                          # weight rows: k_i with constant k_i 2^G real; returns k, row sums (units 2^lo)
    lo = int(we.min()); S = (w.astype(np.int64) * (np.int64(1) << (we - lo))[None, :]).sum(1)
    return rnd(S, D * (np.int64(1) << (G - lo))), S, lo
def flat_constant(v, g):                              # stream-unit lines (biases, GPT-2 tables): k with constant k 2^g
    S = np.asarray(v, dtype=np.int64).reshape(-1, D).sum(1); return rnd(S, D * (np.int64(1) << g)), S
def word_constant(w, e, g):                           # Pythia word rows: constant k 2^g stream units = k 2^(g + s) word units
    s = -(e.astype(np.int64) + 16); assert s.min() >= 0
    S = w.astype(np.int64).sum(1); return rnd(S, D * (np.int64(1) << (g + s))), S, s

lit = {}                                              # per variant: [ones before, ones after, naf before, naf after] per kind
def count(kind, variant, before, after):
    L = lit.setdefault(variant, {}).setdefault(kind, [0, 0, 0, 0])
    L[0] += int(ones(before).sum()); L[1] += int(ones(after).sum()); L[2] += int(naf(before).sum()); L[3] += int(naf(after).sum())

for variant in ("fine", "cut", "ctl"):
    if variant == "cut" and MACH == "160m": log("cut: Pythia-160m has no nonzero constant at the grain that passes; no file"); continue
    out, over16, nonzero, lines = {}, 0, 0, 0
    for name in BANKS:
        w, we, b = M[name + ".w"], M[name + ".e"], M[name + ".b"]
        G = int(we.max()) if variant != "cut" else int(CUT[name]["G_cut"])
        k, S, lo = rows_constant(w, we, G)
        step = k[:, None] * (np.int64(1) << (G - we))[None, :]               # whole units, column by column
        if variant == "ctl": step = step * halves(*w.shape)
        w2 = w.astype(np.int64) - step
        if variant != "ctl":                                                  # every row's level now within half a grain
            S2 = (w2 * (np.int64(1) << (we - lo))[None, :]).sum(1)
            assert (2 * np.abs(S2) <= D * (np.int64(1) << (G - lo))).all()
        else:
            S2 = (w2 * (np.int64(1) << (we - lo))[None, :]).sum(1); assert np.array_equal(S2, S)   # the control keeps the level
        gb = 0 if variant != "cut" else int(CUT[name]["bias_grain"])
        kb, Sb = flat_constant(b, gb); bstep = kb * (np.int64(1) << gb) * np.ones(D, np.int64)
        if variant == "ctl": bstep = bstep * halves(1, D)[0]
        b2 = b - bstep
        over16 += int((np.abs(w2) > 32767).sum()); nonzero += int((k != 0).sum()) + int((kb != 0).sum()); lines += len(k) + 1
        out[name + ".w"] = w2.astype(np.int16) if np.abs(w2).max() <= 32767 else w2.astype(np.int32)
        out[name + ".b"] = b2; out["k." + name] = k; out["G." + name] = np.array(G); out["kb." + name] = kb; out["gb." + name] = np.array(gb)
        count("attention-out" if name.startswith("proj") else "MLP-out", variant, w, w2)
        count("biases", variant, b, b2)
    for key, (tab, e) in TABLES.items():
        g = 0 if variant != "cut" else int(CUT[key.split(".")[0]]["grain"])
        if e is None:                                                         # GPT-2: stream units
            k, S = flat_constant(tab, g); step = (k * (np.int64(1) << g))[:, None] * np.ones((1, D), np.int64)
            if variant == "ctl": step = step * halves(*tab.shape)
            t2 = tab - step
            if variant != "ctl": assert (2 * np.abs(t2.sum(1)) <= D * (np.int64(1) << g)).all()
        else:                                                                 # Pythia: word units, own power
            k, S, s = word_constant(tab, e, g); step = (k * (np.int64(1) << (g + s)))[:, None] * np.ones((1, D), np.int64)
            if variant == "ctl": step = step * halves(*tab.shape)
            t2 = tab.astype(np.int64) - step
            if variant != "ctl": assert (2 * np.abs(t2.sum(1)) <= D * (np.int64(1) << (g + s))).all()
            over16 += int((np.abs(t2) > 32767).sum())
        nonzero += int((k != 0).sum()); lines += len(k)
        out[key] = t2.astype(np.int32) if e is None or np.abs(t2).max() > 32767 else t2.astype(np.int16)
        out["k." + key] = k; out["g." + key] = np.array(g)
        count({"wte": "word rows", "wpe": "position rows", "embed.w": "word rows"}[key], variant, tab, t2)
        del t2, step
    if variant == "cut":                                                      # only what changed goes in the file
        keep = {}
        for n in BANKS:
            if (out["k." + n] != 0).any(): keep[n + ".w"] = out[n + ".w"]
            if (out["kb." + n] != 0).any(): keep[n + ".b"] = out[n + ".b"]
        for key in TABLES:
            if (out["k." + key] != 0).any(): keep[key] = out[key]
        keep.update({k: v for k, v in out.items() if k.startswith(("k.", "kb.", "G.", "gb.", "g."))}); out = keep
        log(f"cut: arrays changed: {sorted(k for k in out if not k.startswith(('k.', 'kb.', 'G.', 'gb.', 'g.')))}")
        for n in BANKS:
            if (out["kb." + n] != 0).any(): log(f"     {n}.b constant {int(out['kb.' + n][0]) << int(out['gb.' + n])} stream units")
    dst = f"{OUT}/zerosum_{TAG}{'' if variant == 'fine' else '_' + variant}.npz"
    np.savez_compressed(dst, **out)
    log(f"\n{variant}: {nonzero:,} of {lines:,} write lines with a nonzero constant; values past 16 places {over16:,}  -> {os.path.basename(dst)} "
        f"({os.path.getsize(dst) / 1e6:.0f} MB, {time.time() - t0:.0f}s)")
    tb = [0, 0, 0, 0]
    for kind, L in lit[variant].items():
        for i in range(4): tb[i] += L[i]
        log(f"  lit places {kind:14s}: 1s {L[0]:>13,} -> {L[1]:>13,} ({100 * (L[1] - L[0]) / L[0]:+7.3f}%)   single-wave {L[2]:>13,} -> {L[3]:>13,} ({100 * (L[3] - L[2]) / L[2]:+7.3f}%)")
    log(f"  lit places {'all writes':14s}: 1s {tb[0]:>13,} -> {tb[1]:>13,} ({100 * (tb[1] - tb[0]) / tb[0]:+7.3f}%)   single-wave {tb[2]:>13,} -> {tb[3]:>13,} ({100 * (tb[3] - tb[2]) / tb[2]:+7.3f}%)")
    del out
json.dump(lit, open(f"{OUT}/zerosum_build_{MACH}.json", "w"), indent=1)
log(f"\ndone ({time.time() - t0:.0f}s)")
