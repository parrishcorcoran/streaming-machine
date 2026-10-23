"""INDEPENDENT CHECKER for zerosum_cut.py / zerosum_build.py / zerosum_run.py (shares no code with them).

HIS WORDS (2026-10-06): "It's a zero sum game. If one weight goes up another goes down. A 10 that becomes a 12 is a 2
/ A 10 that becomes a 8 is -2"

  1. THE FILES. Every write line's constant rebuilt in Python whole numbers (fractions, one line at a time) from the
     original npz and the grains in zerosum_cut_<m>.json; every changed value checked against the zerosum files:
     fine and cut: new = old - k x 2^(G - we_j) on every lane (k the line's level over the lanes, rounded half away
     from zero); ctl: |new - old| the same on every lane, exactly half the lanes up and half down.
  2. GPT-2's read plates' column sums, summed a second way (Python whole numbers over each column of every 7th plate
     and of the top, the rest by einsum).
  3. THE MACHINES' OWN FORWARDS, loaded on their own:
     - Pythia: pythia_base2.py's forward(ids + ref[:15], last=16) with the zerosum arrays put in its PL / EMB_F; the
       head piles hashed as pythia_base2.py hashes them, against foundation0/words_pythia-<m>.json.
     - GPT-2: base2.py's own source run with its M replaced by the zerosum arrays (a separate exec, not judge5);
       base2.py's forward(prefix) for every prefix of the 30 refs texts (648 words), against the plain base2.py.
PREDICTIONS (from the runs being checked): the files exactly as built; column sums as zerosum_cut.py found;
  Pythia-70m cut 30/30 head piles identical, Pythia-14m cut 27/30, every fine and ctl file 0/30; GPT-2: words the
  same at as many of the 648 positions as zerosum_run.py's race winners.
    python3 zerosum/zerosum_cutcheck.py part=files|colsum|pythia|gpt2 machine=...
"""
import sys, os, json, time, hashlib, warnings, numpy as np
from fractions import Fraction
warnings.filterwarnings("ignore", message=".*encountered in matmul")
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/zerosum"
A = dict(a.split("=") for a in sys.argv[1:]); PART, MACH = A.get("part", "files"), A.get("machine", "gpt2")
t0 = time.time(); LOG = open(f"{OUT}/zerosum_cutcheck_{PART}_{MACH}.log", "w")
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def round_half_away(fr):                               # Fraction -> whole number, half away from zero
    n = abs(fr); q = int(n); r = n - q
    if r >= Fraction(1, 2): q += 1
    return q if fr >= 0 else -q

if PART == "files":
    src = np.load(f"{HERE}/base2_gpt2.npz") if MACH == "gpt2" else np.load(f"{HERE}/foundation0/base2_pythia-{MACH}.npz")
    grains = json.load(open(f"{OUT}/zerosum_cut_{MACH}.json"))["grains"]
    D = 768 if MACH == "gpt2" else int(src["d"]); NL = 12 if MACH == "gpt2" else int(src["L"])
    tag = "gpt2" if MACH == "gpt2" else f"pythia-{MACH}"
    for var in ("fine", "cut", "ctl"):
        f = f"{OUT}/zerosum_{tag}{'' if var == 'fine' else '_' + var}.npz"
        if not os.path.exists(f): log(f"{var}: no file"); continue
        Z = np.load(f); bad = lines = nz = 0
        for l in range(NL):
            for b in ("proj", "mproj"):
                n = f"{b}.{l}"; w, we, bias = src[n + ".w"].astype(np.int64), src[n + ".e"].astype(np.int64), src[n + ".b"].astype(np.int64)
                G = int(we.max()) if var != "cut" else int(grains[n]["G_cut"]); gb = 0 if var != "cut" else int(grains[n]["bias_grain"])
                w2 = Z[n + ".w"].astype(np.int64) if n + ".w" in Z.files else w; b2 = Z[n + ".b"].astype(np.int64) if n + ".b" in Z.files else bias
                lo = int(we.min()); sh = [int(e) - lo for e in we]; wl = w.tolist()
                for i in range(w.shape[0]):            # one line at a time: Python whole numbers, then a fraction
                    lev = Fraction(sum(x << s for x, s in zip(wl[i], sh)), D) * Fraction(2) ** lo
                    k = round_half_away(lev / Fraction(2) ** G); lines += 1; nz += k != 0
                    diff = w[i] - w2[i]; unit = np.array([1 << (G - int(e)) for e in we], dtype=np.int64)
                    if var == "ctl":
                        ok = (np.abs(diff) == abs(k) * unit).all() and (k == 0 or ((diff * np.sign(k) > 0).sum() == D // 2))
                    else: ok = np.array_equal(diff, k * unit)
                    bad += not ok
                levb = Fraction(int(bias.sum()), D); kb = round_half_away(levb / Fraction(2) ** gb); lines += 1; nz += kb != 0
                d = bias - b2
                ok = (np.abs(d) == abs(kb) << gb).all() and (kb == 0 or (d * np.sign(kb) > 0).sum() == D // 2) if var == "ctl" else (d == kb << gb).all()
                bad += not ok
        tables = {"wte": None, "wpe": None} if MACH == "gpt2" else {"embed.w": src["embed.e"]}
        for key, e in tables.items():
            tab = src[key].astype(np.int64); t2 = Z[key].astype(np.int64) if key in Z.files else tab
            g = 0 if var != "cut" else int(grains[key.split(".")[0]]["grain"])
            sums = tab.sum(1)
            for t in range(len(tab)):
                s = 0 if e is None else -(int(e[t]) + 16)
                lev = Fraction(int(sums[t]), D) / Fraction(2) ** s          # stream units
                k = round_half_away(lev / Fraction(2) ** g); lines += 1; nz += k != 0
                d = tab[t] - t2[t]; u = k << (g + s) if k >= 0 else -((-k) << (g + s))
                ok = ((np.abs(d) == abs(u)).all() and (u == 0 or (d * np.sign(u) > 0).sum() == D // 2)) if var == "ctl" else (d == u).all()
                bad += not ok
        log(f"{var}: {lines:,} write lines rebuilt in fractions, {nz:,} with a nonzero constant; lines that differ from the file: {bad}  ({time.time() - t0:.0f}s)")

elif PART == "colsum":
    M = np.load(f"{HERE}/base2_gpt2.npz"); names = [f"{b}.{l}" for l in range(12) for b in ("qkv", "fc")] + ["head"]
    ref = json.load(open(f"{OUT}/zerosum_cut_gpt2.json"))["colsum"]; agree = 0
    for i, n in enumerate(names):
        w = M[n + ".w"]
        if i % 7 == 0 or n == "head": cs = np.array([sum(int(x) for x in w[:, j]) for j in range(w.shape[1])], dtype=np.int64)
        else: cs = np.einsum("ij->j", w.astype(np.int64))
        z, rms = int((cs == 0).sum()), float(np.sqrt((cs.astype(float) ** 2).mean()))
        agree += (z == ref[n]["zero"] and abs(rms - ref[n]["rms"]) < 1e-9)
        log(f"  {n:7s} zero {z:5d}/{len(cs)} rms {rms:.3f}   (zerosum_cut.py: {ref[n]['zero']}, {ref[n]['rms']:.3f})")
    log(f"column sums agree with zerosum_cut.py at {agree}/{len(names)} plates ({time.time() - t0:.0f}s)")

elif PART == "pythia":
    path = f"{HERE}/foundation0/pythia_base2.py"; src = open(path).read(); src = src[:src.index('if __name__ == "__main__":')]
    argv = sys.argv; sys.argv = ["x", f"size={MACH}"]; P = {"__file__": path, "__name__": "chk"}; exec(src, P); sys.argv = argv
    refs = json.load(open(f"{HERE}/foundation0/refs_pythia-{MACH}.json"))
    saved = [x["piles_sha256"] for x in json.load(open(f"{HERE}/foundation0/words_pythia-{MACH}.json"))["texts_detail"]]
    PL0, EMB0 = dict(P["PL"]), P["EMB_F"]
    def run():
        piles, words = [], []
        for r in refs:
            want = r["ref"][:16]; got, lg = P["forward"](r["ids"] + want[:-1], last=16)
            piles.append(hashlib.sha256(lg.tobytes()).hexdigest()); words.append(got)
        return piles, words
    p0, w0 = run(); log(f"plain: head piles = words_pythia-{MACH}.json at {sum(a == b for a, b in zip(p0, saved))}/30")
    for var in ("cut", "fine", "ctl"):
        f = f"{OUT}/zerosum_pythia-{MACH}{'' if var == 'fine' else '_' + var}.npz"
        if not os.path.exists(f): continue
        Z = np.load(f); P["PL"].clear(); P["PL"].update(PL0)
        for n in list(PL0):
            if n + ".w" in Z.files or n + ".b" in Z.files:
                w, e, b, wm = PL0[n]
                w = Z[n + ".w"].astype(np.float64) if n + ".w" in Z.files else w; b = Z[n + ".b"] if n + ".b" in Z.files else b
                P["PL"][n] = (w, e, b, float(np.abs(w).max()))
        P["EMB_F"] = Z["embed.w"] if "embed.w" in Z.files else EMB0
        p1, w1 = run()
        log(f"{var}: head piles the same as the plain machine's at {sum(a == b for a, b in zip(p0, p1))}/30 texts "
            f"(differ: {[i for i, (a, b) in enumerate(zip(p0, p1)) if a != b][:8]}); words the same {sum(int(x == y) for a, b in zip(w0, w1) for x, y in zip(a, b))}/480  ({time.time() - t0:.0f}s)")
    P["PL"].clear(); P["PL"].update(PL0); P["EMB_F"] = EMB0

elif PART == "gpt2":
    VAR = A.get("variant", "cut")
    src = open(f"{HERE}/base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]
    src = src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np").replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
    assert 'M = np.load("base2_gpt2.npz")' in src
    def machine(over):                                 # base2.py's own source, its M with the zerosum arrays put in
        Mx = dict(np.load(f"{HERE}/base2_gpt2.npz")); Mx.update(over)
        ns = {"__name__": "chk"}; s2 = src.replace('M = np.load("base2_gpt2.npz")', "M = __M")
        ns["__M"] = Mx; argv = sys.argv; sys.argv = ["x"]; exec(s2, ns); sys.argv = argv; return ns
    refs = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in refs]
    def words(ns):
        return [[ns["forward"](ids[:p + 1]) for p in range(len(ids))] for ids in TEXTS]
    ns = machine({}); w0 = words(ns); del ns
    agree_ref = sum(int(w[len(r["ids"]) - 1 + i] == r["ref"][i]) for w, r in zip(w0, refs) for i in range(16))
    log(f"plain base2.py, every prefix of the 30 refs texts: {sum(len(w) for w in w0)} words; its words = fp32 GPT-2's refs at {agree_ref}/480 ({time.time() - t0:.0f}s)")
    Z = np.load(f"{OUT}/zerosum_gpt2{'' if VAR == 'fine' else '_' + VAR}.npz")
    over = {k: (Z[k].astype(np.int64) if k in ("wte", "wpe") else Z[k]) for k in Z.files if not k.startswith(("k.", "kb.", "G.", "gb.", "g."))}
    ns = machine(over); w1 = words(ns)
    same = sum(int(a == b) for x, y in zip(w0, w1) for a, b in zip(x, y)); n = sum(len(x) for x in w0)
    diff = [(t, p) for t, (x, y) in enumerate(zip(w0, w1)) for p, (a, b) in enumerate(zip(x, y)) if a != b]
    log(f"{VAR}: base2.py's own forward with the zerosum arrays: the word the same as plain at {same}/{n} positions; differ at {diff[:10]}  ({time.time() - t0:.0f}s)")
log(f"done ({time.time() - t0:.0f}s)")
