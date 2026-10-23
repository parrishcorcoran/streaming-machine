"""THE EXACT CUT -- part 4: the way around the road block. Pour the level off BEFORE the 16-place cut.

HIS WORDS (2026-10-06): "It's a zero sum game. If one weight goes up another goes down. A 10 that becomes a 12 is a 2
/ A 10 that becomes a 8 is -2"

Parts 1-3 found where the machine's rounding breaks the zero-sum reading (the session's reading, not his):
  (ii) every read cuts the stream to 16 places at a power set by the largest lane BEFORE the water level pours the
       level off, so a level that is not a whole number of the cut's grain, or that moves the top 1, changes the cut;
  (iii) GPT-2's plates pour the level off with a rounding residue in every column;
  (i)  every write rounds each lane to the stream's grain with the row's level inside it.
THE WAY AROUND (ii) and (iii), one change to the water level and nothing else: it forms the stream's distance from
its own level in whole numbers first, d x R - (sum of R over the lanes) (no divide: d times the distance), and cuts
THAT to 16 places; the rest of the water level is the machine's own (GPT-2: its sums, its log, the folded plate;
Pythia: its centring, its log, gain and offset); eps is carried d^2 larger to match (EPS40 x d^2, a whole number).
A whole-number level on every lane of the stream then cannot reach the cut at all: d(R + L) - sum(R + L) = dR - sum R.

Runs on the 30 refs texts (GPT-2 648 positions, Pythia 622), every read's output, every write's output, the stream
without its level, and the top scores digested per position:
  W0  the water level poured before the cut, the original files                     (this machine's reference)
  W1  the same, with the lines that reach the stream in whole stream units taken level-free at ONE stream unit:
      the biases, the word rows, GPT-2's position rows (from the fine file). Weight plates untouched.
  W2  the same, with the whole fine file (weight rows too)
  (every write's output and the stream are digested without their level, x - x[lane 0]: a level is what is taken off)
  P1  the plain machine (cut before pour) with W1's lines                            (against the plain machine P0)
  W0 against P0: how far the poured-first machine is from the plain one (top words, top 5), and both against the refs.

PREDICTIONS (written before the first run):
  W1 = W0 digit for digit at every position in all four machines: no exceptions (the cut never sees the level).
  W2 differs from W0 from the first read after the first write on, at nearly every position: (i) is untouched.
  P1 differs from P0: GPT-2 from qkv.0's read on (its cut grain there is 2-16 stream units); Pythia from layer 1's
      read on (layer 0's cut grain is 1, the constants pass there).
  W0 against P0: top word the same at 99% or more of positions; the poured-first machine's words agree with the refs
      as often as the plain machine's, within 1-2 words.
    python3 zerosum/zerosum_way.py machine=gpt2|14m|70m|160m
"""
import sys, os, json, time, resource, hashlib, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/zerosum"
A = dict(a.split("=") for a in sys.argv[1:]); MACH = A.get("machine", "gpt2")
t0 = time.time(); LOG = open(f"{OUT}/zerosum_way_{MACH}.log", "w")
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def hrows(X): X = np.ascontiguousarray(X); return [hashlib.blake2b(X[i].tobytes(), digest_size=8).digest() for i in range(len(X))]
ZF = np.load(f"{OUT}/zerosum_{'gpt2' if MACH == 'gpt2' else 'pythia-' + MACH}.npz")

if MACH == "gpt2":
    src = open(f"{HERE}/base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]
    src = src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np").replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
    banks = '[f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]'
    assert banks in src; src = src.replace(banks, '[]')
    cwd = os.getcwd(); os.chdir(HERE); argv = sys.argv; sys.argv = ["x"]; B = {}; exec(src, B); sys.argv = argv; os.chdir(cwd)
    M, ES, D = B["M"], B["ES"], 768
    OVER = {}                                          # name -> (w, e, b) put in place of the file's
    class Plates(dict):
        def __missing__(self, k):
            if k in OVER: return OVER[k]
            return (M[k + ".w"].astype(np.float64), M[k + ".e"], M[k + ".b"])
    B["W"] = Plates(); W = B["W"]
    to16, log2fix, gauge_plain = B["to16"], B["log2fix"], B["gauge_log"]
    EPS40, LOG768 = B["EPS40"], B["LOG768"]
    def gauge_poured(R):                               # the level poured off first, in whole numbers; then base2's own steps
        Rc = D * R - R.sum(1)[:, None]
        x16, ex = to16(Rc, -ES)
        S1, S2 = x16.sum(1), (x16 * x16).sum(1)
        V = 768 * S2 - S1 * S1
        V = V + np.array([(EPS40 * D * D) >> (40 + 2 * int(e)) if 40 + 2 * int(e) >= 0 else (EPS40 * D * D) << -(40 + 2 * int(e)) for e in ex], dtype=object).astype(np.int64)
        return x16, ex, np.right_shift(log2fix(V), 1) - LOG768 + ex * 65536
    WTE0, WPE0 = M["wte"], M["wpe"]
    refs = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in refs]; NP = [len(r["ids"]) for r in refs]
    def run(poured, lines):                            # lines: "none" | "stream" (biases, word and position rows) | "all"
        B["gauge_log"] = gauge_poured if poured else gauge_plain
        OVER.clear()
        for l in range(12):
            for b in ("proj", "mproj"):
                n = f"{b}.{l}"
                if lines == "all": OVER[n] = (ZF[n + ".w"].astype(np.float64), M[n + ".e"], ZF[n + ".b"])
                elif lines == "stream": OVER[n] = (M[n + ".w"].astype(np.float64), M[n + ".e"], ZF[n + ".b"])
        wte = ZF["wte"].astype(np.int64) if lines != "none" else WTE0; wpe = ZF["wpe"].astype(np.int64) if lines != "none" else WPE0
        res = []
        for ids in TEXTS:
            rec = []; P = len(ids); R = wte[ids] + wpe[:P]
            for l in range(12):
                Q = B["gauged_bank"](R, f"qkv.{l}"); rec.append((f"read qkv.{l}", hrows(Q)))
                O = B["plain_bank"](B["attention"](Q), f"proj.{l}"); rec.append((f"write proj.{l}", hrows(O - O[:, :1]))); R = R + O
                H = B["gauged_bank"](R, f"fc.{l}"); rec.append((f"read fc.{l}", hrows(H)))
                O = B["plain_bank"](B["edge"](H), f"mproj.{l}"); rec.append((f"write mproj.{l}", hrows(O - O[:, :1]))); R = R + O
                rec.append((f"stream {l}", hrows(R - R[:, :1])))
            w, we, b = W["head"]; x16, ex, LS = B["gauge_log"](R)
            S = B["power"](B["product"](x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b
            rec.append(("top", hrows(S))); res.append(dict(rec=rec, win=S.argmax(1)))
        return res
    def ref_words(res): return sum(int(x["win"][n - 1 + i] == r["ref"][i]) for x, r, n in zip(res, refs, NP) for i in range(16))
else:
    path = f"{HERE}/foundation0/pythia_base2.py"; src = open(path).read(); src = src[:src.index('if __name__ == "__main__":')]
    argv = sys.argv; sys.argv = ["x", f"size={MACH}"]; B = {"__file__": path, "__name__": "way"}; exec(src, B); sys.argv = argv
    PL, NORM, ES, L, D = B["PL"], B["NORM"], B["ES"], B["L"], B["d"]
    PL0 = dict(PL); EMB0 = B["EMB_F"]; gauge_plain = B["gauge"]; to16, log2fix, power = B["to16"], B["log2fix"], B["power"]
    EPS40 = int(B["EPS40"])
    def gauge_poured(R, name):                         # the level poured off first, in whole numbers; then the machine's own gauge
        g, ge, b = NORM[name]
        Rc = D * R - R.sum(1)[:, None]
        x16, ex = to16(Rc)
        S1, S2 = x16.sum(1), (x16 * x16).sum(1)
        c = D * x16 - S1[:, None]
        V = D * S2 - S1 * S1 + np.array([(EPS40 * D * D) >> (40 + 2 * int(e)) for e in ex], dtype=object).astype(np.int64)
        LS = np.array([log2fix(v) >> 1 for v in V])
        return power(c * g[None, :], (ge + ES) * 65536 - LS[:, None]) + b
    refs = json.load(open(f"{HERE}/foundation0/refs_pythia-{MACH}.json")); TEXTS = [r["ids"] + r["ref"][:15] for r in refs]; NP = [len(r["ids"]) for r in refs]
    def run(poured, lines):
        gauge = gauge_poured if poured else gauge_plain
        PL.clear(); PL.update(PL0)
        for l in range(L):
            for b in ("proj", "mproj"):
                n = f"{b}.{l}"; w, e, bb, wm = PL0[n]
                if lines == "all": w = ZF[n + ".w"].astype(np.float64); PL[n] = (w, e, ZF[n + ".b"], float(np.abs(w).max()))
                elif lines == "stream": PL[n] = (w, e, ZF[n + ".b"], wm)
        emb = ZF["embed.w"] if lines != "none" else EMB0
        res = []
        for ids in TEXTS:
            rec = []; ids = np.asarray(ids); R = B["rshift"](emb[ids].astype(np.int64), -(B["EMB_E"][ids][:, None] + ES))
            for l in range(L):
                g1 = gauge(R, f"ln1.{l}"); g2 = gauge(R, f"ln2.{l}"); rec += [(f"read ln1.{l}", hrows(g1)), (f"read ln2.{l}", hrows(g2))]
                At = B["plain"](B["attention"](B["plain"](g1, f"qkv.{l}")), f"proj.{l}"); Mo = B["plain"](B["edge"](B["plain"](g2, f"fc.{l}")), f"mproj.{l}")
                rec += [(f"write proj.{l}", hrows(At - At[:, :1])), (f"write mproj.{l}", hrows(Mo - Mo[:, :1]))]; R = R + At + Mo
                rec.append((f"stream {l}", hrows(R - R[:, :1])))
            y16, ey = to16(gauge(R, "lnf")); w, e, _, wmax = PL["head"]
            S = B["rshift"](B["product"](y16, w, wmax), -(e[None, :] + ey[:, None] + ES))
            rec.append(("top", hrows(S))); res.append(dict(rec=rec, win=S.argmax(1)))
        PL.clear(); PL.update(PL0)
        return res
    def ref_words(res): return sum(int(x["win"][n - 1 + i] == r["ref"][i]) for x, r, n in zip(res, refs, NP) for i in range(16))

def compare(a, b, label):
    npos = sum(len(x["win"]) for x in a); same = 0; first = {}; writes_only = 0
    for x, y in zip(a, b):
        P = len(x["win"]); bad = np.zeros(P, bool); f = None; badr = np.zeros(P, bool)
        for (k, hx), (k2, hy) in zip(x["rec"], y["rec"]):
            assert k == k2
            d = np.array([p != q for p, q in zip(hx, hy)])
            if d.any() and f is None: f = k
            bad |= d
            if not k.startswith("write"): badr |= d
        same += int((~bad).sum()); writes_only += int((bad & ~badr).sum())
        if f: first[f] = first.get(f, 0) + 1
    win = sum(int((x["win"] == y["win"]).sum()) for x, y in zip(a, b))
    log(f"\n{label}")
    log(f"  positions identical everywhere (every read, write, stream without its level, top score): {same}/{npos}"
        + (f"; differing only in a write's own output: {writes_only}" if writes_only else ""))
    log(f"  top word the same {win}/{npos}" + (f"; first difference per text: " + ", ".join(f"{k} x{v}" for k, v in sorted(first.items(), key=lambda z: -z[1])) if first else ""))
    return dict(same=same, positions=npos, win_same=win, first=first)

out = {}
P0 = run(False, "none"); log(f"P0 plain ({time.time() - t0:.0f}s, {rss()} MB); words = refs {ref_words(P0)}/480")
W0 = run(True, "none"); log(f"W0 poured first ({time.time() - t0:.0f}s, {rss()} MB); words = refs {ref_words(W0)}/480")
out["W0_vs_P0"] = compare(P0, W0, "W0 (level poured before the cut) AGAINST P0 (the plain machine), the original files")
W1 = run(True, "stream"); out["W1_vs_W0"] = compare(W0, W1, "W1 AGAINST W0: biases, word rows" + (", position rows" if MACH == "gpt2" else "") + " level-free at one stream unit, poured before the cut")
del W1
P1 = run(False, "stream"); out["P1_vs_P0"] = compare(P0, P1, "P1 AGAINST P0: the same lines level-free, the plain machine (cut before pour)")
del P1
W2 = run(True, "all"); out["W2_vs_W0"] = compare(W0, W2, "W2 AGAINST W0: the whole fine file (weight rows too), poured before the cut")
out["refs"] = dict(P0=ref_words(P0), W0=ref_words(W0))
json.dump(out, open(f"{OUT}/zerosum_way_{MACH}.json", "w"), indent=1, default=int)
log(f"\ndone ({time.time() - t0:.0f}s, peak {rss()} MB)")
