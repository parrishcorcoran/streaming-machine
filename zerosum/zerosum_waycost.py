"""THE EXACT CUT -- part 5: what the way around costs, and what the level's own work is, in photons a token.

HIS WORDS (2026-10-06): "It's a zero sum game. If one weight goes up another goes down. A 10 that becomes a 12 is a 2
/ A 10 that becomes a 8 is -2"

Counted in combined/machine.py's unit (an add counts the photons of what it adds) on the plain machine's own stream,
every read of every position of the 30 refs texts (GPT-2: 25 reads x 648 positions):
  the level's work in the water level as it is: S1's adds (the 1s of each lane of the cut light), S1's square
      (ones(S1)^2) and its subtract (ones(S1^2))  -- what "the water level no longer finds the level" would remove;
  the pour before the cut (zerosum_way.py): the stream's sum over the lanes (the 1s of every lane of R), d x R as
      512 R + 256 R (one add of 256 R a lane: the 1s of R), and the sum taken off every lane (d x ones(sum R)).
PREDICTION (written before the run): the level's work 0.0027% of plain (zerosum_cut.py measured 98,773 a token);
  the pour before the cut about 25 x 768 x 2 x 14 photons + 25 x 768 x 20: about 1 million a token, about 0.03% of
  plain 3,666,596,259 -- ten times what it lets the water level skip.
    python3 zerosum/zerosum_waycost.py
"""
import sys, os, json, time, numpy as np, warnings
warnings.filterwarnings("ignore", message=".*encountered in matmul")
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/zerosum"; PLAIN = 3666596259
t0 = time.time(); LOG = open(f"{OUT}/zerosum_waycost.log", "w")
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ones(X): return np.bitwise_count(np.abs(np.asarray(X, dtype=np.int64))).astype(np.int64)
src = open(f"{HERE}/base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]
src = src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np").replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
banks = '[f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]'
assert banks in src; src = src.replace(banks, '[]')
cwd = os.getcwd(); os.chdir(HERE); argv = sys.argv; sys.argv = ["x"]; B = {}; exec(src, B); sys.argv = argv; os.chdir(cwd)
M = B["M"]
class Plates(dict):
    def __missing__(self, k): return (M[k + ".w"].astype(np.float64), M[k + ".e"], M[k + ".b"])
B["W"] = Plates()
C = {"level": 0, "pour": 0}; _gl = B["gauge_log"]
def gl(R):
    x16, ex, LS = _gl(R); S1 = x16.sum(1)
    C["level"] += int(ones(x16).sum() + (ones(S1) ** 2).sum() + ones(S1 * S1).sum())
    C["pour"] += int(2 * ones(R).sum() + (768 * ones(R.sum(1))).sum())
    return x16, ex, LS
B["gauge_log"] = gl
refs = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in refs]; WTE, WPE = M["wte"], M["wpe"]; n = 0
for ids in TEXTS:
    P = len(ids); n += P; R = WTE[ids] + WPE[:P]
    for l in range(12):
        R = R + B["plain_bank"](B["attention"](B["gauged_bank"](R, f"qkv.{l}")), f"proj.{l}")
        R = R + B["plain_bank"](B["edge"](B["gauged_bank"](R, f"fc.{l}")), f"mproj.{l}")
    B["gauge_log"](R)
lv, po = C["level"] / n, C["pour"] / n
log(f"{n} positions, 25 reads each")
log(f"  the level's work in the water level as it is: {lv:,.0f} photons a token = {100 * lv / PLAIN:.5f}% of plain")
log(f"  the pour before the cut (the way around):     {po:,.0f} photons a token = {100 * po / PLAIN:.5f}% of plain ({po / lv:.1f} x the level's work)")
json.dump(dict(positions=n, level_per_token=lv, pour_per_token=po, plain=PLAIN), open(f"{OUT}/zerosum_waycost.json", "w"), indent=1)
log(f"done ({time.time() - t0:.0f}s)")
