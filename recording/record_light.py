"""PHASE A, part 1: RECORD THE LIGHT AT EVERY PLATE THAT READS THE STREAM, and the two foundation streams.

The machine is base2.py, untouched (its own lines are run; only the stream is kept where each plate reads it).
Plates that read the stream: qkv.l (the stream entering layer l), fc.l (the stream after layer l's arrival is added),
and the top read (the stream after layer 11). Every position of the 30 refs texts (ids + ref).

  F  the weights alone: position 0 with word row 0 and position row 0 (no light) run through the machine; the stream
     at every plate that reads it.
  S  the first-place stone: the stream at every plate at position 0 for one word: ' The' (383), and ' the' (262),
     ' In' (554), '.' (13), ' Once' (4874) to see how much the stone depends on the word.

CHECKS: this loop's word at the last position == base2.forward on every text; the machine's word at every judged
position against the refs words; each text's position-0 stream == its first word run alone (the stone in context is
the stone alone: position 0 only ever reads itself).
    python3 recording/record_light.py   -> recording/light.npz
"""
import sys, os, json, time, resource, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; os.chdir(HERE)
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]
src = src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np")          # never used by
src = src.replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")                  # the machine
exec(src)                                                    # the machine, untouched
PLATES = [f"{b}.{l}" for l in range(12) for b in ("qkv", "fc")] + ["head"]

def run(R0):                                                 # base2.forward's own lines; the stream kept at each plate
    R = R0; seen = {}
    for l in range(12):
        seen[f"qkv.{l}"] = R.copy()
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        seen[f"fc.{l}"] = R.copy()
        R = R + plain_bank(edge(gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
    seen["head"] = R.copy()
    w, we, b = W["head"]; x16, ex, LS = gauge_log(R)
    top = power(product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b
    return seen, top

vocab = json.load(open(os.path.expanduser("~/.cache/huggingface/hub/models--gpt2/snapshots/"
                                          "607a30d783dfa663caf39e06633721c8d4cfcd7e/vocab.json")))
WORDS = {" The": 383, " the": 262, " In": 554, ".": 13, " Once": 4874}
for s, i in WORDS.items(): assert vocab[s.replace(" ", "Ġ")] == i, (s, i)

t0 = time.time(); out = {}
F, _ = run(np.zeros((1, 768), np.int64))                     # no light at all
for k in PLATES: out[f"F|{k}"] = F[k][0]
for s, i in WORDS.items():
    S, _ = run(M["wte"][[i]] + M["wpe"][:1])
    for k in PLATES: out[f"S{s}|{k}"] = S[k][0]
print(f"foundation streams done ({time.time() - t0:.0f}s)", flush=True)

refs = json.load(open("refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in refs]
NPOS = sum(len(s) for s in TEXTS)
X = {k: np.zeros((NPOS, 768), np.int64) for k in PLATES}
POS, TXT = np.zeros(NPOS, np.int64), np.zeros(NPOS, np.int64)
CHK = dict(forward_same=0, words_same=0, words_n=0, pos0_alone_same=0)
row = 0
for t, seq in enumerate(TEXTS):
    P = len(seq); seen, top = run(M["wte"][seq] + M["wpe"][:P])
    for k in PLATES: X[k][row:row + P] = seen[k]
    CHK["forward_same"] += race(top[-1]) == forward(seq)
    n0 = len(refs[t]["ids"])
    for p in range(n0 - 1, P - 1): CHK["words_same"] += race(top[p]) == seq[p + 1]; CHK["words_n"] += 1
    alone, _ = run(M["wte"][seq[:1]] + M["wpe"][:1])
    CHK["pos0_alone_same"] += all(np.array_equal(alone[k][0], seen[k][0]) for k in PLATES)
    POS[row:row + P] = np.arange(P); TXT[row:row + P] = t; row += P
    if t % 5 == 4: print(f"  {t + 1} texts ({time.time() - t0:.0f}s, {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB) {CHK}", flush=True)
np.savez(f"{HERE}/recording/light.npz", plates=np.array(PLATES), pos=POS, txt=TXT, words=np.array(list(WORDS)),
         **{f"X|{k}": X[k] for k in PLATES}, **out)
print(f"checks: this loop's last word == base2.forward: {CHK['forward_same']}/30 texts;  the machine's word == refs word:"
      f" {CHK['words_same']}/{CHK['words_n']} judged positions;  position 0 in context == the first word alone (every plate):"
      f" {CHK['pos0_alone_same']}/30")
print(f"F at qkv.0 (the stream entering layer 0 with no light) is all zero: {not out['F|qkv.0'].any()}")
print(f"saved recording/light.npz: {NPOS} positions x {len(PLATES)} plates ({time.time() - t0:.0f}s, "
      f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB)")
