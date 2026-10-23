"""THE SAME TOKEN AGAIN AND AGAIN. A word repeated 6 times (positions 0-5), base-2 machine untouched. At every layer's knife
edges (fc piles, 3,072 edges) of each repeat:
  in context, repeat r against repeat r+1: same side? how far did the piles move (size of the change / size of the pile)?
  KV's part:       in context against the same word ALONE at the same position (arrival over only itself)
  position's part: the word alone at position p against alone at p+1
  control:         two different words at the same position, in context
And what the machine says after each repeat."""
import sys, os, json, time, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, HERE)
src = open(f"{HERE}/thinpage/thin_page.py").read(); exec(src[src.index("ARGS = sys.argv"):src.index("def ones(A)")])
WTE, WPE = M["wte"], M["wpe"]
from transformers import GPT2Tokenizer; tok = GPT2Tokenizer.from_pretrained("gpt2")
def fcp(C, l): return np.asarray(C[f"fc.{l}"][2]).astype(np.float64)       # the knife edges' piles at layer l
def side(a, b): return 100.0 * float((np.sign(a) == np.sign(b)).mean())
def move(a, b): return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-9))
R = 6; WORDS = [" the", " cat", " France", "."]
runs = {}
for w in WORDS:
    i = tok.encode(w)[0]; ids = [i] * R; R0 = WTE[ids] + WPE[:R]
    C, top = run(R0, attention); A, _ = run(R0, attention_alone)
    runs[w] = (C, A, top)
    print(f"\n{w!r} x {R}: what the machine says after each repeat:", [tok.decode([int(np.argmax(top[p]))]) for p in range(R)])
    for l in (0, 3, 6, 9, 11):
        cc = [fcp(C, l)[p] for p in range(R)]; aa = [fcp(A, l)[p] for p in range(R)]
        rr = " ".join(f"{side(cc[p + 1], cc[p]):3.0f}%/{move(cc[p + 1], cc[p]):.2f}" for p in range(1, R - 1))
        kv = " ".join(f"{side(cc[p], aa[p]):3.0f}%/{move(cc[p], aa[p]):.2f}" for p in range(1, R))
        po = " ".join(f"{side(aa[p + 1], aa[p]):3.0f}%/{move(aa[p + 1], aa[p]):.2f}" for p in range(1, R - 1))
        print(f"  layer {l:2d}  repeat->next repeat (same side / moved): {rr}")
        print(f"            in context vs alone, same position (KV's part): {kv}")
        print(f"            alone p -> alone p+1 (position's part):        {po}")
print("\ncontrol: two different words at the same position, both repeated in context (same side / moved), position 3:")
for l in (0, 3, 6, 9, 11):
    print(f"  layer {l:2d}  " + "  ".join(f"{a.strip() or a}/{b.strip() or b}: {side(fcp(runs[a][0], l)[3], fcp(runs[b][0], l)[3]):3.0f}%/{move(fcp(runs[a][0], l)[3], fcp(runs[b][0], l)[3]):.2f}" for a, b in [(' the', ' cat'), (' cat', ' France'), (' France', '.')]))
