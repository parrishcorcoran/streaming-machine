"""SUPERPOSING AS AN ADD OF ANGLES. After each water level every position's light has the same size, so it lies on one
sphere, and any change between two lights there is a turn by an angle. Measured on the base-2 machine's own light at
each layer's first plate (qkv.l input, after the water level):
  - the angle each new token turns the pattern (same word repeated; and real text, token to token)
  - do the turns ADD? angle(1->3) against angle(1->2) + angle(2->3) (same plane: they add exactly; at right angles: they
    add like the sides of a right triangle, sqrt(a^2 + b^2))"""
import sys, os, json, time, math, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, HERE)
src = open(f"{HERE}/thinpage/thin_page.py").read(); exec(src[src.index("ARGS = sys.argv"):src.index("def ones(A)")])
WTE, WPE = M["wte"], M["wpe"]
from transformers import GPT2Tokenizer; tok = GPT2Tokenizer.from_pretrained("gpt2")
def light(C, l): x = np.asarray(C[f"qkv.{l}"][0]).astype(np.float64); return x - x.mean(1, keepdims=True)
def ang(a, b): return math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))))))
def report(name, X):
    P = X.shape[0]; steps = [ang(X[p], X[p + 1]) for p in range(1, P - 1)]
    add = []                                                    # does the turn from p to p+2 equal the two steps added?
    for p in range(1, P - 2):
        a, b, c = ang(X[p], X[p + 1]), ang(X[p + 1], X[p + 2]), ang(X[p], X[p + 2])
        add.append((c, a + b, math.hypot(a, b)))
    s = " ".join(f"{v:5.1f}" for v in steps)
    ad = "  ".join(f"{c:4.1f} vs {ab:4.1f}|{h:4.1f}" for c, ab, h in add[:4])
    return f"{name:34s} each token turns the pattern (deg): {s}\n{'':34s} two turns: actual vs added|right-angle: {ad}"
L_SHOW = (0, 3, 6, 9, 11)
for w in [" the", " cat"]:
    i = tok.encode(w)[0]; ids = [i] * 6; C, _ = run(WTE[ids] + WPE[:6], attention)
    print(f"\n{w!r} repeated:")
    for l in L_SHOW: print(f" layer {l:2d}  " + report("", light(C, l)).lstrip())
refs = json.load(open(f"{HERE}/refs_many.json")); ids = (refs[0]["ids"] + refs[0]["ref"])[:12]
C, _ = run(WTE[ids] + WPE[:12], attention)
print(f"\nreal text {tok.decode(ids)!r}:")
for l in L_SHOW: print(f" layer {l:2d}  " + report("", light(C, l)).lstrip())
# control: the angle between two unrelated lights (different texts, same position)
ids2 = (refs[5]["ids"] + refs[5]["ref"])[:12]; C2, _ = run(WTE[ids2] + WPE[:12], attention)
print("\ncontrol, two unrelated texts at the same position (deg), position 6: " + " ".join(f"L{l}:{ang(light(C, l)[6], light(C2, l)[6]):.1f}" for l in L_SHOW) + "   (two random directions: about 90)")
