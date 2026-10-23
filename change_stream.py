"""The machine stays hot; piece after piece goes in; each piece's change is marked (XOR of before and after).
Earlier spots never change, so each piece's marks are exactly its own row at every depth and part.
    python3 change_stream.py [text number]
"""
import sys, json, numpy as np
T = int(sys.argv[1]) if len(sys.argv) > 1 else 0
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]; sys.argv = ["x"]; exec(src)
from transformers import GPT2Tokenizer; tok = GPT2Tokenizer.from_pretrained("gpt2")
refs = json.load(open("refs_many.json"))
exec(open("change_map.py").read().split("r = refs[0]")[0].split("refs = json.load")[1].split("\n", 1)[1])   # state() and marks()
r = refs[T]; ids = r["ids"] + r["ref"][:16]; S = len(r["ids"]); st = state(ids)
parts = ["stream", "query", "key", "value", "arrival", "edge buckets"]
print(f"{r['prompt']!r}, then 16 pieces, the machine hot the whole time\n")
print("piece  word              marked   top places(12+)  bottom(0-11)   " + " ".join(f"{p[:6]:>7s}" for p in parts))
for p in range(S, len(ids)):
    tot, top, byp = 0, 0, {q: 0 for q in parts}
    for k, v in st.items():
        n, pl = marks(np.zeros_like(v[p:p + 1]), v[p:p + 1]); tot += n; top += int(pl[12:].sum()); byp[k[1]] += n
    print(f"  {p - S + 1:2d}   {tok.decode([ids[p]])!r:16s} {tot:8,}   {top:8,}        {tot - top:8,}     " + " ".join(f"{byp[q]:7,}" for q in parts))
