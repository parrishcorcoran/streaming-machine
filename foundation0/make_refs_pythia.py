"""The 30 texts of ../refs_many.json, carried over to Pythia, and fp32 Pythia's own greedy words for each.

  1. Each text's GPT-2 ids are decoded with GPT-2's tokenizer (checked equal to the stored prompt) and the text is
     cut again with Pythia's own tokenizer (no start token added; Pythia's files had none in front of a text).
  2. For each size, fp32 Pythia (the finished plates, step143000, fp16 values widened to fp32, eager arrival) gives its
     own 16 greedy words: the whole text run again for each word, as ../make_refs.py does for GPT-2, with how far each
     word led its runner-up (the margin) and its top 5.
Writes texts_pythia.json and refs_pythia-<size>.json.
    python3 make_refs_pythia.py [sizes=14m,70m,160m] [words=16]
"""
import os, sys, json, torch
from transformers import GPT2Tokenizer, AutoTokenizer, GPTNeoXForCausalLM

A = dict(a.split("=") for a in sys.argv[1:])
SIZES, WORDS = A.get("sizes", "14m,70m,160m").split(","), int(A.get("words", 16))
HERE = os.path.dirname(os.path.abspath(__file__)); MD = os.path.join(HERE, "models")
torch.set_grad_enabled(False)
g2 = GPT2Tokenizer.from_pretrained("gpt2"); pt = AutoTokenizer.from_pretrained(os.path.join(MD, "tokenizer"))
texts = []
for r in json.load(open(os.path.join(HERE, "..", "refs_many.json"))):
    t = g2.decode(r["ids"]); assert t == r["prompt"], (t, r["prompt"])
    ids = pt(t, add_special_tokens=False).input_ids; assert pt.decode(ids) == t, (pt.decode(ids), t)
    texts.append(dict(prompt=t, gpt2_ids=r["ids"], ids=ids))
assert pt("x").input_ids == pt("x", add_special_tokens=False).input_ids, "the tokenizer adds a start token"
json.dump(texts, open(os.path.join(HERE, "texts_pythia.json"), "w"), indent=0)
print(f"{len(texts)} texts; Pythia ids per text {min(len(t['ids']) for t in texts)}-{max(len(t['ids']) for t in texts)}")
for size in SIZES:
    m = GPTNeoXForCausalLM.from_pretrained(os.path.join(MD, f"pythia-{size}-step143000"), torch_dtype=torch.float32,
                                           attn_implementation="eager").eval()
    out = []
    for t in texts:
        seq = list(t["ids"]); margins, top5 = [], []
        for _ in range(WORDS):
            lg = m(torch.tensor([seq])).logits[0, -1].double(); tk = lg.topk(5)
            seq.append(int(tk.indices[0])); margins.append(round(float(tk.values[0] - tk.values[1]), 5)); top5.append(tk.indices.tolist())
        out.append(dict(prompt=t["prompt"], ids=t["ids"], ref=seq[len(t["ids"]):], margins=margins, top5=top5))
    json.dump(out, open(os.path.join(HERE, f"refs_pythia-{size}.json"), "w"), indent=0)
    print(f"pythia-{size}: {len(out)} texts x {WORDS} words; smallest margins {sorted(x for o in out for x in o['margins'])[:5]}")
    for o in out[:3]: print(f"   {o['prompt']!r:30s} -> {pt.decode(o['ref'])!r}")
    del m
