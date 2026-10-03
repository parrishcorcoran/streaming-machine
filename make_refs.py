"""fp32 GPT-2's own greedy words for many prompts, with how far each word led its runner-up (the margin)."""
import json, torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
torch.set_grad_enabled(False)
tok = GPT2Tokenizer.from_pretrained("gpt2"); m = GPT2LMHeadModel.from_pretrained("gpt2").eval()
prompts = ["The capital of France is", "Once upon a time", "In 1492, Columbus", "The quick brown fox", "Water boils at",
  "My favorite food is", "The theory of relativity states that", "To make a cup of tea, you", "The president of the United States",
  "Two plus two equals", "The largest planet in our solar system is", "She opened the door and saw", "def fibonacci(n):",
  "The best way to learn a language is", "In the beginning God created", "The stock market fell today because",
  "Scientists have discovered a new species of", "I think therefore", "The recipe calls for two cups of", "It was a dark and stormy",
  "The meaning of life is", "Python is a programming language that", "The cat sat on the", "Breaking news: a major earthquake",
  "The first rule of fight club is", "Shakespeare wrote", "Photosynthesis is the process by which", "The Great Wall of China",
  "My name is John and I", "Tomorrow the weather will be"]
out = []
for p in prompts:
    ids = tok(p).input_ids; n = min(16, 32 - len(ids)); seq = ids[:]; margins = []
    for _ in range(n):
        lg = m(torch.tensor([seq])).logits[0, -1]; t2 = lg.topk(2)
        seq.append(int(t2.indices[0])); margins.append(round(float(t2.values[0] - t2.values[1]), 4))
    out.append(dict(prompt=p, ids=ids, ref=seq[len(ids):], margins=margins))
json.dump(out, open("refs_many.json", "w"))
print(f"{len(out)} prompts written")
