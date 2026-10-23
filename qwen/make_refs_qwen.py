"""fp32 Qwen3-0.6B's own greedy words for the 30 prompts, with each word's lead over the runner-up."""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B"); m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.float32).eval()
prompts = [r["prompt"] for r in json.load(open("../refs_many.json"))]
out = []
for p in prompts:
    ids = tok(p).input_ids; seq = list(ids); margins = []
    for _ in range(16):
        lg = m(torch.tensor([seq])).logits[0, -1]; t2 = lg.topk(2)
        seq.append(int(t2.indices[0])); margins.append(round(float(t2.values[0] - t2.values[1]), 4))
    out.append(dict(prompt=p, ids=ids, ref=seq[len(ids):], margins=margins))
    print(repr(p), "->", repr(tok.decode(seq[len(ids):])))
json.dump(out, open("refs_qwen.json", "w"))
