"""Turn the base into zero. The backward pass found the base (the average table row, 72% of each row's size) delivers
about 3% of the next token's lead: it cancels on the way up. Here every table row going in has its base set to zero
(row - average row), the top read untouched. Control: take away a random vector of the same size instead.
PREDICTION (written first): base to zero keeps the top word at 90-97% of positions; the control keeps far fewer."""
import json, torch
from transformers import GPT2LMHeadModel
torch.set_grad_enabled(False)
m = GPT2LMHeadModel.from_pretrained("gpt2").eval(); tr = m.transformer
W = tr.wte.weight; base = W.mean(0)
g = torch.Generator().manual_seed(0); rnd = torch.randn(base.shape, generator=g); rnd = rnd / rnd.norm() * base.norm()
texts = json.load(open("judge5_texts.json")); refs = json.load(open("refs_many.json"))
wins = [t["ids"] if isinstance(t, dict) else t for t in (texts if isinstance(texts, list) else texts["windows"])]
def run(take):
    same1 = in5 = n = 0
    for w in wins:
        ids = torch.tensor([w])
        full = m(ids).logits[0]
        x = W[ids[0]] - take
        cut = m(inputs_embeds=x[None]).logits[0]
        same1 += int((full.argmax(1) == cut.argmax(1)).sum()); in5 += int((cut.topk(5, 1).indices == full.argmax(1)[:, None]).any(1).sum()); n += len(w)
    kept = 0
    for r in refs:
        seq = torch.tensor([r["ids"] + r["ref"]]); x = W[seq[0]] - take
        top = m(inputs_embeds=x[None]).logits[0].argmax(1)[len(r["ids"]) - 1: -1]
        kept += int((top == torch.tensor(r["ref"])).sum())
    return same1 / n * 100, in5 / n * 100, kept
print(f"base size {base.norm():.3f}; a table row's median size {W.norm(dim=1).median():.3f}")
for name, take in [("nothing taken (check)", torch.zeros_like(base)), ("BASE TO ZERO", base), ("CONTROL: random of the same size", rnd)]:
    a, b, c = run(take)
    print(f"  {name:34s} same top word {a:6.2f}% | full word in its top 5 {b:6.2f}% | 30 texts: {c}/480 words")
