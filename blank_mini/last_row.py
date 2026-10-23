"""The blank mini model, end to end: the last row (every word's pile at the top read).
PREDICTION (written first): not all 0s. One word stands out -- the word that came in -- about 10x the spread above
the rest; every other word sits in speckle. The layers' part of the last row is speckle only, about the size of the
table part's speckle."""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
torch.set_grad_enabled(False)
tok = GPT2Tokenizer.from_pretrained("gpt2")
m = GPT2LMHeadModel.from_pretrained("blank_mini/model").double().eval(); tr = m.transformer
text = ("The capital of France is Paris. It is known for the Eiffel Tower, the Louvre and its cafes. "
        "Every year millions of people visit the city to see its museums and walk along the river.")
ids = tok(text, return_tensors="pt").input_ids; P = ids.shape[1]
blocks = []; hs = [b.register_forward_hook(lambda mod, i, o: blocks.append(o[0][0])) for b in tr.h]
out = m(ids); [h.remove() for h in hs]
top = blocks[-1]                                   # the stream at the top, before the top water level
came_in = tr.wte(ids)[0] + tr.wpe(torch.arange(P))
layers = top - came_in                             # everything the layers added, end to end
E = tr.wte.weight
row = out.logits[0]                                # the last row, as the model reads it
print(f"blank mini model, {P} positions, {E.shape[0]:,} words in the last row\n")
print("THE LAST ROW (every word's pile at the top), per position, median over positions:")
srt = row.sort(1, descending=True).values
spread = row.std(1)
inw = row.gather(1, ids[0][:, None])[:, 0]
print(f"  spread of all words' piles:           {spread.median():.3f}")
print(f"  the word that came in:                {inw.median():+.3f}  = {(inw/spread).median():5.1f}x the spread   (top word in {int((row.argmax(1)==ids[0]).sum())}/{P} positions)")
print(f"  the next-highest word:                {(srt[:,1]/spread).median():5.1f}x the spread")
print(f"  words within 1x the spread of 0:      {((row.abs() < spread[:,None]).double().mean()*100):.1f}%")
print("\nSPLIT, the last row without the top water level (straight dot with the table), median over positions:")
for name, part in [("what came in (table row + position)", came_in), ("everything the layers added", layers), ("the whole top stream", top)]:
    r = part @ E.T; sp = r.std(1); w = r.gather(1, ids[0][:, None])[:, 0]
    print(f"  {name:38s} speckle {sp.median():.4f} | the word that came in {(w/sp).median():6.1f}x | size {part.norm(dim=1).median():.3f}")
print(f"\nend to end: the layers' total push is {(layers.norm(dim=1)/came_in.norm(dim=1)).median():.2f}x the size of what came in;"
      f" pointing along what came in: {torch.nn.functional.cosine_similarity(layers, came_in, dim=1).median():+.3f}")
