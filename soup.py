"""THE SOUP. Every layer, every position, every step, all at once. Nothing waits and nothing is reset.

The prompt and the next N words are all in the machine from the start. Every step, every layer works on what the
layer below passed up the step before, for every position at once. Every future position's word is whatever the top
says at the position before it, right now, and is re-read every step. Wrong words are simply overwritten as the flow
corrects itself. Each position only looks backward, so the soup can only come to rest on the model's own text.

Printed per step: how many of the N words are already the model's word, and what the soup currently says.
    python3 soup.py [--words 16] [--prompt "..."]
"""
import argparse, time, torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

torch.set_grad_enabled(False)
ap = argparse.ArgumentParser()
ap.add_argument("--prompt", default="The capital of France is")
ap.add_argument("--words", type=int, default=16)
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--flow", default="now", choices=["now", "shuffle", "lagged"], help="now: each layer reads what is below it now (bottom up); shuffle: the same in a random order each step; lagged: what was below it last step")
a = ap.parse_args()

tok = GPT2Tokenizer.from_pretrained("gpt2")
m = GPT2LMHeadModel.from_pretrained("gpt2").double().eval(); tr = m.transformer; L = len(tr.h)
ids = tok(a.prompt, return_tensors="pt").input_ids[0].tolist(); S, N = len(ids), a.words

want = ids[:]                                                    # the model's own text, one word at a time
for _ in range(N): want.append(int(m(torch.tensor([want])).logits[0, -1].argmax()))
want = want[S:]

T = ids + [ids[-1]] * N                                          # the future starts as copies of the last word
P = S + N
state = [torch.zeros(1, P, 768, dtype=torch.float64) for _ in range(L + 1)]
print(f"{a.prompt!r} + {N} words, all {P} positions and all {L} layers flowing at once; each layer reads {a.flow}")
print(f"model : {tok.decode(want)!r}\n")
t0 = time.time(); first_right = [None] * N; rest = None
for step in range(1, a.steps + 1):
    state[0] = tr.wte(torch.tensor([T])) + tr.wpe(torch.arange(P))[None]
    run = lambda k, h: (lambda o: o[0] if isinstance(o, tuple) else o)(tr.h[k](h))
    if a.flow == "lagged":
        new = [state[0]] + [run(k, state[k]) for k in range(L)]                  # each layer reads last step's
    else:
        new = list(state); order = list(range(L)) if a.flow == "now" else torch.randperm(L).tolist()
        for k in order: new[k + 1] = run(k, new[k])                              # each layer reads what is below it now
    moved = sum(int((new[k] != state[k]).any(-1).sum()) for k in range(1, L + 1)); state = new
    tops = (tr.ln_f(state[L])[0] @ tr.wte.weight.T).argmax(-1).tolist()
    newT = ids + [tops[S - 1 + j] for j in range(N)]              # every future word: the top at the position before it, now
    changed = sum(x != y for x, y in zip(newT, T)); T = newT
    right = [T[S + j] == want[j] for j in range(N)]
    for j in range(N):
        if right[j] and first_right[j] is None: first_right[j] = step
        if not right[j]: first_right[j] = None
    if step <= 20 or step % 10 == 0 or (moved == 0 and changed == 0):
        print(f"  step {step:3d}: {sum(right):2d}/{N} words right, {changed:2d} words changed, {moved:4d} position-layers moving   {tok.decode(T[S:])!r}")
    if moved == 0 and changed == 0: rest = step; break
dt = time.time() - t0
print(f"\n{'came to rest' if rest else 'still moving'} at step {rest or a.steps}; "
      f"{'IDENTICAL' if T[S:] == want else 'DIFFERENT'} to the model; {dt:.1f} s")
print(f"one word at a time, one layer a step, this would take {13 * N} steps; each word was right from step: {first_right}")
