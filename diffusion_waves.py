"""Pieces settled per wave: the soup with GPT-2's plate against the soup with DiffuGPT's plate (GPT-2 re-exposed for
diffusion: every position sees every other, unknown positions hold a blank mark). Plain fp32 pass, written out here.
  reference : one piece per wave, left to right (the rest stay blank)
  soup      : every unknown position takes its current read each wave, until nothing changes
    python3 diffusion_waves.py
"""
import glob, json, math, torch
from safetensors.torch import load_file
from transformers import GPT2LMHeadModel
torch.set_grad_enabled(False)
refs = json.load(open("refs_many.json")); N, BOS, BLANK = 16, 50256, 10541

def plate_diffugpt():
    p = glob.glob("/Users/abundancemachine/.cache/huggingface/hub/models--diffusionfamily--diffugpt-s/snapshots/*/")[0]
    w = load_file(p + "model.safetensors"); w = {k.replace("denoise_model.", ""): v.float() for k, v in w.items()}
    w["wte.weight"] = w.pop("embed_tokens.weight"); return w
def plate_gpt2():
    return {k.replace("transformer.", ""): v.float() for k, v in GPT2LMHeadModel.from_pretrained("gpt2").state_dict().items() if k.startswith("transformer.")}

def ln(x, w, b): m = x.mean(-1, keepdim=True); v = ((x - m) ** 2).mean(-1, keepdim=True); return (x - m) / torch.sqrt(v + 1e-5) * w + b
def gelu(x): return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
def reads(W, ids, both_ways):
    P = len(ids); x = W["wte.weight"][ids] + W["wpe.weight"][:P]
    for l in range(12):
        g = lambda k: W[f"h.{l}.{k}"]
        h = ln(x, g("ln_1.weight"), g("ln_1.bias")) @ g("attn.c_attn.weight") + g("attn.c_attn.bias")
        q, k, v = h.split(768, -1); q, k, v = [t.view(P, 12, 64).transpose(0, 1) for t in (q, k, v)]
        s = q @ k.transpose(1, 2) / 8
        if not both_ways: s = s.masked_fill(torch.triu(torch.ones(P, P, dtype=torch.bool), 1), -1e30)
        a = (torch.softmax(s, -1) @ v).transpose(0, 1).reshape(P, 768)
        x = x + a @ g("attn.c_proj.weight") + g("attn.c_proj.bias")
        x = x + gelu(ln(x, g("ln_2.weight"), g("ln_2.bias")) @ g("mlp.c_fc.weight") + g("mlp.c_fc.bias")) @ g("mlp.c_proj.weight") + g("mlp.c_proj.bias")
    return (ln(x, W["ln_f.weight"], W["ln_f.bias"]) @ W["wte.weight"].T).argmax(-1)      # position i's read is the piece at i+1

def run(W, both_ways, prompt_ids):
    pre = [BOS] + prompt_ids; S = len(pre)
    ref = pre + [BLANK] * N                                   # one piece per wave, left to right
    for k in range(N): ref[S + k] = int(reads(W, ref, both_ways)[S + k - 1])
    cur = pre + [BLANK] * N; hist = []                        # the soup: every unknown position takes its read, every wave
    for wave in range(1, 60):
        r = reads(W, cur, both_ways); new = pre + [int(r[S + k - 1]) for k in range(N)]
        hist.append(new[S:])
        if new == cur: break
        cur = new
    fin = hist[-1]; settled = []
    for j in range(N):
        k = len(hist)
        while k > 0 and hist[k - 1][j] == fin[j]: k -= 1
        settled.append(k + 1)
    waves = len(hist) - 1 if hist[-1] == hist[-2] else len(hist)
    return ref[S:], fin, waves, sum(a == b for a, b in zip(ref[S:], fin)), settled

from transformers import GPT2Tokenizer
tok = GPT2Tokenizer.from_pretrained("gpt2")
for name, W, both in [("GPT-2 plate", plate_gpt2(), False), ("DiffuGPT plate", plate_diffugpt(), True)]:
    tw, tm = 0, 0
    print(f"\n#### {name}")
    for r in refs:
        ref, fin, waves, match, settled = run(W, both, r["ids"])
        tw += waves; tm += match
        if refs.index(r) < 4: print(f"  {r['prompt']!r}\n    one at a time: {tok.decode(ref)!r}\n    soup ({waves} waves, {match}/16 same): {tok.decode(fin)!r}")
    print(f"  over 30 texts: {30 * N} pieces in {tw} waves = {30 * N / tw:.2f} pieces per wave; soup's text same as one-at-a-time: {tm}/{30 * N}")
