"""PyTorch GPT-2 greedy, 30 prompts x 16 words (refs_many.json), with its own key/value cache: words/s including the prompt
and after it, 5 timed passes after one warm-up pass; words compared with fp32's (the refs).
    python3 gpu_fr/torch_speed.py mps|cpu"""
import sys, json, time, statistics as st, torch
from transformers import GPT2LMHeadModel
torch.set_grad_enabled(False)
devname = sys.argv[1]; dev = torch.device(devname); dt = torch.float16 if devname == "mps" else torch.float32
m = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=dt).to(dev).eval()
refs = json.load(open("/Users/abundancemachine/streaming-machine/refs_many.json"))
def sync(): torch.mps.synchronize() if devname == "mps" else None
def run(r):
    ids = torch.tensor([r["ids"]], device=dev); sync(); t0 = time.perf_counter()
    out = m(ids, use_cache=True); w = int(out.logits[0, -1].argmax()); past = out.past_key_values; gen = [w]; sync(); t1 = time.perf_counter()
    for _ in range(15):
        out = m(torch.tensor([[w]], device=dev), past_key_values=past, use_cache=True); past = out.past_key_values; w = int(out.logits[0, -1].argmax()); gen.append(w)
    sync(); t2 = time.perf_counter(); return gen, t2 - t0, t2 - t1
for r in refs[:3]: run(r)                                                     # warm-up
inc, aft = [], []
for k in range(5):
    T = A = 0.0; same = 0
    for r in refs: g, t, a = run(r); T += t; A += a; same += g == r["ref"][:16]
    inc.append(480 / T); aft.append(450 / A)
    print(f"pass {k + 1}: {480 / T:.1f} words/s including the prompt, {450 / A:.1f} after it; words = fp32 on {same}/30", flush=True)
print(f"{devname} {dt}: median {st.median(inc):.1f} words/s including the prompt, {st.median(aft):.1f} after it")
