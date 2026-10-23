"""fp32 Granite-3.1-1B-A400M (base) and its own greedy words for the 30 prompts. Kept for each word: its lead over the
runner-up. Kept for each position: the router's gap between its 8th and 9th expert (the edge between lit and dark),
smallest over the 24 layers.

The fp32 sum is written here and reads the plates straight from the file. Only the 8 lit experts of a position are
read, so the process stays small.
    python3 ref_granite.py check     # this sum against the transformers code cut to 2 layers, same weights
    python3 ref_granite.py           # writes refs_granite.json
"""
import sys, json, glob, torch
from safetensors import safe_open
from transformers import AutoTokenizer
torch.set_grad_enabled(False)
REPO = "ibm-granite/granite-3.1-1b-a400m-base"
SNAP = glob.glob(f"/Users/abundancemachine/.cache/huggingface/hub/models--{REPO.replace('/', '--')}/snapshots/*/")[0]
F = {}
for f in glob.glob(SNAP + "*.safetensors"):
    h = safe_open(f, "pt")
    for k in h.keys(): F[k] = h
L, D, NH, NKV, HD, NE, TOPK, I = 24, 1024, 16, 8, 64, 32, 8, 512
EMB_MUL, ATT_MUL, RES_MUL, LOGIT_DIV, EPS, THETA = 12.0, 0.015625, 0.22, 6.0, 1e-6, 1500000.0
def get(k): return F[k].get_tensor(k).float()
def expert(k, e): return F[k].get_slice(k)[e].float()                 # one expert's plate, read alone
SMALL = {k: get(k) for k in F if "block_sparse_moe.input_linear" not in k and "block_sparse_moe.output_linear" not in k}

def rms(x, g): return g * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS))
def rot(x, cos, sin):
    a, b = x[..., :HD // 2], x[..., HD // 2:]; return x * cos + torch.cat([-b, a], -1) * sin

def forward(ids, layers=L):
    P = len(ids); x = SMALL["model.embed_tokens.weight"][ids] * EMB_MUL
    inv = 1.0 / (THETA ** (torch.arange(0, HD, 2, dtype=torch.int64).float() / HD))
    fr = torch.arange(P).float()[:, None] * inv[None, :]; emb = torch.cat([fr, fr], -1); cos, sin = emb.cos(), emb.sin()
    mask = torch.full((P, P), float("-inf")).triu(1); gaps = torch.full((P,), float("inf"))
    for l in range(layers):
        p = f"model.layers.{l}."
        h = rms(x, SMALL[p + "input_layernorm.weight"])
        q = (h @ SMALL[p + "self_attn.q_proj.weight"].T).view(P, NH, HD).transpose(0, 1)
        k = (h @ SMALL[p + "self_attn.k_proj.weight"].T).view(P, NKV, HD).transpose(0, 1)
        v = (h @ SMALL[p + "self_attn.v_proj.weight"].T).view(P, NKV, HD).transpose(0, 1)
        q, k = rot(q, cos, sin), rot(k, cos, sin)
        k, v = k.repeat_interleave(NH // NKV, 0), v.repeat_interleave(NH // NKV, 0)   # head h reads kv head h // 2
        a = torch.softmax(q @ k.transpose(1, 2) * ATT_MUL + mask, -1)
        x = x + (((a @ v).transpose(0, 1).reshape(P, D)) @ SMALL[p + "self_attn.o_proj.weight"].T) * RES_MUL
        h = rms(x, SMALL[p + "post_attention_layernorm.weight"])
        lg = h @ SMALL[p + "block_sparse_moe.router.layer.weight"].T
        s = lg.sort(-1, descending=True).values; gaps = torch.minimum(gaps, s[:, TOPK - 1] - s[:, TOPK])
        tv, ti = lg.topk(TOPK, -1); gate = torch.softmax(tv, -1); out = torch.zeros(P, D)
        for e in ti.unique().tolist():
            rows, slot = (ti == e).nonzero(as_tuple=True)
            gu = h[rows] @ expert(p + "block_sparse_moe.input_linear.weight", e).T
            y = (torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]) @ expert(p + "block_sparse_moe.output_linear.weight", e).T
            out.index_add_(0, rows, gate[rows, slot][:, None] * y)
        x = x + out * RES_MUL
    x = rms(x, SMALL["model.norm.weight"])
    return (x @ SMALL["model.embed_tokens.weight"].T) / LOGIT_DIV, gaps

tok = AutoTokenizer.from_pretrained(REPO)
if len(sys.argv) > 1 and sys.argv[1] == "check":
    from transformers import AutoConfig, GraniteMoeForCausalLM
    cfg = AutoConfig.from_pretrained(REPO); cfg.num_hidden_layers = 2
    m = GraniteMoeForCausalLM(cfg).float().eval()
    sd = {k: get(k) for k in F if "layers." not in k or int(k.split(".")[2]) < 2}
    missing, unexpected = m.load_state_dict(sd, strict=False); print("missing", missing, "unexpected", unexpected)
    for p in ["The capital of France is", "def fibonacci(n):"]:
        ids = tok(p).input_ids; ref = m(torch.tensor([ids])).logits[0]; mine, _ = forward(ids, layers=2)
        print(f"{p!r}: largest difference {float((ref - mine).abs().max()):.2e} (logits up to {float(ref.abs().max()):.1f}); "
              f"same top word at every position: {bool((ref.argmax(-1) == mine.argmax(-1)).all())}")
    sys.exit()

prompts = [r["prompt"] for r in json.load(open("../refs_many.json"))]
out = []
for p in prompts:
    ids = tok(p).input_ids; seq = list(ids); margins = []
    for _ in range(16):
        lg, gaps = forward(seq); t2 = lg[-1].topk(2)
        seq.append(int(t2.indices[0])); margins.append(round(float(t2.values[0] - t2.values[1]), 4))
    _, gaps = forward(seq[:-1])                                             # every position that made a word
    out.append(dict(prompt=p, ids=ids, ref=seq[len(ids):], margins=margins, router_gaps=[round(float(g), 5) for g in gaps]))
    print(f"{p!r:40s} -> {tok.decode(seq[len(ids):])!r}  (smallest lead {min(margins):.3f}, smallest router gap {float(gaps.min()):.4f})", flush=True)
json.dump(out, open("refs_granite.json", "w"))
