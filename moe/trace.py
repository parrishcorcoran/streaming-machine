"""Trace the whole-number Granite machine against fp32, part by part, layer by layer, on one prompt."""
import sys, glob, numpy as np, torch
NL = int(sys.argv[1]) if len(sys.argv) > 1 else 24; sys.argv = sys.argv[:1]
import granite_base2 as g
from safetensors import safe_open
from transformers import AutoTokenizer
torch.set_grad_enabled(False)
SNAP = glob.glob("/Users/abundancemachine/.cache/huggingface/hub/models--ibm-granite--granite-3.1-1b-a400m-base/snapshots/*/")[0]
F = {}
for f in glob.glob(SNAP + "*.safetensors"):
    h = safe_open(f, "pt")
    for k in h.keys(): F[k] = h
def get(k): return F[k].get_tensor(k).double()
def ex(k, e): return F[k].get_slice(k)[e].double()
def rms(x, w): return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))
def rot(x, cos, sin): a, b = x[..., :32], x[..., 32:]; return x * cos + torch.cat([-b, a], -1) * sin
def rel(mach, ref):                                          # machine whole numbers at 2^16 against fp
    m = torch.tensor(mach / 2.0 ** 16); return float((m - ref).norm() / ref.norm())
tok = AutoTokenizer.from_pretrained("ibm-granite/granite-3.1-1b-a400m-base")
ids = tok("The capital of France is").input_ids; P = len(ids); pos = np.arange(P)
x = get("model.embed_tokens.weight")[ids] * 12.0; R = np.asarray(g.EMBED[ids]).astype(np.int64)
print(f"entry: {rel(R, x):.2e}")
inv = 1.0 / (1500000.0 ** (torch.arange(0, 64, 2).double() / 64)); fr = torch.arange(P).double()[:, None] * inv[None, :]
emb = torch.cat([fr, fr], -1); cos, sin = emb.cos(), emb.sin(); mask = torch.full((P, P), float("-inf")).triu(1).double()
for l in range(NL):
    p = f"model.layers.{l}."
    h = rms(x, get(p + "input_layernorm.weight"))
    q = h @ get(p + "self_attn.q_proj.weight").T * 0.015625; k = h @ get(p + "self_attn.k_proj.weight").T; v = h @ get(p + "self_attn.v_proj.weight").T
    QKV = g.gauged_slab(R, g.plate(f"qkv.{l}"))
    e_q, e_k, e_v = rel(QKV[:, :1024], q), rel(QKV[:, 1024:1536], k), rel(QKV[:, 1536:], v)
    qh = rot(q.view(P, 16, 64).transpose(0, 1), cos, sin); kh = rot(k.view(P, 8, 64).transpose(0, 1), cos, sin)
    vh = v.view(P, 8, 64).transpose(0, 1)
    a = torch.softmax(qh @ kh.repeat_interleave(2, 0).transpose(1, 2) + mask, -1)
    att = (a @ vh.repeat_interleave(2, 0)).transpose(0, 1).reshape(P, 1024)
    ATT = g.attention(QKV, pos); e_att = rel(ATT, att)
    x = x + att @ get(p + "self_attn.o_proj.weight").T * 0.22
    R = R + g.plain_slab(ATT, g.plate(f"o.{l}")); e_x1 = rel(R, x)
    h = rms(x, get(p + "post_attention_layernorm.weight"))
    lg = h @ get(p + "block_sparse_moe.router.layer.weight").T
    tv, ti = lg.topk(8, -1); gate = torch.softmax(tv, -1); out = torch.zeros(P, 1024, dtype=torch.float64)
    for e in ti.unique().tolist():
        rows, slot = (ti == e).nonzero(as_tuple=True)
        gu = h[rows] @ ex(p + "block_sparse_moe.input_linear.weight", e).T
        y = (torch.nn.functional.silu(gu[:, :512]) * gu[:, 512:]) @ ex(p + "block_sparse_moe.output_linear.weight", e).T
        out.index_add_(0, rows, gate[rows, slot][:, None] * y)
    rp = g.gauged_slab(R, g.plate(f"router.{l}")); e_r = rel(rp, lg)
    OUT, lit = g.experts(R, l)
    same_lit = sum(set(lit[i].tolist()) == set(ti[i].tolist()) for i in range(P))
    e_out = rel(OUT, out * 0.22)
    x = x + out * 0.22; R = R + OUT
    print(f"layer {l:2d}: q {e_q:.1e} k {e_k:.1e} v {e_v:.1e} | arrival {e_att:.1e} | stream {e_x1:.1e} | router {e_r:.1e}, "
          f"same 8 lit {same_lit}/{P} | experts' write {e_out:.1e} | stream {rel(R, x):.1e} | largest |stream| {np.abs(R).max() / 65536:.1f}", flush=True)
