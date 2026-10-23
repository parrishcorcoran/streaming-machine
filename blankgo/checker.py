"""Independent rebuild of the builds, by a separate checker (rule 10). Shares nothing with model.py, snapio.py or
build.py except the files they wrote: its own blank reader, its own shave replay, its own windows, and Hugging Face's
own GPT-2 (GPT2LMHeadModel, float64, untied output table, gelu_new = GPT-2's tanh bend) in place of model.forward.
    python3 checker.py width=64 layers=2 recipe=A [replay=30]

  1. Every snapshot: W rebuilt from blank_<shape>.npz + snaps/<..>_step<t>.npz (C: the shave chain replayed here with
     its own loop), loaded into GPT2LMHeadModel. Held-out loss on the 128 windows (rebuilt here from data.npz) against
     build_<..>.json; top word at the 30 texts' 480 positions against evals/eval_<..>.json's finished words.
  2. The building replayed from the blank for the first `replay` pushes with GPT2LMHeadModel's slopes, its own
     plain/scaled push and shave, each rounded to 2^-30 and added as whole numbers; P compared with the stored P at
     step 10 and 30 (and the stored snapshots up to `replay`).

  3. (added after the first w64 L2 run, before any other) the blank taken out (P alone in the drawn parts) and the
     blank scrambled within each neuron (draw 1; the permutation rebuilt here from the same seed rule as snapio's),
     held-out loss against evals/eval_<..>.json's 'pushes' and 'scr1'.

PREDICTIONS (the session's, written before any run of this script):
  K1. Held-out loss equal to the build's within 1e-9 at every snapshot; top words 480 of 480 at every snapshot.
  K2. The replayed pushes equal the stored ones in all but a few values per million at steps 10 and 30, and those
      differ by 1-2 grains (2^-30): the slopes come from a different order of summation in the other code.
  K3. (written with part 3, before its first run) pushes-alone and scr1 held-out losses equal eval.py's within 1e-9.
"""
import os, sys, json, math, numpy as np, torch, torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

A = dict(a.split("=") for a in sys.argv[1:])
d, L, RC = int(A.get("width", 64)), int(A.get("layers", 2)), A.get("recipe", "A")
REPLAY = int(A.get("replay", 30))
HERE = os.path.dirname(os.path.abspath(__file__)); G = 2.0 ** 30
sh = f"w{d}_L{L}"; nm = f"{sh}_{RC}"
log = json.load(open(os.path.join(HERE, f"build_{nm}.json")))
ev = json.load(open(os.path.join(HERE, "evals", f"eval_{nm}.json")))
blank = dict(np.load(os.path.join(HERE, f"blank_{sh}.npz")))
z = np.load(os.path.join(HERE, "data.npz")); build_ids, held = z["build"].astype(np.int64), z["held"].astype(np.int64)
VOC, CTX = 8193, 128
starts = np.linspace(0, len(held) - CTX - 2, 128).astype(np.int64); win = np.stack([held[s:s + CTX + 1] for s in starts])
texts = torch.tensor(ev["texts"])
torch.set_default_dtype(torch.float64)
cfg = GPT2Config(vocab_size=VOC, n_positions=CTX, n_embd=d, n_layer=L, n_head=d // 16, activation_function="gelu_new",
                 resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, layer_norm_epsilon=1e-5, tie_word_embeddings=False)
hf = GPT2LMHeadModel(cfg).double()
DRAWN = ["wte", "wpe", "head"] + [f"{k}.{l}.w" for l in range(L) for k in ("qkv", "proj", "fc", "mproj")]

def hf_name(n):                                       # this folder's names -> GPT2LMHeadModel's
    if n == "wte": return "transformer.wte.weight"
    if n == "wpe": return "transformer.wpe.weight"
    if n == "head": return "lm_head.weight"
    if n.startswith("lnf"): return "transformer.ln_f." + ("weight" if n.endswith("g") else "bias")
    k, l, p = n.split(".")
    sub = {"ln1": "ln_1", "ln2": "ln_2", "qkv": "attn.c_attn", "proj": "attn.c_proj", "fc": "mlp.c_fc", "mproj": "mlp.c_proj"}[k]
    return f"transformer.h.{l}.{sub}." + ("weight" if p in ("w", "g") else "bias")

params = dict(hf.named_parameters())
assert set(hf_name(n) for n in blank) == set(params), "names do not cover the model"

def load(W):
    with torch.no_grad():
        for n, a in W.items():
            x = torch.from_numpy(a.astype(np.float64) / G)
            params[hf_name(n)].copy_(x.T if n == "head" else x)        # lm_head keeps [words][lanes]

def held_loss():
    tot = []
    with torch.no_grad():
        for i in range(0, 128, 16):
            w = torch.from_numpy(win[i:i + 16]); lg = hf(w[:, :-1]).logits
            tot.append(F.cross_entropy(lg.reshape(-1, VOC), w[:, 1:].reshape(-1), reduction="none").view(16, -1).mean(1))
    return float(torch.cat(tot).mean())

def lr(t, peak):
    return peak * t / 30 if t <= 30 else peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (t - 30) / 2970)))

peak, wd = log["peak"], log["wd"]
print(f"checker {nm}")
# 1. every snapshot
Bt = {n: a.copy() for n, a in blank.items()}; done = 0; worst = 0.0; worst_var = 0.0; words_ok = []
for t in sorted(int(s) for s in log["snaps"]):
    for s in range(done + 1, t + 1):
        if wd:
            f = lr(s, peak) * wd
            for n in DRAWN: Bt[n] = Bt[n] - np.rint(Bt[n] * f).astype(np.int64)
    done = t
    sp = np.load(os.path.join(HERE, "snaps", f"{nm}_step{t}.npz"))
    W = {n: Bt[n] + sp["P." + n] for n in blank}
    load(W); lo = held_loss(); diff = abs(lo - log["snaps"][str(t)]["held_loss"]); worst = max(worst, diff)
    with torch.no_grad(): top = hf(texts).logits[:, 8:, :].argmax(-1).numpy()
    ok = int((top == np.array(ev["snaps"][str(t)]["finished"]["top1"])).sum()); words_ok.append(ok)
    print(f"  step {t:5d}  held-out loss {lo:.10f}  build's {log['snaps'][str(t)]['held_loss']:.10f}  |diff| {diff:.2e}  top words {ok}/480", flush=True)
    Pn = {n: sp["P." + n] for n in blank}
    def scr1(n, a):
        rng = np.random.default_rng(7000 + 101 * 1 + sum(map(ord, n))); ax = 1 if n in ("wte", "wpe") else 0
        return np.take_along_axis(a, np.argsort(rng.random(a.shape), axis=ax), axis=ax)
    for var, Wv in (("pushes", {n: (Pn[n] if n in DRAWN else W[n]) for n in blank}),
                    ("scr1", {n: (Pn[n] + scr1(n, Bt[n]) if n in DRAWN else W[n]) for n in blank})):
        load(Wv); lv = held_loss(); dv = abs(lv - ev["snaps"][str(t)][var]["held_loss"]); worst_var = max(worst_var, dv)
        print(f"         {var:7s} held-out loss {lv:.10f}  eval's {ev['snaps'][str(t)][var]['held_loss']:.10f}  |diff| {dv:.2e}", flush=True)
# 2. replay the first pushes
W = {n: a.copy() for n, a in blank.items()}; Bc = {n: a.copy() for n, a in blank.items()}; Pc = {n: np.zeros_like(a) for n, a in blank.items()}
m = {n: np.zeros(a.shape) for n, a in blank.items()}; v = {n: np.zeros(a.shape) for n, a in blank.items()}
st = np.random.default_rng(20261006 + 1).integers(0, len(build_ids) - CTX - 1, size=(3000, 16))
rep = {}
for t in range(1, REPLAY + 1):
    load(W); hf.zero_grad()
    ids = torch.from_numpy(np.stack([build_ids[s:s + CTX + 1] for s in st[t - 1]]))
    lg = hf(ids[:, :-1]).logits
    F.cross_entropy(lg.reshape(-1, VOC), ids[:, 1:].reshape(-1)).backward()
    a = lr(t, peak)
    for n in blank:
        g = params[hf_name(n)].grad.numpy(); g = g.T if n == "head" else g
        if RC == "A": u = a * g
        else:
            m[n] = 0.9 * m[n] + 0.1 * g; v[n] = 0.95 * v[n] + 0.05 * g * g
            u = a * (m[n] / (1 - 0.9 ** t)) / (np.sqrt(v[n] / (1 - 0.95 ** t)) + 1e-8)
        push = -np.rint(u * G).astype(np.int64)
        sb = sp_ = 0
        if wd and n in DRAWN:
            sb = np.rint(Bc[n] * (a * wd)).astype(np.int64); sp_ = np.rint(Pc[n] * (a * wd)).astype(np.int64)
        Bc[n] = Bc[n] - sb; Pc[n] = Pc[n] + push - sp_; W[n] = Bc[n] + Pc[n]
    if str(t) in log["snaps"]:
        sp = np.load(os.path.join(HERE, "snaps", f"{nm}_step{t}.npz"))
        dif = np.concatenate([(Pc[n] - sp["P." + n]).ravel() for n in blank]); tot = dif.size
        rep[t] = dict(values=int(tot), differ=int((dif != 0).sum()), max_grains=int(np.abs(dif).max()))
        print(f"  replay step {t}: pushes differ in {rep[t]['differ']:,} of {tot:,} values; largest difference {rep[t]['max_grains']} grains", flush=True)
json.dump(dict(worst_loss_diff=worst, worst_removed_scrambled_diff=worst_var, words=words_ok, replay=rep), open(os.path.join(HERE, "evals", f"checker_{nm}.json"), "w"), indent=1)
print(f"worst |held-out loss difference| {worst:.2e} (finished), {worst_var:.2e} (blank out, scrambled); top words {min(words_ok)}-{max(words_ok)} of 480")
