"""Independent rebuild of work.py, inplace.py and remove.py's readings, by separate code (rule 10). Shares nothing with
model.py, snapio.py, bmachine.py, work.py, inplace.py or remove.py except the files the builds wrote: its own blank
reader, its own shave replay, its own windows, Hugging Face's own GPT-2 (GPT2LMHeadModel, float64, untied output table,
gelu_new) with hooks that keep the light entering every plate, and its own lane orders and scrambles (different draws).
    python3 workcheck.py width=64 layers=2 recipe=A

It reads the float model's light, not the whole-number machine's, at weights n x 2^-30 (not the 16-place machine
files); the machine agrees with the float to 0.0066 nats in the lead, so the readings should agree closely but not
digit for digit.
  1. per plate, every snapshot, on the same 128 held-out windows: k = -2<b,d>/(|b|^2+|d|^2), gamma = 1 + <d,b>/<b,b>,
     b = light . Bt, d = light . P; the dodge ratio |light . Bt| / |light[lanes in a random order] . Bt| (3 draws, its
     own orders); the cancelled share of the finished piles and of the blank's part, 1 - sum|pile| / sum |x||w|.
  2. in place: s = (W . W0)/(W0 . W0) per drawn part; lane sums of P on every write line (exact int64).
  3. remove: float top word at the 30 texts' 480 positions for P alone and for the blank alone (gains, offsets, biases
     finished) against the finished; held-out loss of P alone and of the blank alone on remove.py's first 64 windows.

PREDICTIONS (the session's, written before any run of this script):
  K1. k within 0.01 and gamma within 0.03 (or 3% where |gamma| > 1) of work.py's, every plate and snapshot; dodge
      ratio within 0.03; cancelled shares within 0.005.
  K1b. the size of the blank's part of the piles, RMS of b, within 1% of work.py's (draw-free; the dodge ratio also
      carries each script's own lane-order draws, so K1's 0.03 may be missed by the draws alone).
  K2. in-place s within 1e-9 of inplace.py's; every write line's lane sum the same whole number.
  K3. top word as finished within 5 of 480 of remove.py's machine counts for P alone and the blank alone; the held-out
      losses of P alone and the blank alone within 0.01 of remove.py's machine losses (same 64 windows).
"""
import os, sys, json, math, numpy as np, torch, torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

np.seterr(all="ignore")
A = dict(a.split("=") for a in sys.argv[1:])
d, L, RC = int(A.get("width", 64)), int(A.get("layers", 2)), A.get("recipe", "A")
HERE = os.path.dirname(os.path.abspath(__file__)); G = 2.0 ** 30
sh = f"w{d}_L{L}"; nm = f"{sh}_{RC}"
log = json.load(open(os.path.join(HERE, f"build_{nm}.json")))
blank = dict(np.load(os.path.join(HERE, f"blank_{sh}.npz")))
z = np.load(os.path.join(HERE, "data.npz")); held = z["held"].astype(np.int64)
VOC, CTX = 8193, 128
starts = np.linspace(0, len(held) - CTX - 2, 128).astype(np.int64); win = np.stack([held[s:s + CTX + 1] for s in starts])
texts = torch.tensor(json.load(open(os.path.join(HERE, "evals", f"eval_{nm}.json")))["texts"])
WK = json.load(open(os.path.join(HERE, "evals", f"work_{nm}.json")))
IP = json.load(open(os.path.join(HERE, "evals", f"inplace_{nm}.json")))
RM = json.load(open(os.path.join(HERE, "evals", f"remove_{nm}.json")))
torch.set_default_dtype(torch.float64)
cfg = GPT2Config(vocab_size=VOC, n_positions=CTX, n_embd=d, n_layer=L, n_head=d // 16, activation_function="gelu_new",
                 resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, layer_norm_epsilon=1e-5, tie_word_embeddings=False)
hf = GPT2LMHeadModel(cfg).double().eval()
DRAWN = ["wte", "wpe", "head"] + [f"{k}.{l}.w" for l in range(L) for k in ("qkv", "proj", "fc", "mproj")]

def hf_name(n):
    if n == "wte": return "transformer.wte.weight"
    if n == "wpe": return "transformer.wpe.weight"
    if n == "head": return "lm_head.weight"
    if n.startswith("lnf"): return "transformer.ln_f." + ("weight" if n.endswith("g") else "bias")
    k, l, p = n.split(".")
    sub = {"ln1": "ln_1", "ln2": "ln_2", "qkv": "attn.c_attn", "proj": "attn.c_proj", "fc": "mlp.c_fc", "mproj": "mlp.c_proj"}[k]
    return f"transformer.h.{l}.{sub}." + ("weight" if p in ("w", "g") else "bias")

params = dict(hf.named_parameters())
def load(W):
    with torch.no_grad():
        for n, a in W.items():
            x = torch.from_numpy(a.astype(np.float64) / G); params[hf_name(n)].copy_(x.T if n == "head" else x)

MODS = {}
for l in range(L):
    blk = hf.transformer.h[l]
    MODS[f"qkv.{l}"], MODS[f"proj.{l}"], MODS[f"fc.{l}"], MODS[f"mproj.{l}"] = blk.attn.c_attn, blk.attn.c_proj, blk.mlp.c_fc, blk.mlp.c_proj
MODS["head"] = hf.lm_head
CAP = {}
for k, mod in MODS.items():
    mod.register_forward_hook(lambda mod, inp, out, k=k: CAP.__setitem__(k, inp[0].detach().reshape(-1, inp[0].shape[-1]).numpy().copy()))
PART = lambda k: k + ".w" if k != "head" else "head"

def lr(t, peak):
    return peak * t / 30 if t <= 30 else peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (t - 30) / 2970)))

peak, wd = log["peak"], log["wd"]
Bt = {n: a.copy() for n, a in blank.items()}; done = 0
worst = dict(k=0.0, gamma=0.0, dodge=0.0, rms_b_rel=0.0, cancel=0.0, s=0.0, lane_mismatch=0, top1=0, blank_loss=0.0)
out = {}
print(f"workcheck {nm}", flush=True)
for t in sorted(int(s) for s in log["snaps"]):
    for s_ in range(done + 1, t + 1):
        if wd:
            f = lr(s_, peak) * wd
            for n in DRAWN: Bt[n] = Bt[n] - np.rint(Bt[n] * f).astype(np.int64)
    done = t
    sp = np.load(os.path.join(HERE, "snaps", f"{nm}_step{t}.npz"))
    P = {n: sp["P." + n] for n in blank}; W = {n: Bt[n] + P[n] for n in blank}
    # 2. in place and lane sums
    for n in DRAWN:
        w0 = blank[n].astype(np.float64).ravel()
        s_val = float(W[n].astype(np.float64).ravel() @ w0 / (w0 @ w0))
        worst["s"] = max(worst["s"], abs(s_val - IP["snaps"][str(t)]["in_place"][n]["s"]))
    lines = [P["wte"], P["wpe"]] + [P[f"{k}.{l}.{p}"] if p == "w" else P[f"{k}.{l}.{p}"][None, :] for l in range(L) for k in ("proj", "mproj") for p in ("w", "b")]
    allsum = np.concatenate([x.sum(1) for x in lines])
    rms = float(np.sqrt((allsum.astype(np.float64) ** 2).mean()))
    rms_ip = IP["snaps"][str(t)]["lanesum"]["whole_model"]["rms_sum_grains"]
    mx_ip = IP["snaps"][str(t)]["lanesum"]["whole_model"]["max_abs_sum_grains"]
    if int(np.abs(allsum).max()) != mx_ip or abs(rms - rms_ip) > 1e-6 * max(rms, 1): worst["lane_mismatch"] += 1
    # 1. the work on the float light
    load(W)
    pl_parts = {k: (Bt[PART(k)].astype(np.float64) / G, P[PART(k)].astype(np.float64) / G) for k in MODS}
    acc = {k: dict(bb=0.0, dd=0.0, bd=0.0, at=0.0, ab=0.0, mw=0.0, ms=0.0, perm=[0.0] * 3, n=0) for k in MODS}
    rng = np.random.default_rng(424242)
    perms = {k: [rng.permutation(pl_parts[k][0].shape[0]) for _ in range(3)] for k in MODS}
    with torch.no_grad():
        for i in range(128):
            hf(torch.from_numpy(win[i:i + 1, :-1]))
            for k in MODS:
                x = CAP[k]; Bm, Pm = pl_parts[k]
                b = x @ Bm; dd = x @ Pm; a = acc[k]
                a["bb"] += float((b * b).sum()); a["dd"] += float((dd * dd).sum()); a["bd"] += float((b * dd).sum())
                a["at"] += float(np.abs(b + dd).sum()); a["ab"] += float(np.abs(b).sum()); a["n"] += b.size
                ax = np.abs(x); a["mw"] += float((ax @ np.abs(Bm + Pm)).sum()); a["ms"] += float((ax @ np.abs(Bm)).sum())
                for j in range(3):
                    bp = x[:, perms[k][j]] @ Bm; a["perm"][j] += float((bp * bp).sum())
    rec = {}
    for k, a in acc.items():
        wk = WK["snaps"][str(t)]["plates"][k]
        r = dict(dodge=math.sqrt(a["bb"] / np.mean(a["perm"])), cancel_finished=1 - a["at"] / a["mw"], cancel_blank=1 - a["ab"] / a["ms"])
        worst["dodge"] = max(worst["dodge"], abs(r["dodge"] - wk["dodge_ratio"]))
        r["rms_b"] = math.sqrt(a["bb"] / a["n"]); worst["rms_b_rel"] = max(worst["rms_b_rel"], abs(r["rms_b"] / wk["rms_b"] - 1))
        worst["cancel"] = max(worst["cancel"], abs(r["cancel_finished"] - wk["cancel_finished"]), abs(r["cancel_blank"] - wk["cancel_blank"]))
        if a["dd"] > 0:
            r["k"] = -2 * a["bd"] / (a["bb"] + a["dd"]); r["gamma"] = 1 + a["bd"] / a["bb"]
            worst["k"] = max(worst["k"], abs(r["k"] - wk["k"]))
            gd = abs(r["gamma"] - wk["gamma"]); gd = gd / abs(wk["gamma"]) if abs(wk["gamma"]) > 1 else gd
            worst["gamma"] = max(worst["gamma"], gd)
        rec[k] = r
    # 3. remove: float words and the blank alone's loss
    def top1(Wv):
        load(Wv)
        with torch.no_grad(): return hf(texts).logits[:, 8:, :].argmax(-1).numpy()
    def held_loss(Wv):
        load(Wv); tot = []
        with torch.no_grad():
            for i in range(0, RM["loss_windows"], 16):
                w_ = torch.from_numpy(win[i:i + 16]); lg = hf(w_[:, :-1]).logits
                tot.append(F.cross_entropy(lg.reshape(-1, VOC), w_[:, 1:].reshape(-1), reduction="none"))
        return float(torch.cat(tot).mean())
    fin = top1(W)
    Wp = {n: (P[n] if n in DRAWN else W[n]) for n in blank}; Wb = {n: (Bt[n] if n in DRAWN else W[n]) for n in blank}
    rm = RM["snaps"][str(t)]
    tp, tb = int((top1(Wp) == fin).sum()), int((top1(Wb) == fin).sum()); lb = held_loss(Wb); lp = held_loss(Wp) if t > 0 else None
    worst["top1"] = max(worst["top1"], abs(tp - rm["pushes"]["top1_as_finished"]), abs(tb - rm["blank"]["top1_as_finished"]))
    worst["blank_loss"] = max(worst["blank_loss"], abs(lb - rm["blank"]["loss"]))
    if lp is not None: worst["pushes_loss"] = max(worst.get("pushes_loss", 0.0), abs(lp - rm["pushes"]["loss"]))
    out[t] = dict(plates=rec, pushes_top1_as_finished=tp, blank_top1_as_finished=tb, blank_loss=lb, pushes_loss=lp)
    print(f"  step {t:5d}  " + "  ".join(f"{k} k {r['k']:+.3f}/{WK['snaps'][str(t)]['plates'][k]['k']:+.3f} g {r['gamma']:.3f}/{WK['snaps'][str(t)]['plates'][k]['gamma']:.3f}"
                                         for k, r in rec.items() if "k" in r)
          + f"  | P alone top1 {tp} (machine {rm['pushes']['top1_as_finished']})  blank alone top1 {tb} (machine {rm['blank']['top1_as_finished']}) loss {lb:.4f} (machine {rm['blank']['loss']:.4f})", flush=True)
json.dump(dict(worst=worst, snaps=out), open(os.path.join(HERE, "evals", f"workcheck_{nm}.json"), "w"), indent=1)
print("worst: " + "  ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in worst.items()))
