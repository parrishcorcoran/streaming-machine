"""INDEPENDENT REBUILD of keep_late.py's cases (rule 10), by a separate checker that shares no code with it.

It does not read the base-2 grid. It takes Pythia's own fp16 files at step0 (the start) and step143000 (finished) from
../models/, makes each case's plates in fp32 directly (W - s x W0, s per plate as stored by ../foundation_share.py),
and runs fp32 Pythia (transformers' GPTNeoX, eager arrival) on the same fed-in words. Gains, offsets and biases are the
finished ones in every case. The keep set is rebuilt here from its definition (the qkv plates of the last third of
the layers); the random keep sets are read from keep_late_<size>.json as data (they are draws, not code).
  fin  b  k  k_pl  rk@1..3   compared position by position with the base-2 case
  ck   the scrambled start with this checker's own draw (a different scramble), so only the share kept is compared

PREDICTION, written before this checker first ran:
  - fin: checker word = base-2 finished word at 99.5% or more of positions.
  - b, k, k_pl, rk@n: checker word = base-2 case word at 95% or more of positions; the share of finished words kept
    within 3 points of the base-2 run's.
  - ck (own draw): the share kept within 5 points of the base-2 ck draws' range.

  python3 keep_checker.py size=160m
"""
import os, sys, json, time, resource
import numpy as np, torch
from safetensors.numpy import load_file
from transformers import GPTNeoXForCausalLM

A = dict(a.split("=") for a in sys.argv[1:])
SIZE = A.get("size", "160m")
LATE = os.path.dirname(os.path.abspath(__file__)); HERE = os.path.dirname(LATE); MD = os.path.join(HERE, "models")
torch.set_grad_enabled(False)
B2 = json.load(open(os.path.join(LATE, f"keep_late_{SIZE}.json")))
SHARE = json.load(open(os.path.join(HERE, "foundation_share.json")))["sizes"][SIZE]["plates"]
refs = json.load(open(os.path.join(HERE, f"refs_pythia-{SIZE}.json")))[:B2["texts"]]
WORDS = B2["words"]
m = GPTNeoXForCausalLM.from_pretrained(os.path.join(MD, f"pythia-{SIZE}-step143000"), torch_dtype=torch.float32,
                                       attn_implementation="eager").eval()
NL = m.config.num_hidden_layers
F = load_file(os.path.join(MD, f"pythia-{SIZE}-step143000", "model.safetensors"))
S0 = load_file(os.path.join(MD, f"pythia-{SIZE}-step0", "model.safetensors"))
NAMES = {"embed": "gpt_neox.embed_in.weight", "head": "embed_out.weight"}
for l in range(NL):
    NAMES[f"qkv.{l}"] = f"gpt_neox.layers.{l}.attention.query_key_value.weight"
    NAMES[f"proj.{l}"] = f"gpt_neox.layers.{l}.attention.dense.weight"
    NAMES[f"fc.{l}"] = f"gpt_neox.layers.{l}.mlp.dense_h_to_4h.weight"
    NAMES[f"mproj.{l}"] = f"gpt_neox.layers.{l}.mlp.dense_4h_to_h.weight"
TABS = ("head", "embed")
LATE_QKV = {f"qkv.{l}" for l in range(NL) if 3 * l >= 2 * NL}          # the last third of the layers
params = dict(m.named_parameters())
LOGF = open(os.path.join(LATE, f"keep_checker_{SIZE}.log"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
log(f"keep_checker.py size {SIZE}: fp32 Pythia from its own fp16 files; late qkv rebuilt here {sorted(LATE_QKV)}; "
    f"base-2 keep set {B2['keep']}; same: {sorted(LATE_QKV) == sorted(B2['keep'])}")

def own_scramble(pn, key):
    w0 = S0[pn].ravel()
    return w0[np.random.default_rng(4242 + sum(map(ord, key))).permutation(w0.size)].reshape(S0[pn].shape)

def untouched(case, name):
    if case == "fin": return True
    if case == "b": return False
    if case in ("k", "ck"): return name in LATE_QKV
    if case == "k_pl": return name in LATE_QKV or name in TABS
    if case.startswith("rk@"): return name in set(B2["rk"][case[3:]])
    raise ValueError(case)

def set_case(case):
    for name, pn in NAMES.items():
        w = F[pn].astype(np.float32)
        if not untouched(case, name):
            w0 = (own_scramble(pn, name) if case == "ck" else S0[pn]).astype(np.float32)
            w = w - np.float32(SHARE[name]["s"]) * w0
        params[pn].data.copy_(torch.from_numpy(w))

cases = ["fin"] + [c for c in ("b", "k", "k_pl", "rk@1", "rk@2", "rk@3") if c in B2["cases"]] + ["ck"]
FIN = None
for case in cases:
    t0 = time.time(); set_case(case); top1 = []
    for r in refs:
        seq = r["ids"] + r["ref"][:WORDS - 1]
        lg = m(torch.tensor([seq])).logits[0].double()
        top1.append(lg.topk(1).indices[:, 0].tolist())
    if case == "fin": FIN = top1
    n = sum(len(x) for x in top1)
    kept = sum(a == b for x, y in zip(top1, FIN) for a, b in zip(x, y))
    if case == "ck":
        cr = [B2["cases"][c]["top1_all"] for c in B2["cases"] if c.startswith("ck@")]
        log(f"  {case:6s} (own scramble) finished words kept: checker {kept / n * 100:5.1f}%  base-2 ck draws "
            f"{min(cr) * 100:5.1f}..{max(cr) * 100:5.1f}%  ({time.time() - t0:.1f}s)")
        continue
    b2 = B2["cases"][case]["top1"]
    same_b2 = sum(a == b for x, y in zip(top1, b2) for a, b in zip(x, y))
    log(f"  {case:6s} checker word = base-2 case word at {same_b2}/{n} ({same_b2 / n * 100:5.1f}%);  "
        f"finished words kept: checker {kept / n * 100:5.1f}%  base-2 {B2['cases'][case]['top1_all'] * 100:5.1f}%  ({time.time() - t0:.1f}s)")
log(f"peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30:.2f} GB")
