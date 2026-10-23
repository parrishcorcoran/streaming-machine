"""INDEPENDENT REBUILD of remove_foundation.py's cases (rule 10), by a separate checker that shares no code with it.

It does not read the base-2 grid (base2_pythia-<size>.npz) for the plates. It takes Pythia's own fp16 files at step0
(the start) and step143000 (finished) from models/, makes each case's plates in fp32 directly, and runs fp32 Pythia
(transformers' GPTNeoX, eager arrival) on the same fed-in words. s per plate is the one stored by foundation_share.py.
Its shuffle control scrambles the fp16 start values within each plate with its own draw (a different scramble from the
base-2 run's), so for ca/cb only the share of words kept is compared, not position by position.
  fin  a (W - W0)  b (W - s W0)  ca cb (scrambled W0, own draw)  a_tab b_tab  a_pl b_pl  a_L<l> b_L<l>
Gains, norm offsets and biases are the finished ones in every case.

PREDICTION, written before this checker first ran:
  - fin: the checker's word = the base-2 finished machine's word at 99.5% or more of all positions.
  - every case without a scramble: the checker's word = the base-2 case's word at 95% or more of positions in case a,
    97% or more in the others; the share of the finished machine's words kept within 3 points of the base-2 run's.
  - ca, cb (another scramble): the share kept within 5 points of the base-2 run's.

ADDED after the first checker runs (written before the src=remove_<size>-extra.json runs): the one-table cases
  b_head, b_emb, a_head, a_emb: the checker's word = the base-2 case's word at 97% or more of positions, the share kept
  within 3 points (70m b_emb: every word, as in the base-2 run).

  python3 checker_remove.py size=70m [src=remove_70m-extra.json]
"""
import os, sys, json, time, resource
import numpy as np, torch
from safetensors.numpy import load_file
from transformers import GPTNeoXForCausalLM

A = dict(a.split("=") for a in sys.argv[1:])
SIZE = A.get("size", "70m")
HERE = os.path.dirname(os.path.abspath(__file__)); MD = os.path.join(HERE, "models")
torch.set_grad_enabled(False)
SRC = A.get("src", f"remove_{SIZE}.json")
B2 = json.load(open(os.path.join(HERE, SRC)))
SHARE = json.load(open(os.path.join(HERE, "foundation_share.json")))["sizes"][SIZE]["plates"]
refs = json.load(open(os.path.join(HERE, f"refs_pythia-{SIZE}.json")))[:B2["texts"]]
WORDS = B2["words"]
m = GPTNeoXForCausalLM.from_pretrained(os.path.join(MD, f"pythia-{SIZE}-step143000"), torch_dtype=torch.float32,
                                       attn_implementation="eager").eval()
L = m.config.num_hidden_layers
F = load_file(os.path.join(MD, f"pythia-{SIZE}-step143000", "model.safetensors"))
S = load_file(os.path.join(MD, f"pythia-{SIZE}-step0", "model.safetensors"))
NAMES = {"embed": "gpt_neox.embed_in.weight", "head": "embed_out.weight"}
for l in range(L):
    NAMES[f"qkv.{l}"] = f"gpt_neox.layers.{l}.attention.query_key_value.weight"
    NAMES[f"proj.{l}"] = f"gpt_neox.layers.{l}.attention.dense.weight"
    NAMES[f"fc.{l}"] = f"gpt_neox.layers.{l}.mlp.dense_h_to_4h.weight"
    NAMES[f"mproj.{l}"] = f"gpt_neox.layers.{l}.mlp.dense_4h_to_h.weight"
TABLES = ["head", "embed"]; PLATES = [k for k in NAMES if k not in TABLES]
params = dict(m.named_parameters())
LOGF = open(os.path.join(HERE, f"checker_remove_{SIZE}{'-extra' if 'src' in A else ''}.log"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()

def scrambled(name):
    w0 = S[NAMES[name]].ravel()
    return w0[np.random.default_rng(777 + len(name) * 1000 + sum(map(ord, name))).permutation(w0.size)].reshape(S[NAMES[name]].shape)

def touched(case, name):
    kind, _, part = case.partition("_")
    if case == "fin": return False
    if part == "": return True
    if part == "tab": return name in TABLES
    if part == "pl": return name in PLATES
    if part == "head": return name == "head"
    if part == "emb": return name == "embed"
    return name in PLATES and name.split(".")[1] == part[1:]

def set_case(case):
    kind = case.partition("_")[0]
    for name, pn in NAMES.items():
        w = F[pn].astype(np.float32)
        if touched(case, name):
            w0 = (S[pn] if kind in ("a", "b") else scrambled(name)).astype(np.float32)
            w = w - w0 if kind in ("a", "ca") else w - np.float32(SHARE[name]["s"]) * w0
        params[pn].data.copy_(torch.from_numpy(w))

cases = [c for c in B2["cases"] if c in ("fin", "a", "b", "ca", "cb", "a_tab", "b_tab", "a_pl", "b_pl",
         "a_head", "b_head", "a_emb", "b_emb") or c.startswith(("a_L", "b_L"))]
log(f"checker_remove.py size {SIZE}: fp32 Pythia from its own fp16 files, {len(refs)} texts, cases {len(cases)}")
FIN = None
for case in cases:
    t0 = time.time(); set_case(case); top1 = []
    for r in refs:
        seq = r["ids"] + r["ref"][:WORDS - 1]
        lg = m(torch.tensor([seq])).logits[0].double()
        top1.append(lg.topk(1).indices[:, 0].tolist())
    if case == "fin": FIN = top1
    b2 = B2["cases"][case]["top1"]; b2fin = B2["cases"]["fin"]["top1"]
    n = sum(len(x) for x in top1)
    same_b2 = sum(a == b for x, y in zip(top1, b2) for a, b in zip(x, y))
    kept = sum(a == b for x, y in zip(top1, FIN) for a, b in zip(x, y))
    log(f"  {case:8s} checker word = base-2 case word at {same_b2}/{n} ({same_b2 / n * 100:5.1f}%);  "
        f"finished words kept: checker {kept / n * 100:5.1f}%  base-2 {B2['cases'][case]['top1_all'] * 100:5.1f}%  ({time.time() - t0:.1f}s)")
log(f"peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30:.2f} GB")
