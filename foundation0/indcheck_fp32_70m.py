"""SEPARATE CHECKER: the fp32 reference the whole-number forward (indcheck_forward70.py) is checked against FIRST.
2026-10-06. Not a machine of the work; a yardstick only.

fp32 Pythia-70m (transformers' GPTNeoX, eager arrival) on the first 10 texts of refs_pythia-70m.json, each text's own
words then fp32's 16 greedy words fed in (positions as remove_foundation.py counts them). Cases:
  fin  the finished plates;  a  W - W0 on every plate and both word tables;  b  W - s W0, s per plate (this checker's
  own exact s from indcheck_weights70.json). Gains, norm offsets and biases finished in every case.
Writes indcheck_fp32_70m.json: top-1 and top-5 at every position.

PREDICTION (written before it ran): fin's top-1 equals refs_pythia-70m.json's greedy words at all 160 continuation
positions (10 texts x 16). a and b are recorded, not judged here.
"""
import os, json
import numpy as np, torch
from safetensors import safe_open
from transformers import GPTNeoXForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__)); MD = os.path.join(HERE, "models")
torch.set_grad_enabled(False)
refs = json.load(open(os.path.join(HERE, "refs_pythia-70m.json")))[:10]
S = {k: v["s"] for k, v in json.load(open(os.path.join(HERE, "indcheck_weights70.json")))["plates"].items()}
m = GPTNeoXForCausalLM.from_pretrained(os.path.join(MD, "pythia-70m-step143000"), torch_dtype=torch.float32,
                                       attn_implementation="eager").eval()
f1 = safe_open(os.path.join(MD, "pythia-70m-step143000", "model.safetensors"), "np")
f0 = safe_open(os.path.join(MD, "pythia-70m-step0", "model.safetensors"), "np")
NAMES = {"embed": "gpt_neox.embed_in.weight", "head": "embed_out.weight"}
for l in range(6):
    p = f"gpt_neox.layers.{l}."
    NAMES[f"qkv.{l}"] = p + "attention.query_key_value.weight"; NAMES[f"proj.{l}"] = p + "attention.dense.weight"
    NAMES[f"fc.{l}"] = p + "mlp.dense_h_to_4h.weight"; NAMES[f"mproj.{l}"] = p + "mlp.dense_4h_to_h.weight"
params = dict(m.named_parameters())
OUT = {}
for case in ("fin", "a", "b"):
    for name, key in NAMES.items():
        w = f1.get_tensor(key).astype(np.float64)
        if case != "fin":
            w0 = f0.get_tensor(key).astype(np.float64)
            w = w - w0 if case == "a" else w - S[name] * w0
        params[key].data.copy_(torch.from_numpy(w.astype(np.float32)))
    t1, t5 = [], []
    for r in refs:
        seq = r["ids"] + r["ref"][:15]
        lg = m(torch.tensor([seq])).logits[0].double().numpy()
        order = np.lexsort((np.arange(lg.shape[1])[None, :].repeat(lg.shape[0], 0), -lg), axis=1)[:, :5]
        t1.append(order[:, 0].tolist()); t5.append(order.tolist())
    OUT[case] = dict(top1=t1, top5=t5)
    if case == "fin":
        ok = sum(int(a == b) for x, r in zip(t1, refs) for a, b in zip(x[len(r["ids"]) - 1:len(r["ids"]) + 15], r["ref"]))
        print(f"fin: fp32 top-1 = refs greedy words at {ok} of {16 * len(refs)} continuation positions")
    print(case, "done")
json.dump(OUT, open(os.path.join(HERE, "indcheck_fp32_70m.json"), "w"))
