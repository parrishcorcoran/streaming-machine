"""Independent rebuild of balance_even_part.py (a separate checker): torch loader, the power found by stepping, each
row's even part as the plate met by an even light (a ones vector, whole numbers), agreement by torch.corrcoef.
Compares agreement, ratio and the three root-mean-squares with even_part.json.

    python3 balance_even_checker.py
"""
import os, json, torch
from safetensors.torch import safe_open
HERE = os.path.dirname(os.path.abspath(__file__)); MD = os.path.join(os.path.dirname(HERE), "models")
ref = json.load(open(os.path.join(HERE, "even_part.json")))
torch.set_grad_enabled(False); bad = 0; n = 0
KEY = {"qkv": ("attention.query_key_value.weight", 1), "attn_out": ("attention.dense.weight", 0),
       "mlp_in": ("mlp.dense_h_to_4h.weight", 1), "mlp_out": ("mlp.dense_4h_to_h.weight", 0)}
for size, rows in ref.items():
    F = {st: safe_open(os.path.join(MD, f"pythia-{size}-{st}", "model.safetensors"), "pt") for st in ("step0", "step143000")}
    for r in rows:
        nm = r["plate"]
        if nm == "words_in": key, ax = "gpt_neox.embed_in.weight", 1
        elif nm == "words_out": key, ax = "embed_out.weight", 1
        else: k, l = nm.split("."); key, ax = f"gpt_neox.layers.{l}." + KEY[k][0], KEY[k][1]
        top = max(float(F[s].get_tensor(key).float().abs().max()) for s in F)
        e = 0
        while top / 2.0 ** e > 32767: e += 1
        while top / 2.0 ** (e - 1) <= 32767: e -= 1
        S, W = {}, {}
        for s in F:
            q = torch.round(F[s].get_tensor(key).float() / 2.0 ** e).to(torch.int64)
            if ax == 0: q = q.T
            S[s] = (q @ torch.ones(q.shape[1], dtype=torch.int64)).double()          # an even light on every lane
            W[s] = (q * q).sum(1).double().sqrt(); del q
        s0, s1, w0, w1 = S["step0"], S["step143000"], W["step0"], W["step143000"]
        mine = dict(agree=float(torch.corrcoef(torch.stack([s0, s1]))[0, 1]), ratio=float((s0 * s1).sum() / (s0 * s0).sum()),
                    rms_start=float(((s0 / w0) ** 2).mean().sqrt()), rms_fin_startwidth=float(((s1 / w0) ** 2).mean().sqrt()),
                    rms_fin_own=float(((s1 / w1) ** 2).mean().sqrt()))
        n += 1
        diff = max(abs(mine[k] - r[k]) for k in mine)
        if diff > 1e-9: bad += 1; print("MISMATCH", size, nm, mine, {k: r[k] for k in mine})
    print(size, "done", flush=True)
print(f"{n} plates checked, {bad} mismatches")
