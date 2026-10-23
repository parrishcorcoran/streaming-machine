"""The even part of every set of weights that spans the stream's lanes: its sum over those lanes, which is what an
even light (every lane equally bright) meets. Start (step0) against finished (step143000) of the SAME row, on the
same grid as balance_count.py (one power of two per plate, shared by start and finished, 16 places).

  writes into the stream: input word table per word, attention-out per line, MLP-out per line
  reads from the stream:  query/key/value per neuron, MLP-in per neuron, output word table per word

For each plate: agreement (correlation) of the finished even part with the start's even part of the same row; the
ratio finished/start (sum of start x finished over sum of start^2); the root-mean-square of the even part in the
start row's coin-toss width (sqrt of the start row's sum of squares) at start and at finished, and in the finished
row's own width at finished. Shuffle control: the finished row against a different row's start. Order-free check:
rows of both plates permuted together give the same agreement. Prediction 6 in PREDICTION.txt, written first.

    python3 balance_even_part.py [sizes=70m,160m]
"""
import os, sys, json, math, numpy as np
from safetensors import safe_open

A = dict(a.split("=") for a in sys.argv[1:])
SIZES = A.get("sizes", "70m,160m").split(",")
HERE = os.path.dirname(os.path.abspath(__file__)); MD = os.path.join(os.path.dirname(HERE), "models")
STEPS = ("step0", "step143000")


def plates(L):                     # name, key, axis summed over (the stream's lanes), write or read
    for l in range(L):
        p = f"gpt_neox.layers.{l}."
        yield f"qkv.{l}", p + "attention.query_key_value.weight", 1, "read"
        yield f"attn_out.{l}", p + "attention.dense.weight", 0, "write"
        yield f"mlp_in.{l}", p + "mlp.dense_h_to_4h.weight", 1, "read"
        yield f"mlp_out.{l}", p + "mlp.dense_4h_to_h.weight", 0, "write"
    yield "words_in", "gpt_neox.embed_in.weight", 1, "write"
    yield "words_out", "embed_out.weight", 1, "read"


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))


out = {}
for SIZE in SIZES:
    cfg = json.load(open(os.path.join(MD, f"pythia-{SIZE}-step143000", "config.json")))
    F = {st: safe_open(os.path.join(MD, f"pythia-{SIZE}-{st}", "model.safetensors"), "np") for st in STEPS}
    ref = {r["plate"]: r for r in json.load(open(os.path.join(HERE, f"balance_{SIZE}.json")))}
    rows = []
    print(f"\npythia-{SIZE}: even part of each row over the stream's lanes (agreement = correlation, start row vs finished row)")
    print(f"{'plate':11s} {'kind':5s} {'rows':>6s} | agree  shuffle  order-free | ratio fin/start | rms in start widths: start  finished | rms in own width: finished")
    rng = np.random.default_rng(6)
    for name, key, ax, kind in plates(cfg["num_hidden_layers"]):
        top = max(float(np.abs(F[st].get_tensor(key)).max()) for st in STEPS)
        e = math.ceil(math.log2(top / 32767)); assert e == ref[name]["power"]
        s, w = {}, {}
        for st in STEPS:
            q = np.round(F[st].get_tensor(key).astype(np.float32) / np.float32(2.0 ** e)).astype(np.int16)
            s[st] = q.sum(axis=ax, dtype=np.int64).astype(np.float64)
            q2 = q.astype(np.int32); q2 *= q2
            w[st] = np.sqrt(q2.sum(axis=ax, dtype=np.int64).astype(np.float64))
            if st == "step143000":          # order-free: rows permuted (and lanes permuted) give the same row sums, moved
                pr = rng.permutation(q.shape[1 - ax]); pl = rng.permutation(q.shape[ax])
                qp = q[pr][:, pl] if ax == 1 else q[pl][:, pr]
                assert np.array_equal(qp.sum(axis=ax, dtype=np.int64), s[st][pr].astype(np.int64))
                del qp
            del q, q2
        s0, s1, w0, w1 = s["step0"], s["step143000"], w["step0"], w["step143000"]
        perm = rng.permutation(len(s0))
        r = dict(plate=name, kind=kind, rows=len(s0), agree=corr(s0, s1), shuffle=corr(s0[perm], s1),
                 ratio=float((s0 * s1).sum() / (s0 * s0).sum()),
                 rms_start=float(np.sqrt(np.mean((s0 / w0) ** 2))), rms_fin_startwidth=float(np.sqrt(np.mean((s1 / w0) ** 2))),
                 rms_fin_own=float(np.sqrt(np.mean((s1 / w1) ** 2))), growth=float(np.sqrt(np.mean(w1 ** 2) / np.mean(w0 ** 2))))
        rows.append(r)
        print(f"{name:11s} {kind:5s} {r['rows']:6d} | {r['agree']:+.3f}  {r['shuffle']:+.3f}   same      | {r['ratio']:+.3f}          "
              f"|                     {r['rms_start']:.2f}   {r['rms_fin_startwidth']:6.2f}   |                  {r['rms_fin_own']:.2f}"
              f"   (rows grew x{r['growth']:.2f})", flush=True)
    out[SIZE] = rows
json.dump(out, open(os.path.join(HERE, "even_part.json"), "w"), indent=0)
