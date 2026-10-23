"""Convert Pythia to the all-base-2 machine. The only place a float exists: it reads Pythia's fp16 plates once
(safetensors, one plate at a time) and writes every number as a whole number with a power-of-two scale.

  THE START AND THE FINISHED PLATE SHARE ONE POWER OF TWO PER NEURON. For every plate the power is the smallest that
  holds the larger of the two plates' largest values in 16 places (|q| <= 32767), so both are whole numbers on the
  same grid and   finished = start + built   holds digit for digit (built = finished - start, whole numbers, kept
  implicitly as .w - .s). Any amount of the start can be taken out in whole numbers.

  Pythia's parts as they are, nothing folded across them:
  - plates [in][out], 16 places, one power of two per output neuron (per column):
        qkv.l   query_key_value   (per head: q, k, v, head_size each, Pythia's own order)
        proj.l  attention.dense
        fc.l    mlp.dense_h_to_4h
        mproj.l mlp.dense_4h_to_h
        head    embed_out [hidden][words]: one power of two per word (the machine reads a word's column)
  - embed   embed_in [words][hidden]: one power of two per word (the machine reads a word's row)
  - biases and the norms' offsets: whole numbers at the stream's scale 2^-16 (all 0 at the start)
  - the norms' gains: 16 places, one power of two per gain vector shared by start and finished (all 1 at the start).
    The gain is NOT folded into the plate after it: folding would mix the finished gain into the start, and the
    plate would no longer be the plate the start was drawn on.
  - tables: the share past the knife edge Phi(x) (erf form, as base2_convert.py builds PHI); the arrival e^-d; log2
    and 2^x of a 16-digit fine part; the turns cos, sin at 2^-30 for every position (rotary on the first quarter of
    each head, pairs (i, i + quarter/2)); the norm's eps.
  Keys: X.w finished, X.s start, X.e power of two; X.b finished bias, X.bs start bias; lnN.g / .gs / .ge / .b / .bs.

  also=step1000,step10000,...: steps between, written as X.<step> (int32) on the SAME per-neuron power of two as the
  start and finished (the power is set by those two only, so the machine's grid does not change); counts of values
  that pass 16 places are reported.

    python3 pythia_convert.py size=14m|70m|160m [start=step0] [finished=step143000] [also=step1000,step10000,step50000]
"""
import os, sys, json, math, numpy as np
from safetensors import safe_open

A = dict(a.split("=") for a in sys.argv[1:])
SIZE, START, FIN = A.get("size", "14m"), A.get("start", "step0"), A.get("finished", "step143000")
ALSO = [x for x in A.get("also", "").split(",") if x]
HERE = os.path.dirname(os.path.abspath(__file__)); MD = os.path.join(HERE, "models")
cfg = json.load(open(os.path.join(MD, f"pythia-{SIZE}-{FIN}", "config.json")))
d, L, NH = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"]; HS = d // NH; ROT = int(HS * cfg["rotary_pct"])
assert cfg["use_parallel_residual"] and cfg["hidden_act"] == "gelu" and not cfg["tie_word_embeddings"]
ES = 16
f1 = safe_open(os.path.join(MD, f"pythia-{SIZE}-{FIN}", "model.safetensors"), "np")
f0 = safe_open(os.path.join(MD, f"pythia-{SIZE}-{START}", "model.safetensors"), "np")
fx = {st: safe_open(os.path.join(MD, f"pythia-{SIZE}-{st}", "model.safetensors"), "np") for st in ALSO}
over16 = {st: 0 for st in ALSO}
def get(f, k): return f.get_tensor("gpt_neox." + k if not k.startswith("embed_out") else k).astype(np.float64)

out, rep = {}, []
def shared16(Wf, Ws, axis, name):
    """16 places, one power of two per neuron along `axis` (0: per column, 1: per row), shared by start and finished"""
    mf, ms = np.abs(Wf).max(axis), np.abs(Ws).max(axis)
    e = np.ceil(np.log2(np.maximum(np.maximum(mf, ms), 1e-30) / 32767)).astype(np.int64)
    eown = np.ceil(np.log2(np.maximum(mf, 1e-30) / 32767)).astype(np.int64)
    sh = np.expand_dims(e, axis)
    qf, qs = np.round(Wf / 2.0 ** sh).astype(np.int64), np.round(Ws / 2.0 ** sh).astype(np.int64)
    assert np.abs(qf).max() <= 32767 and np.abs(qs).max() <= 32767
    rep.append(dict(plate=name, neurons=int(e.size), start_sets_power=int((e > eown).sum()),
                    finished_off_grid=int((qf * 2.0 ** sh != Wf).sum()), start_off_grid=int((qs * 2.0 ** sh != Ws).sum()), values=int(Wf.size)))
    return qf.astype(np.int16), qs.astype(np.int16), e

def stream(v): return np.round(v * 2.0 ** ES).astype(np.int64)

def plate(name, key, bias=True):                       # torch Linear [out, in] -> [in][out]
    Wf, Ws = get(f1, key + ".weight").T, get(f0, key + ".weight").T
    out[name + ".w"], out[name + ".s"], out[name + ".e"] = shared16(Wf, Ws, 0, name)
    if bias: out[name + ".b"], out[name + ".bs"] = stream(get(f1, key + ".bias")), stream(get(f0, key + ".bias"))
    for st, f in fx.items():                           # a step between, on the same grid
        q = np.round(get(f, key + ".weight").T / 2.0 ** out[name + ".e"][None, :]).astype(np.int64)
        over16[st] += int((np.abs(q) > 32767).sum()); out[f"{name}.{st}"] = q.astype(np.int32)
        if bias: out[f"{name}.b.{st}"] = stream(get(f, key + ".bias"))
    if fx: del out[name + ".w"], out[name + ".s"]       # steps-only file: the grid's powers are all it keeps of these (peak under 3 GB)

def norm(name, key):
    gf, gs = get(f1, key + ".weight"), get(f0, key + ".weight")
    ge = int(np.ceil(np.log2(max(np.abs(gf).max(), np.abs(gs).max()) / 32767)))
    out[name + ".g"], out[name + ".gs"], out[name + ".ge"] = np.round(gf / 2.0 ** ge).astype(np.int64), np.round(gs / 2.0 ** ge).astype(np.int64), np.array(ge)
    out[name + ".b"], out[name + ".bs"] = stream(get(f1, key + ".bias")), stream(get(f0, key + ".bias"))
    for st, f in fx.items():
        out[f"{name}.g.{st}"], out[f"{name}.b.{st}"] = np.round(get(f, key + ".weight") / 2.0 ** ge).astype(np.int64), stream(get(f, key + ".bias"))

for l in range(L):
    p = f"layers.{l}."
    norm(f"ln1.{l}", p + "input_layernorm"); norm(f"ln2.{l}", p + "post_attention_layernorm")
    plate(f"qkv.{l}", p + "attention.query_key_value"); plate(f"proj.{l}", p + "attention.dense")
    plate(f"fc.{l}", p + "mlp.dense_h_to_4h"); plate(f"mproj.{l}", p + "mlp.dense_4h_to_h")
norm("lnf", "final_layer_norm")
plate("head", "embed_out", bias=False)
Ef, Es = get(f1, "embed_in.weight"), get(f0, "embed_in.weight")
out["embed.w"], out["embed.s"], out["embed.e"] = shared16(Ef, Es, 1, "embed"); del Ef, Es
for st, f in fx.items():
    q = np.round(get(f, "embed_in.weight") / 2.0 ** out["embed.e"][:, None]).astype(np.int64)
    over16[st] += int((np.abs(q) > 32767).sum()); out[f"embed.{st}"] = q.astype(np.int32); del q
if fx: del out["embed.w"], out["embed.s"]

# tables
out["phi"] = np.array([round(0.5 * (1 + math.erf(((i - 32768) / 4096) / math.sqrt(2))) * 65536) for i in range(65536)], dtype=np.int64)
SC = 10; arr = []; t = 0
while True:
    a = round(math.exp(-t / 2 ** SC) * 2 ** 24)
    if a == 0: break
    arr.append(a); t += 1
out["arrival"] = np.array(arr, dtype=np.int64)
out["log2"] = np.array([round(math.log2(1 + m / 65536) * 65536) for m in range(65536)], dtype=np.int64)
out["exp2"] = np.array([round(2 ** (f / 65536) * 65536) for f in range(65536)], dtype=np.int64)
inv = float(cfg["rotary_emb_base"]) ** (-np.arange(0, ROT, 2) / ROT); ang = np.arange(cfg["max_position_embeddings"])[:, None] * inv[None, :]
out["cos"], out["sin"] = np.round(np.cos(ang) * 2 ** 30).astype(np.int64), np.round(np.sin(ang) * 2 ** 30).astype(np.int64)
out["eps40"] = np.array(round(cfg["layer_norm_eps"] * d * d * 2 ** 40))
out["half_log2_hs"] = np.array(round(math.log2(HS) / 2 * 65536)); assert out["half_log2_hs"] * 2 == math.log2(HS) * 65536
for k, v in dict(d=d, L=L, heads=NH, hs=HS, rot=ROT, vocab=cfg["vocab_size"]).items(): out[k] = np.array(v)
dst = os.path.join(HERE, f"base2_pythia-{SIZE}.npz" if not ALSO else f"base2_pythia-{SIZE}-steps.npz")
if ALSO: out = {k: v for k, v in out.items() if any(k.endswith("." + st) for st in ALSO) or k.endswith(".e") or k.endswith(".ge")}
np.savez(dst, **out)
if ALSO:
    print(f"wrote {dst}: steps {ALSO} on the shared grid (int32); values past 16 places: {over16}"); sys.exit()
json.dump(rep, open(os.path.join(HERE, f"convert_pythia-{SIZE}.json"), "w"), indent=0)
nw = sum(r["values"] for r in rep); nn = sum(r["neurons"] for r in rep); ss = sum(r["start_sets_power"] for r in rep)
print(f"wrote {dst}: {nw / 1e6:.2f}M weights x2 (finished, start) on one power of two per neuron; d {d}, layers {L}, heads {NH}, "
      f"head size {HS}, turned {ROT}; arrival table {len(arr):,}")
print(f"  neurons whose power is set by the start (the finished plate alone would sit one place finer): {ss:,} of {nn:,}")
print(f"  values not exactly on the grid: finished {sum(r['finished_off_grid'] for r in rep):,}, start {sum(r['start_off_grid'] for r in rep):,} of {nw:,}")
for r in rep:
    if r["start_sets_power"]: print(f"    {r['plate']:10s} start sets the power on {r['start_sets_power']:,} of {r['neurons']:,} neurons")
