"""Write each snapshot as a whole-number machine (as ../base2_convert.py and ../foundation0/pythia_convert.py do).
    python3 convert.py width=64 layers=2 recipe=A

The only place a float exists on the machine side: it reads the snapshot's whole numbers at 2^-30 (snapio.snapshots,
checked against the build's hashes) and writes every number as a whole number with a power-of-two scale.

  THE BLANK AND THE FINISHED PLATE SHARE ONE POWER OF TWO PER NEURON: the smallest that holds the larger of the two
  plates' largest values in 16 places (|q| <= 32767). So   finished = blank + built   holds digit for digit
  (built = .w - .s). The parts as they are, nothing folded across them (folding the gain into the next plate would mix
  the finished gain into the blank):
  - plates [in][out], 16 places, one power of two per output neuron (column): qkv.l (q all heads, k, v, GPT-2's order),
    proj.l, fc.l, mproj.l; head = the output word table [lanes][words], one power of two per word
  - wte (input word table) and wpe (position rows): 16 places, one power of two per row
  - biases and the water levels' offsets: whole numbers at the stream's scale 2^-16 (the blank's are all 0)
  - the water levels' gains: 16 places, one power of two per gain vector, shared with the blank's gain (exactly 1)
  - tables (tables.npz, written once): the share past the knife edge for GPT-2's own bend, the tanh form
    0.5 (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))) at 65,536 places (x = (i - 32768) / 4096); the arrival e^-d at
    every distance of the score's scale 2^-10 until it rounds to 0 (at 2^-24); log2 and 2^x of a 16-digit fine part
  Keys: X.w finished, X.s blank, X.e power; X.b, X.bs; lnN.g, .gs, .ge, .b, .bs; wte.*, wpe.*, head.*; eps40.

PREDICTIONS (the session's, written before any run of this script):
  C1. Every value fits 16 places on the shared grid (asserted), every snapshot.
  C2. Neurons whose power is set by the blank (the finished plate alone would sit one place finer): 0 at step 0
      (finished = blank); under 15% of neurons at every later snapshot of every build.
"""
import os, sys, math, json, numpy as np
from model import HERE, GRID, V, T, HS, layout, tag
from snapio import snapshots

A = dict(a.split("=") for a in sys.argv[1:])
d, L, RECIPE = int(A.get("width", 64)), int(A.get("layers", 2)), A.get("recipe", "A")
ES = 16
OUT = os.path.join(HERE, "machines"); os.makedirs(OUT, exist_ok=True)

TAB = os.path.join(OUT, "tables.npz")
if not os.path.exists(TAB):
    c = math.sqrt(2 / math.pi)
    share = np.array([round(0.5 * (1 + math.tanh(c * (x + 0.044715 * x ** 3))) * 65536) for x in ((i - 32768) / 4096 for i in range(65536))], dtype=np.int64)
    SC, arr, t = 10, [], 0
    while True:
        a = round(math.exp(-t / 2 ** SC) * 2 ** 24)
        if a == 0: break
        arr.append(a); t += 1
    np.savez(TAB, share=share, arrival=np.array(arr, dtype=np.int64),
             log2=np.array([round(math.log2(1 + m / 65536) * 65536) for m in range(65536)], dtype=np.int64),
             exp2=np.array([round(2 ** (f / 65536) * 65536) for f in range(65536)], dtype=np.int64))

def shared16(Wf, Ws, axis):
    """16 places, one power of two per neuron (axis 0: per column, 1: per row), shared by the blank and the finished.
    Wf, Ws are whole numbers at 2^-30; the power returned is absolute (value = q x 2^e)."""
    mx = np.maximum(np.abs(Wf).max(axis), np.abs(Ws).max(axis)).astype(np.float64)
    mf = np.abs(Wf).max(axis).astype(np.float64)
    s = np.ceil(np.log2(np.maximum(mx, 1) / 32767)).astype(np.int64)          # shift from the 2^-30 grid
    sown = np.ceil(np.log2(np.maximum(mf, 1) / 32767)).astype(np.int64)
    sh = np.expand_dims(s, axis)
    qf, qs = np.rint(Wf / 2.0 ** sh).astype(np.int64), np.rint(Ws / 2.0 ** sh).astype(np.int64)
    assert np.abs(qf).max() <= 32767 and np.abs(qs).max() <= 32767
    return qf.astype(np.int16), qs.astype(np.int16), s - GRID, int((s > sown).sum()), int(s.size)

def stream(n): return np.rint(n / 2.0 ** (GRID - ES)).astype(np.int64)

rep = {}
for t, Bt, P, W in snapshots(d, L, RECIPE):
    out, blank_sets, neurons = {}, 0, 0
    def put(key, Wf, Ws, axis):
        global blank_sets, neurons
        out[key + ".w"], out[key + ".s"], out[key + ".e"], b, n = shared16(Wf, Ws, axis); blank_sets += b; neurons += n
    for l in range(L):
        for nm in ("qkv", "proj", "fc", "mproj"):
            put(f"{nm}.{l}", W[f"{nm}.{l}.w"], Bt[f"{nm}.{l}.w"], 0)
            out[f"{nm}.{l}.b"], out[f"{nm}.{l}.bs"] = stream(W[f"{nm}.{l}.b"]), stream(Bt[f"{nm}.{l}.b"])
    for nm in [f"ln1.{l}" for l in range(L)] + [f"ln2.{l}" for l in range(L)] + ["lnf"]:
        gf, gs = W[nm + ".g"], Bt[nm + ".g"]
        sg = int(math.ceil(math.log2(max(int(np.abs(gf).max()), int(np.abs(gs).max())) / 32767)))
        out[nm + ".g"], out[nm + ".gs"] = np.rint(gf / 2.0 ** sg).astype(np.int64), np.rint(gs / 2.0 ** sg).astype(np.int64)
        assert max(np.abs(out[nm + ".g"]).max(), np.abs(out[nm + ".gs"]).max()) <= 32767
        out[nm + ".ge"] = np.array(sg - GRID)
        out[nm + ".b"], out[nm + ".bs"] = stream(W[nm + ".b"]), stream(Bt[nm + ".b"])
    put("wte", W["wte"], Bt["wte"], 1); put("wpe", W["wpe"], Bt["wpe"], 1); put("head", W["head"], Bt["head"], 0)
    out["eps40"] = np.array(round(1e-5 * d * d * 2 ** 40)); out["half_log2_hs"] = np.array(round(math.log2(HS) / 2 * 65536))
    for k, v in dict(d=d, L=L, heads=d // HS, hs=HS, vocab=V, ctx=T, step=t).items(): out[k] = np.array(v)
    np.savez_compressed(os.path.join(OUT, f"{tag(d, L)}_{RECIPE}_step{t}.npz"), **out)
    rep[t] = dict(blank_sets_power=blank_sets, neurons=neurons)
    print(f"  step {t:5d}: written; neurons whose power the blank sets {blank_sets:,} of {neurons:,} ({100 * blank_sets / neurons:.1f}%)", flush=True)
json.dump(rep, open(os.path.join(OUT, f"convert_{tag(d, L)}_{RECIPE}.json"), "w"), indent=1)
