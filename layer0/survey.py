"""LAYER 0, STEP 2: THE SURVEY. Which water-level shifts occur at layer 0's plates, on the texts we have.

At qkv.0 the shift s is fixed by (word, position) (entry/survey_s.py covered all 51,463,168 pairs). Past the arrival
it is not: the light at fc.0 is R1 = word + position + the arrival's write, and the arrival's write depends on the
earlier tokens. So the shifts at proj.0 (the arrival's output o), fc.0 (R1) and mproj.0 (the bend's output) can only be
surveyed on texts. This runs base2.py's own layer 0 (its functions, through judge5) on the 30 refs texts (ids + ref,
648 positions) and the 300 judge windows (7,200 positions) and records, per position:
  s    qkv.0's shift (fixed by the pair)        so  proj.0's shift (o's water level)
  s1   fc.0's shift (R1's water level)          sm  mproj.0's shift (the bend output's water level)
  k    the places fc.0's gain rounds off (power(pile, F): 16 - I), its least and most over the 3,072 neurons
  the line holding R1's largest size, and how much of that line is word, position, proj bias, arrival write.

PREDICTION (the session's, written before running):
  s1 in {5, 6} only, on the judge windows as on the refs texts (the pilot: s1 = 5 at 636, 6 at 12 positions);
  so in {1, 2, 3}, mostly 2; sm between 3 and 6; k between 15 and 32 at every neuron (so the gain always rounds).
    python3 layer0/survey.py      -> layer0/survey.npz, layer0/survey.log (via tee)
"""
import sys, os, json, time, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")
HERE = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, HERE); os.chdir(HERE)
import judge5 as J
M, W, ES = J.M, J.W, J.ES
t0 = time.time()
def gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
REFS = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in REFS]
WIN = [w["ids"] for w in J.WINDOWS]
WTE, WPE = M["wte"], M["wpe"]; BP = W["proj.0"][2]
wf, wef, bf = W["fc.0"]

def layer0_record(ids):                               # base2.py's layer 0, its own functions, the shifts kept
    ids = np.asarray(ids); P = len(ids); A, B = WTE[ids], WPE[:P]; R = A + B
    x16, ex, LS = J.gauge_log(R); s = ex + ES
    QKV = J.gauged_bank(R, "qkv.0"); o = J.attention(QKV)
    o16, eo = J.to16(o, -ES); so = eo + ES
    att = J.plain_bank(o, "proj.0"); R1 = R + att
    x1, e1, LS1 = J.gauge_log(R1); s1 = e1 + ES
    F = (wef[None, :] + e1[:, None] + ES) * 65536 - LS1[:, None]; I = F >> 16; k = 16 - I
    H = J.gauged_bank(R1, "fc.0"); G = J.edge(H)
    g16, eg = J.to16(G, -ES); sm = eg + ES
    big = np.abs(R1).argmax(1)
    parts = np.stack([A[np.arange(P), big], B[np.arange(P), big], np.broadcast_to(BP[big], (P,)), (att - BP[None, :])[np.arange(P), big]], 1)
    return dict(s=s, so=so, s1=s1, sm=sm, kmin=k.min(1), kmax=k.max(1), big=big, parts=parts,
                R1max=np.abs(R1).max(1), omax=np.abs(o).max(1))

out = {}
for label, texts in (("refs", TEXTS), ("judge", WIN)):
    recs = [layer0_record(t) for t in texts]
    cat = {key: np.concatenate([r[key] for r in recs]) for key in recs[0]}
    pos = np.concatenate([np.arange(len(t)) for t in texts])
    out[label] = cat; out[label]["pos"] = pos
    print(f"{label}: {len(texts)} texts, {len(pos):,} positions   ({time.time() - t0:.0f}s, {gb():.2f} GB)", flush=True)
    for key in ("s", "so", "s1", "sm"):
        v, c = np.unique(cat[key], return_counts=True)
        print(f"   {key:3s}: " + ", ".join(f"{int(a)}: {int(b):,}" for a, b in zip(v, c)))
        if key in ("so", "s1", "sm"):
            v0, c0 = np.unique(cat[key][pos == 0], return_counts=True)
            print(f"        at position 0: " + ", ".join(f"{int(a)}: {int(b):,}" for a, b in zip(v0, c0)))
    print(f"   k (places fc.0's gain rounds off): least {int(cat['kmin'].min())}, most {int(cat['kmax'].max())}")
    v, c = np.unique(cat["big"], return_counts=True); order = np.argsort(-c)[:5]
    print("   line holding R1's largest size: " + ", ".join(f"line {int(v[i])}: {int(c[i]):,}" for i in order))
    pr = cat["parts"]; tot = np.abs(pr).sum(1)
    print(f"   that line's parts, median share of |word|+|position|+|proj bias|+|arrival write|: "
          + ", ".join(f"{n} {100 * np.median(np.abs(pr[:, j]) / tot):.1f}%" for j, n in enumerate(("word", "position", "proj bias", "arrival write"))))
    print(f"   R1's largest size: {int(cat['R1max'].min()):,} .. {int(cat['R1max'].max()):,}  (2^19 = 524,288; 2^20 = 1,048,576; 2^21 = 2,097,152)")
    print(f"   o's largest size:  {int(cat['omax'].min()):,} .. {int(cat['omax'].max()):,}", flush=True)
np.savez(f"{HERE}/layer0/survey.npz", **{f"{l}_{k}": v for l, d in out.items() for k, v in d.items()})
print(f"saved layer0/survey.npz   max memory {gb():.2f} GB, {time.time() - t0:.0f}s")
