"""THE LIGHT AT EVERY PLATE AND EVERY WRITE INTO THE STREAM, for the finished and the blank Pythia. 2026-10-06.

HIS WORDS (2026-10-06), only these are his: "The blank work must be in the embedding table equal in math to the MLP.
It has to cancel. So it's either tiny bit of all the 150k tokens embeddings vs MLP, we must not be the only one to
realize this . Our work of canceling might be doing some of it to begin with. Lining up and cancelling might do most.
Can you line up the blanks and see if it all cancels?"

This script measures nothing of that yet. It runs the whole-number machine (../foundation0/pythia_base2.py, loaded as
it is, no edits, its own parts) on the 30 texts of texts_pythia.json, each fed its prompt plus that size's own 15 fed-in
words (refs_pythia-<size>.json: 622 positions at 70m, as lane_light.py and steady_light.py used), and records
  - the light each plate meets, exactly as its product meets it: the 16-place whole numbers x16 of pythia_base2.to16
    (qkv.l and fc.l: the gauge's output; proj.l: the arrival's output; mproj.l: the bend's output; head: the top gauge)
  - every write into the stream at every position, at the stream's scale 2^-16: the word's row, and at every layer the
    attention write and the MLP write (biases included, as the machine adds them)
plates=finished (step143000) gives the finished machine's own light; plates=start (step0) the blank machine's own light
on the same words.

CHECKS (PREDICTION, written before the first run):
  - finished: the head's piles at the 16 read places of every text have the same sha256 as words_pythia-<size>.json
    (pythia_base2.py's own run): 30 of 30 texts.
  - start: the same piles as pythia_base2.forward(...) run here on the same words (the module's own forward, a second
    call of the same code): 30 of 30.
  - every position: the word row plus every write equals the stream at the top, digit for digit.
  - texts_pythia.json's ids equal refs_pythia-<size>.json's prompt ids for all 30 texts.

    python3 capture.py size=70m plates=finished      (writes lineblanks/light_<size>_<plates>.npz)
"""
import os, sys, json, hashlib, importlib.util, resource, time
import numpy as np

A = dict(a.split("=", 1) for a in sys.argv[1:])
SIZE, PLATES = A.get("size", "70m"), A.get("plates", "finished")
OUT = os.path.dirname(os.path.abspath(__file__)); F0 = os.path.join(os.path.dirname(OUT), "foundation0")
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
t0 = time.time()

argv = sys.argv; sys.argv = ["pythia_base2.py", f"size={SIZE}", f"plates={PLATES}"]
spec = importlib.util.spec_from_file_location("pythia_base2_lb", os.path.join(F0, "pythia_base2.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); sys.argv = argv
assert m.PLATES == PLATES and m.ORDER == "plain" and (m.PERM == np.arange(m.d)).all()
L, d, ES = m.L, m.d, m.ES

def forward_capture(ids):
    """pythia_base2.forward's own steps, every position read at the head; records each plate's light and every write"""
    ids = np.asarray(ids)
    R = m.rshift(m.pick(m.EMB_F[ids], m.EMB_S[ids])[:, m.PERM], -(m.EMB_E[ids][:, None] + ES))
    writes = [R.copy()]; light = {}
    for l in range(L):
        X1 = m.gauge(R, f"ln1.{l}"); light[f"qkv.{l}"] = m.to16(X1)[0]
        Ao = m.attention(m.plain(X1, f"qkv.{l}")); light[f"proj.{l}"] = m.to16(Ao)[0]
        At = m.plain(Ao, f"proj.{l}")
        X2 = m.gauge(R, f"ln2.{l}"); H = m.plain(X2, f"fc.{l}"); light[f"fc.{l}"] = m.to16(X2)[0]
        Eo = m.edge(H); light[f"mproj.{l}"] = m.to16(Eo)[0]
        Mo = m.plain(Eo, f"mproj.{l}")
        writes += [At, Mo]; R = R + At + Mo
    y16, ey = m.to16(m.gauge(R, "lnf")); light["head"] = y16
    w, e, _, wmax = m.PL["head"]
    lg = m.rshift(m.product(y16[-16:], w, wmax), -(e[None, :] + ey[-16:, None] + ES))
    W = np.stack(writes, 1)
    assert np.array_equal(W.sum(1), R), "word row + writes != the stream at the top"
    return light, W, lg

texts = json.load(open(os.path.join(F0, "texts_pythia.json")))
refs = json.load(open(os.path.join(F0, f"refs_pythia-{SIZE}.json")))
assert len(texts) == 30 and all(t["ids"] == r["ids"] for t, r in zip(texts, refs[:30]))
ref_sha = None
if PLATES == "finished":
    ref_sha = [x["piles_sha256"] for x in json.load(open(os.path.join(F0, f"words_pythia-{SIZE}.json")))["texts_detail"]]
LIGHT = {}; WR = []; rows = []; same = 0; p0 = 0
for i, r in enumerate(refs[:30]):
    ids = r["ids"] + r["ref"][:15]
    light, W, lg = forward_capture(ids)
    sha = hashlib.sha256(lg.tobytes()).hexdigest()
    if ref_sha is not None: ok = sha == ref_sha[i]
    else: ok = sha == hashlib.sha256(m.forward(ids, last=16)[1].tobytes()).hexdigest()
    same += ok
    for k, v in light.items():
        assert np.abs(v).max() <= 1 << 15; LIGHT.setdefault(k, []).append(v.astype(np.int32))
    WR.append(W); rows.append((p0, p0 + len(ids))); p0 += len(ids)
    print(f"  text {i:2d} {len(ids):3d} positions  head piles {'SAME' if ok else 'DIFFERENT'}  ({time.time() - t0:.0f}s, {rss()} MB)", flush=True)
LIGHT = {k: np.concatenate(v) for k, v in LIGHT.items()}; WR = np.concatenate(WR)
dst = os.path.join(OUT, f"light_{SIZE}_{PLATES}.npz")
np.savez(dst, writes=WR, rows=np.array(rows), **{"light." + k: v for k, v in LIGHT.items()})
print(f"{SIZE} {PLATES}: {p0} positions; head piles at the 16 read places the same as "
      f"{'words_pythia-' + SIZE + '.json' if ref_sha else 'pythia_base2.forward'} in {same} of 30 texts; "
      f"word row + writes = the top stream at every position (asserted); wrote {dst} ({time.time() - t0:.0f}s, max {rss()} MB)")
