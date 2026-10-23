"""flat_build/2/plate_shuffle.py -- THE PLATE SHUFFLE CONTROL for pick 2 (prediction P11 in PREDICTION.txt, written
before this ran). Each gate's weights shuffled across the light lines (numpy's generator keyed [99, slab, layer], each
column permuted on its own), every gate keeping exactly its own values. Same groups (key [1, slab, layer, 12]), same
light. A (gbank) and C (the run) both laid on the shuffled plate; piles against the shuffled plate's direct product.
    python3 flat_build/2/plate_shuffle.py   -> flat_build/2/out/plate_shuffle.json, .log
"""
import sys, json, time, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/flat_build/2")
_argv = sys.argv; sys.argv = [sys.argv[0]]
import machine as MC
sys.argv = _argv
HERE = MC.HERE; OUT = MC.OUT; GB = MC.GB; SLABS = MC.SLABS; WORK = MC.WORK
LOG = open(f"{OUT}/plate_shuffle.log", "w")
def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
REAL = json.load(open(f"{OUT}/machine_g12.json"))["banks"]
Wz = np.load(f"{HERE}/base2_gpt2.npz")
light = {k: np.load(f"{HERE}/groups/light_gpt2/{k}.npy", mmap_mode="r") for k in ("qkv", "proj", "fc", "mproj")}
log("THE PLATE SHUFFLE CONTROL (each gate's weights shuffled across the lines), g = 12, all 48 banks, 648 positions")
res = {}; t0 = time.time()
for l in range(12):
    for kind in ("qkv", "proj", "fc", "mproj"):
        w = np.asarray(Wz[f"{kind}.{l}.w"]).astype(np.int64); N, M = w.shape
        ws = np.random.default_rng([99, SLABS.index(kind), l]).permuted(w, axis=0)
        assert np.array_equal(np.sort(ws, 0), np.sort(w, 0)), "a gate's values changed"
        x = np.ascontiguousarray(light[kind][l], np.int32); key = [1, SLABS.index(kind), l, 12]
        direct = GB.plain_product(x, ws.astype(np.int16))
        gp = GB.GPlate(ws, [(kind, 0, M)], 12, key); pA, cA = MC.a_read(gp, x)
        assert not (pA != direct).any(), "A on the shuffled plate"
        hA = int(gp.up[0]).bit_length(); del gp, pA
        rp = MC.RunPlate(ws, 12, key); pC, cC = rp.read(x)
        assert not (pC != direct).any(), "C on the shuffled plate"
        a, c = MC.tot(cA), MC.tot(cC); r = REAL[f"{kind}.{l}"]
        rec = dict(A=a, C=c, hA=hA, hA_real=r["A_h"], C_reads=rp.nreads, C_reads_real=r["C_reads_a_position"],
                   A_reads=a["nreads"] // len(x), A_reads_real=r["A"]["nreads"] // len(x))
        res[f"{kind}.{l}"] = rec
        sv = sum(a[k] for k in WORK) - sum(c[k] for k in WORK); svr = sum(r["A"][k] for k in WORK) - sum(r["C"][k] for k in WORK)
        log(f"  {kind:5s}.{l:<2d} exact (A and C = the shuffled plate's direct product); A reads a position {rec['A_reads']:,} "
            f"(real {rec['A_reads_real']:,}); C reads {rp.nreads:,} (real {r['C_reads_a_position']:,}, {100 * (rp.nreads / r['C_reads_a_position'] - 1):+.2f}%); "
            f"saving A - C {100 * sv / a['plain']:.3f} point (real {100 * svr / r['A']['plain']:.3f})")
        json.dump(res, open(f"{OUT}/plate_shuffle.json", "w"))
        del rp, pC
tot = {}
for n, rec in res.items():
    kind = n.split(".")[0]
    for kk in (kind, "ALL"):
        t = tot.setdefault(kk, dict(Cr=0, Crr=0, sv=0, svr=0, pl=0, plr=0))
        t["Cr"] += rec["C_reads"]; t["Crr"] += rec["C_reads_real"]
        t["sv"] += sum(rec["A"][k] for k in WORK) - sum(rec["C"][k] for k in WORK)
        r = REAL[n]; t["svr"] += sum(r["A"][k] for k in WORK) - sum(r["C"][k] for k in WORK)
        t["pl"] += rec["A"]["plain"]; t["plr"] += r["A"]["plain"]
for kk, t in tot.items():
    log(f"  {kk:6s} C reads a position: shuffled plate {t['Cr']:,} against real {t['Crr']:,} ({100 * (t['Cr'] / t['Crr'] - 1):+.2f}%); "
        f"the run's saving: shuffled plate {100 * t['sv'] / t['pl']:.3f} point of its plain, real {100 * t['svr'] / t['plr']:.3f} "
        f"({100 * (t['sv'] / t['pl']) / (t['svr'] / t['plr']):.1f}% of the real)")
log(f"done ({time.time() - t0:.0f}s, {MC.rss_mb()} MB)")
