"""WHERE ON EACH PLATE THE LIGHT MEETS THE RELIEF: the relief's glow, input lane by input lane. 2026-10-06.
A look in real numbers, not a test of his sentence and not the machine's path.

HIS SENTENCE (his exact words): "It should be 100% removable after training."
SESSION'S RESTATEMENT (not his): what survives of the random start in plate W is round(s x W0), the relief R.

WHY THIS LOOK: keep_late.py kept the relief on the plates that carry the late water level and removed it from 92% of the
plate weights; 59-61% of the words stayed. steady_light.py found the light meets the relief at 0.5-0.8 of the relief's
share of the plate (fc, mproj, proj, the early qkv), the glow mostly on the word's own light, which no fixed light can
carry. The way around in the picture: the relief matters only where the light meets it. Each plate's input lanes are
lit unevenly; the relief on a lane the light hardly reaches gives hardly any glow. This look asks how unevenly: on each
plate, the input lanes ordered by the light's own size there (mean square over the 622 fed-in positions of the finished
machine), the relief kept on the brightest p of the lanes (rows) and removed from the rest:
  glow left(p) = mean |x . R_dim|^2 / mean |x . R|^2   (R_dim: the relief on the lanes outside the brightest p)
for p = 5, 10, 25, 50% of the lanes, a fixed list (no p is picked). The weights whose relief goes: 1 - p of the plate.
Lanes are ranked on the same light the glow is measured on: this is the most favourable case for the way around, and
a run of the machine on held-out texts would have to confirm any of it.

PREDICTIONS, written before the first run of this script:
  (1) qkv and fc (the gauge's lanes): glow left 40-80% at p = 10%, 20-60% at p = 25%.
  (2) mproj (the bend's lanes): 50-85% at p = 10%. proj (the arrival's lanes): 60-85% at p = 10%.
  (3) the output word table: 10-50% at p = 10% (its light is 98-99% the standing light, held on few lanes).
  (4) at p = 50%, every plate kind keeps 15-45% of its glow.
If (1)-(2) hold, keeping the relief on the bright lanes alone cannot take away most of the glow while removing most of
the relief: the word's own light is spread over the lanes.

  python3 lane_light.py size=160m
"""
import os, sys, json, resource
import numpy as np

A = dict(a.split("=", 1) for a in sys.argv[1:])
SIZE = A.get("size", "160m")
LATE = os.path.dirname(os.path.abspath(__file__)); HERE = os.path.dirname(LATE)
LOGF = open(os.path.join(LATE, f"lane_light_{SIZE}.log"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()

src = open(os.path.join(LATE, "plate_look.py")).read()                   # the machine's parts, as level_meeting.py does
cut = src.index("# ---------------------------------------------------------------- the runs")
G = {"__file__": os.path.join(LATE, "plate_look.py"), "__name__": "plate_look_parts"}
sys_argv = sys.argv; sys.argv = ["plate_look.py", f"size={SIZE}", "tag=lane_light_parts"]
exec(compile(src[:cut], "plate_look.py", "exec"), G)
sys.argv = sys_argv
os.remove(os.path.join(LATE, f"plate_look_{SIZE}-lane_light_parts.log"))
L, ES, M = G["L"], G["ES"], G["M"]
plain, gauge, rshift = G["plain"], G["gauge"], G["rshift"]
SHARE = G["SHARE"]; times, to30 = G["times"], G["to30"]
PS = (0.05, 0.10, 0.25, 0.50)

def forward_capture(ids, cap):
    """the finished machine (pythia_base2.py's parts, unchanged), recording the light arriving at every plate"""
    ids = np.asarray(ids); EW, EE = G["PL"]["embed"]
    R = rshift(EW[ids].astype(np.int64), -(EE[ids][:, None] + ES))
    for l in range(L):
        X1 = gauge(R, f"ln1.{l}"); QKV = plain(X1, f"qkv.{l}"); Ao = G["attention"](QKV)
        X2 = gauge(R, f"ln2.{l}"); Eo = G["edge"](plain(X2, f"fc.{l}"))
        for k, x in ((f"qkv.{l}", X1), (f"proj.{l}", Ao), (f"fc.{l}", X2), (f"mproj.{l}", Eo)):
            cap.setdefault(k, []).append((x / 2.0 ** ES).astype(np.float32))
        R = R + plain(Ao, f"proj.{l}") + plain(Eo, f"mproj.{l}")
    cap.setdefault("head", []).append((gauge(R, "lnf") / 2.0 ** ES).astype(np.float32))

refs = json.load(open(os.path.join(HERE, f"refs_pythia-{SIZE}.json")))[:30]
G["set_case"]("fin")
CAP = {}
for r in refs: forward_capture(r["ids"] + r["ref"][:15], CAP)
CAP = {k: np.concatenate(v) for k, v in CAP.items()}
G["PL"].clear(); G["CUR"].clear()
log(f"lane_light.py  size {SIZE}: {len(CAP['head'])} positions; a look in real numbers; p = {PS}")

RES = {}
for name in [f"{k}.{l}" for l in range(L) for k in ("qkv", "proj", "fc", "mproj")] + ["head"]:
    X = CAP[name].astype(np.float64); e = 2.0 ** M[name + ".e"].astype(np.float64)
    Rr = times(to30(SHARE[name]["s"]), M[name + ".s"].astype(np.int64)).astype(np.float64) * e[None, :]
    order = np.argsort(-(X ** 2).mean(0), kind="stable")                  # the lanes, brightest first
    whole = float(((X @ Rr) ** 2).sum(1).mean())
    lit = (X ** 2).mean(0)[order]; o = dict(left={}, light_share={})
    for p in PS:
        n = max(1, int(round(p * X.shape[1]))); dim = order[n:]
        o["left"][p] = float(((X[:, dim] @ Rr[dim]) ** 2).sum(1).mean()) / whole
        o["light_share"][p] = float(lit[:n].sum() / lit.sum())
    RES[name] = o
    log(f"  {name:8s} glow left with the relief kept on the brightest p of the lanes: "
        + "  ".join(f"{int(100*p):2d}% -> {100*o['left'][p]:5.1f}%" for p in PS)
        + "   | the light's share in those lanes: " + " ".join(f"{100*o['light_share'][p]:5.1f}" for p in PS))
    del X, Rr
log("\nby kind (median over layers): glow left at p = " + ", ".join(f"{int(100*p)}%" for p in PS))
for k in ("qkv", "proj", "fc", "mproj"):
    v = [RES[f"{k}.{l}"]["left"] for l in range(L)]
    log(f"  {k:6s} " + "  ".join(f"{100*np.median([x[p] for x in v]):5.1f}%" for p in PS))
json.dump({k: {"left": {str(p): v for p, v in o["left"].items()}, "light_share": {str(p): v for p, v in o["light_share"].items()}}
           for k, o in RES.items()}, open(os.path.join(LATE, f"lane_light_{SIZE}.json"), "w"))
log(f"peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30:.2f} GB")
