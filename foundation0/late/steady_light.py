"""WHERE THE RELIEF'S GLOW COMES FROM: the steady light and the word's own light, at every plate. 2026-10-06.
A look in real numbers, not a test of his sentence and not the machine's path.

HIS SENTENCE (his exact words): "It should be 100% removable after training."
SESSION'S RESTATEMENT (not his): what survives of the random start in plate W is round(s x W0), the relief.

WHY THIS LOOK: removing the relief from a plate adds a glow x . R to the light leaving it (x the light arriving, R the
relief). The light arriving has a steady part, the same at every position of every text, and the word's own part.
  - exactly steady by the machine's wiring (a relation, not a measurement): the gauge's offset beta reaches every
    plate after a gauge (qkv, fc, the output word table) unchanged; the value part of qkv's bias reaches proj unchanged
    (the arrival's shares add up to 1). The glow of the relief on this light is one fixed vector, beta . R, which the
    plate's own bias (a fixed light, never a draw) could carry exactly, so removing the relief would not lose it.
  - the mean light over the 622 fed-in positions of the 30 texts (includes the standing pattern): a measured look.
  - the rest: the word's own light. Its glow cannot be moved into any fixed light.
For each plate, on the finished machine (the same fed-in words as keep_late.py), with R = round(s W0) exactly as
removed there, in real numbers:
  whole   mean over positions of |x R|^2                    the relief's whole glow
  exact   |beta R|^2  (qkv, fc, head: gauge offset; proj: value bias; mproj: none)
  mean    |mean(x) R|^2
  left    mean of |(x - beta) R|^2 / whole                  what would stay after moving the exact steady glow
  size    sqrt(whole / mean |x W|^2)                        the relief's glow against the finished plate's light
          (W the finished plate, without its bias)

PREDICTIONS, written before the first run of this script:
  (1) qkv, fc and the output word table: the exact steady part (beta) carries 1-15% of the relief's glow; the mean
      light carries 20-70%.
  (2) proj: the value bias carries 10-60% of the relief's glow.
  (3) mproj: the mean light carries 10-50%.
  (4) size: between 0.3 and 1.5 times the plate's rho at every plate (the built part lines up with the light more
      than a random relief does, so mostly below rho).
If (1) holds, moving the exact steady glow into the biases cannot bring back most of the lost words: the glow that
costs them is on the word's own light.

  python3 steady_light.py size=160m
"""
import os, sys, json, resource
import numpy as np

A = dict(a.split("=", 1) for a in sys.argv[1:])
SIZE = A.get("size", "160m")
LATE = os.path.dirname(os.path.abspath(__file__)); HERE = os.path.dirname(LATE)
LOGF = open(os.path.join(LATE, f"steady_light_{SIZE}.log"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()

src = open(os.path.join(LATE, "plate_look.py")).read()                   # the machine's parts, as level_meeting.py does
cut = src.index("# ---------------------------------------------------------------- the runs")
G = {"__file__": os.path.join(LATE, "plate_look.py"), "__name__": "plate_look_parts"}
sys_argv = sys.argv; sys.argv = ["plate_look.py", f"size={SIZE}", "tag=steady_light_parts"]
exec(compile(src[:cut], "plate_look.py", "exec"), G)
sys.argv = sys_argv
os.remove(os.path.join(LATE, f"plate_look_{SIZE}-steady_light_parts.log"))
L, NH, HS, d, ES, M = G["L"], G["NH"], G["HS"], G["d"], G["ES"], G["M"]
plain, gauge, rshift, to16 = G["plain"], G["gauge"], G["rshift"], G["to16"]
SHARE = G["SHARE"]; times, to30 = G["times"], G["to30"]

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
CAP = {k: np.concatenate(v).astype(np.float64) for k, v in CAP.items()}
G["PL"].clear(); G["CUR"].clear()                                        # the machine's plates are no longer needed
log(f"steady_light.py  size {SIZE}: {len(CAP['head'])} positions; a look in real numbers")

def steady(name):
    kind, _, l = name.partition(".")
    if kind in ("qkv", "fc"): return M[f"{'ln1' if kind == 'qkv' else 'ln2'}.{l}.b"] / 2.0 ** ES
    if name == "head": return M["lnf.b"] / 2.0 ** ES
    if kind == "proj":
        b = M[f"qkv.{l}.b"] / 2.0 ** ES
        return np.concatenate([b[h * 3 * HS + 2 * HS:h * 3 * HS + 3 * HS] for h in range(NH)])
    return None

RES = {}
for name in [f"{k}.{l}" for l in range(L) for k in ("qkv", "proj", "fc", "mproj")] + ["head"]:
    X = CAP[name]; e = 2.0 ** M[name + ".e"].astype(np.float64)
    Rr = times(to30(SHARE[name]["s"]), M[name + ".s"].astype(np.int64)) * e[None, :]
    W = M[name + ".w"].astype(np.float64) * e[None, :]
    Y = X @ Rr; whole = float((Y ** 2).sum(1).mean()); fullW = float(((X @ W) ** 2).sum(1).mean())
    xm = X.mean(0); mean = float(((xm @ Rr) ** 2).sum())
    c = steady(name)
    exact = float(((c @ Rr) ** 2).sum()) if c is not None else 0.0
    left = float((((X - c[None, :]) @ Rr) ** 2).sum(1).mean()) / whole if c is not None else 1.0
    o = dict(exact=exact / whole, mean=mean / whole, left=left, size=float(np.sqrt(whole / fullW)), rho=SHARE[name]["rho"],
             beta_vs_light=float(np.sqrt((c ** 2).sum() / (X ** 2).sum(1).mean())) if c is not None else None)
    RES[name] = o
    log(f"  {name:8s} exact steady {100*o['exact']:5.1f}%  mean light {100*o['mean']:5.1f}%  left after moving the exact "
        f"part {100*o['left']:5.1f}%  | glow/plate light {o['size']:.4f} (rho {o['rho']:.4f}, x{o['size']/o['rho']:.2f})"
        + (f"  | |steady light|/|light| {o['beta_vs_light']:.3f}" if c is not None else ""))
    del X, Rr, W, Y
log("\nby kind (median over layers): exact steady / mean light / left after moving the exact part / glow over rho")
for k in ("qkv", "proj", "fc", "mproj"):
    v = [RES[f"{k}.{l}"] for l in range(L)]
    log(f"  {k:6s} {100*np.median([x['exact'] for x in v]):5.1f}%  {100*np.median([x['mean'] for x in v]):5.1f}%  "
        f"{100*np.median([x['left'] for x in v]):5.1f}%  x{np.median([x['size']/x['rho'] for x in v]):.2f}")
json.dump(RES, open(os.path.join(LATE, f"steady_light_{SIZE}.json"), "w"))
log(f"peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30:.2f} GB")
