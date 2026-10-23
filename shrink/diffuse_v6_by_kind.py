"""DIFFUSE v6 -- where does the carry hurt? v4's first-order carry (in steps) on ONE kind of plate at a time, the
other three developed plain, all at the same exposure. If any kind's light is mostly common to its lines (mproj's
lines carry the bend's output, which is small and of one sign on most lines; proj's lines carry the attention's
average of v, which holds v's bias on every token), the carry should cost nothing there or even help.
Prediction before running: at v12 A=5 (plain 22) qkv and fc carries lose words (their light is the stream, centred
by the folded gauge: no common part survives); mproj and proj the same or +-1."""
from diffuse_lib import *
args = sys.argv[1:] or ["v12:5", "v12:6"]
for a in args:
    rule, A = a.split(":"); A = int(A)
    for kind in ("qkv", "proj", "fc", "mproj"):
        def dev(n, q, s, kind=kind):
            if n.split(".")[0] == kind: return develop_carry(q, s, axis=0, kind=1, unit=True)
            return develop_plain(q, s)
        run(f"v6 carry (steps) on {kind} only, {rule} A={A}", rule, A, dev, note=f"first order in steps on {kind}, rest plain")
