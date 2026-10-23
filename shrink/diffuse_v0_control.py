"""DIFFUSE v0 -- control: plain nearest development at round one's exposures (v12 half-place meters, v14 next path),
written to this family's file and decoded back. Same exposure, same file format, same coder as every carry version
that follows, so any change in size or words afterwards is the carry alone.
Prediction before running: sizes within a few hundred bytes of round one's (v12 A=8 56.742, A=7 51.756, A=6 46.109,
A=5 41.133; v14 A=6 45.859, A=5 40.561) and the same words (30, 28, 29, 22; 29, 26) -- the same numbers are stored."""
from diffuse_lib import *
args = sys.argv[1:] or ["v12:8", "v12:7", "v12:6", "v12:5", "v14:6", "v14:5"]
for a in args:
    rule, A = a.split(":"); A = int(A)
    run(f"v0 plain nearest {rule} A={A}", rule, A, lambda n, q, s: develop_plain(q, s),
        note="control: no carry, round one's development")
