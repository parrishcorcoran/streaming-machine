"""COMBINED v1 -- each bank kind its own exposure, on the whole machine (tables c=34, everything else exact).
Picture: every developed plate lays grain on the light that passes it; the grain of all 48 plates and the tables adds
in power at the race at the top. A half place coarser on a kind saves ~1/2 bit on each of its weights and doubles that
kind's grain power. The cheapest grain is where the light does not carry it forward:
  - fc   : its light goes through the bend; a neuron below its edge passes almost nothing, so its grain is shaded.
  - qkv  : q, k grain only shifts arrival times by a little; v grain is averaged over the positions.
  - proj : small (7.1M weights), its whole cost is 4.8 MB, a coarse proj saves little but costs little.
  - mproj: its light is the bent light, bright on few neurons, and its grain goes straight into the stream.
Round one (whole place 2 coarser on one kind, at a=4): proj 29, qkv 27, fc 27, mproj 24 -> per weight mproj is the
most sensitive (6 words / 28M) and proj, qkv, fc about equal (1/7M, 3/21M, 3/28M).
Predictions before running (base all A=8 = 30/30, 83.04 MB):
  proj A=4: 30   fc A=6: 29-30   qkv A=6: 29   mproj A=6: 28
  mixed (proj 4, qkv 7, fc 6, mproj 8): 29 at ~76.1 MB;  mixed with mproj one half place finer (9): 29-30 at ~77.8 MB"""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from combined_lib import *
def per_kind(Ak, rule="v12"): return cfg_by(lambda n: (rule, Ak[kind(n)]))
RUNS = [
    ("v1 proj A=4, rest A=8", dict(qkv=8, proj=4, fc=8, mproj=8), "30"),
    ("v1 fc A=6, rest A=8", dict(qkv=8, proj=8, fc=6, mproj=8), "29-30"),
    ("v1 qkv A=6, rest A=8", dict(qkv=6, proj=8, fc=8, mproj=8), "29"),
    ("v1 mproj A=6, rest A=8", dict(qkv=8, proj=8, fc=8, mproj=6), "28"),
    ("v1 mixed proj 4, qkv 7, fc 6, mproj 8", dict(qkv=7, proj=4, fc=6, mproj=8), "29"),
    ("v1 mixed proj 4, qkv 7, fc 6, mproj 9", dict(qkv=7, proj=4, fc=6, mproj=9), "29-30"),
]
sel = [int(i) for i in sys.argv[1].split(",")] if len(sys.argv) > 1 else range(len(RUNS))
for i in sel:
    name, Ak, pred = RUNS[i]
    judge(name, per_kind(Ak), 34, note="per bank kind, rule v12, tables c=34", predicted=pred)
