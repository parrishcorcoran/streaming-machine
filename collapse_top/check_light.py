"""Is the saved light (collapse_top/light_top.npz) the machine's own? Refs texts 0, 14, 29 and judge5 windows 0, 150,
299 are run again here with judge5's steps (as light_top.py does) and the light at the top (x16, ex, LS) compared
exactly with the saved one.
    python3 collapse_top/check_light.py
"""
import sys, os, json, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, HERE); os.chdir(HERE)
import judge5 as J
M = J.M; WTE, WPE = M["wte"], M["wpe"]
L = np.load(f"{HERE}/collapse_top/light_top.npz"); SRC, WIN = L["src"], L["win"]
refs = json.load(open(f"{HERE}/refs_many.json"))
def light_at_top(ids, rows):
    P = len(ids); R = WTE[ids] + WPE[:P]
    for l in range(12):
        R = R + J.plain_bank(J.attention(J.gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        R = R + J.plain_bank(J.edge(J.gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
    return J.gauge_log(R)                                  # every row, then the read rows taken
def compare(x16, ex, ls, g, name):
    ok = np.array_equal(x16, L["x16"][g]) and np.array_equal(ex, L["ex"][g]) and np.array_equal(ls, L["LS"][g])
    print(f"  {name}: {len(g)} positions, saved light identical: {ok}")
for t in (0, 14, 29):
    r = refs[t]; ids = r["ids"] + r["ref"][:16]; rows = np.arange(len(r["ids"]) - 1, len(ids) - 1)
    x16, ex, ls = light_at_top(ids, rows); g = np.flatnonzero((SRC == 0) & (WIN == t))
    compare(x16[rows], ex[rows], ls[rows], g, f"refs text {t}")
for i in (0, 150, 299):
    x16, ex, ls = light_at_top(J.WINDOWS[i]["ids"], None); rows = np.arange(J.FIRST, J.LAST)
    g = np.flatnonzero((SRC == 1) & (WIN == i)); compare(x16[rows], ex[rows], ls[rows], g, f"window {i}")
