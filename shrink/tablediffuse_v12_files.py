"""TABLEDIFFUSE v12 -- the best tables written to ONE real file each (tablediffuse_store.py), read back by a reader that
sees only the file, checked equal to what was judged, and judged again FROM THE FILE: words(entry=D, top=T) with D and
T exactly as the reader rebuilt and printed them. Then the whole machine from files: round one's plates file (v12 A=8,
56,742,543 bytes, read by plates_best_store.read_into_machine) with each tables file.
Files (scratchpad):
  tables_E8_c32.tdf    the E8 parity screen at c=32 (5 realizations {30,29,29,30,30}, as the cube at c=32)
  tables_D8_c34.tdf    the D8 parity screen at c=34 (5 realizations {28,29,29,30,29}, the cube at c=34 {29,29,29,30,29})
  tables_cube_c34.tdf  round one's own c=34 table, counting coder (identical values: 29/30)
Predicted: sizes 31,519,5xx / 23,763,4xx / 24,358,2xx bytes (+16 header +8 lengths, -2 widths, the lens 3,072 as before;
the D8/E8 files carry the 1,536-byte groups as in v9/v10); words from the files 30 / 28 / 29, exactly v9/v10/v8's.
Whole machine with plates v12 A=8: 30/30-ish for E8 c=32 (88.3 MB), 28-30 for the c=34 files (80.5 / 81.1 MB) --
round one's v12 A=8 + c=34 gave 30 (81.7 MB).
    python3 shrink/tablediffuse_v12_files.py
"""
import os, sys, time, json, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink"); sys.path.insert(0, "/Users/abundancemachine/streaming-machine")
os.chdir("/Users/abundancemachine/streaming-machine")
import tablediffuse_lib as td, tablediffuse_store as ts, develop_lib as dl
tb = td.machine(); E = tb._WTE0
SP = td.SP

def words_DT(D, T):
    head = tb.W["head"]; Tf = T.T.astype(np.float64)
    try:
        tb.W["head"] = (Tf, head[1], head[2]); return int(tb.words(entry=D))
    finally:
        tb.W["head"] = head

BEST = [("tables_E8_c32.tdf", 32, "E8"), ("tables_D8_c34.tdf", 34, "D8"), ("tables_cube_c34.tdf", 34, "cube")]
files = {}
for fn, c, kind in BEST:
    t0 = time.time(); path = os.path.join(SP, fn)
    if kind == "cube": D, e = td.develop(E, c, "nearest"); grp = cw = None
    else: D, e, grp, cw, o = td.lattice_develop(E, c, kind)
    size = ts.write(path, D, c, kind, grp, cw)
    Db, Tb, info = ts.read(path)
    same_D = bool(np.array_equal(Db, D)); same_T = bool(np.array_equal(Tb, dl.print_top(D))); del D, cw
    w = words_DT(Db, Tb); del Db, Tb
    row = dict(name=f"v12 FILE {fn}", spec=dict(c=c, kind=kind), file=path, file_bytes=size, MB=round(size / 1e6, 3),
               bits_per_value=round(8 * size / E.size, 4), roundtrip_ok=same_D and same_T, words=w,
               secs=round(time.time() - t0, 1))
    td.record(row); files[fn] = (path, size)

# the whole machine from files: plates (round one's v12 A=8 file) + each tables file
src = open("shrink/plates_best_store.py").read(); exec(src[:src.index("for label, rule, A, k in")])
pf = os.path.join(SP, "plates_v12_A8.bin"); ps = os.path.getsize(pf)
read_into_machine(pf)
print("plates v12 A=8 from file, tables as stored:", tb.words(), flush=True)
for fn, (path, size) in files.items():
    t0 = time.time(); Db, Tb, info = ts.read(path); w = words_DT(Db, Tb); del Db, Tb
    row = dict(name=f"v12 WHOLE MACHINE plates_v12_A8.bin + {fn}", plates_bytes=ps, tables_bytes=size,
               MB=round((ps + size) / 1e6, 3), words=w, secs=round(time.time() - t0, 1))
    td.record(row)
