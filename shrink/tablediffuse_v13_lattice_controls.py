"""TABLEDIFFUSE v13 -- controls for the parity screens.
(a) shuffle control: E8's grain (D - E) moved to random other words on the same line (one fixed-seed reshuffle per
    line): each line keeps exactly E8's grain values (x1.72 power), the 8-line group structure inside each word is gone.
    If the lattice's arrangement mattered (beyond its power), the control would differ from E8; the relation says only
    the power reaches the readers.  E8 c=32 and c=34, canonical + phases 1, 2.
    Predicted: each control within +-1 of its E8 realization (E8: c=32 {30,29,29}, c=34 {28,28,29}).
(b) the judge route: the E8 c=34 canonical table judged once through words(entry=D, top=T) exactly as the harness
    takes it (the int64 route), to check the W-head route used everywhere gives the same count (28).
    python3 shrink/tablediffuse_v13_lattice_controls.py
"""
import sys, time, numpy as np; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
import tablediffuse_lib as td, develop_lib as dl
tb = td.machine(); E = tb._WTE0
for c in (32, 34):
    for p in (None, 1, 2):
        t0 = time.time(); D, e, grp, cw, o = td.lattice_develop(E, c, "E8", phase_seed=p); D += o[None, :].astype(np.int32); del cw
        if c == 34 and p is None:
            T = dl.print_top(D); w = int(tb.words(entry=D, top=T)); del T
            td.record(dict(name="v13 judge route check: E8 c=34 via words(entry=D, top=T)", words=w, secs=round(time.time() - t0, 1)))
        rng = np.random.default_rng(1300 + c + (p or 0))
        for i in range(768):
            g = D[:, i].astype(np.int64) - E[:, i]; D[:, i] = (E[:, i] + g[rng.permutation(E.shape[0])]).astype(np.int32)
        row = dict(name=f"v13 CONTROL E8 grain on random words of the same line, c={c}" + ("" if p is None else f" phase seed={p}"),
                   spec=dict(c=c, kind="E8", phase_seed=p))
        row.update(td.judge(D)); row["grain_steps2"] = round(td.grain_steps2(D, E, e), 4)
        row["secs"] = round(time.time() - t0, 1); td.record(row); del D
