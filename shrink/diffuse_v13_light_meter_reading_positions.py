"""DIFFUSE v13 -- the light meter read where the word is read. The diagnostic diffuse_look_position0.py (v12 A=5):
all developed 22/30; position 0's piles exact 24; positions 1-4 exact (only position 0 developed) 28; last position
exact 28. The words are hurt by the error of the reading positions' own piles, read by THEIR light. v10 metered the
whole light of positions 0-4, and position 0's light (the first word: a few lines hundreds of times brighter, the same
in every text) set the brightest lines' steps 4-4.5 places finer than the rest. Here the meter reads the light power
of positions 1-4 only (same whole-number meter as v10: G^2 summed, half places of the light). File, decoder, neuron
terms and plain nearest development as v10.
Prediction before running: between v10 and round one; beats round one's row-size meter at matched size only if the
reading positions' light differs from what the plate rows say (one word at ~46 or ~51 MB would be the sign)."""
import sys
_argv = sys.argv; sys.argv = sys.argv[:1]
src = open(__file__.replace("diffuse_v13_light_meter_reading_positions.py", "diffuse_v10_light_meter.py")).read()
src = src[src.index("from diffuse_dark import"):src.index("WT = windows()")]
exec(src)
sys.argv = _argv
def light_power_14(wins):
    P = {}; R = [_tb.M["wte"][ids] + _tb.M["wpe"][:len(ids)] for ids in wins]
    def gauged_light(r):
        x16, ex, LS = _tb.gauge_log(r)
        return _tb.power(x16 << 8, (ex * 65536 - LS)[:, None])
    for l in range(12):
        for bank, pb, nxt in ((f"qkv.{l}", f"proj.{l}", _tb.attention), (f"fc.{l}", f"mproj.{l}", _tb.edge)):
            P[bank] = sum((gauged_light(r)[1:] ** 2).sum(0) for r in R)
            o = [_tb.gauged_bank(r, bank) for r in R]; x = [nxt(ok) for ok in o]
            P[pb] = sum(((xk[1:] >> 8) ** 2).sum(0) for xk in x)
            R = [r + _tb.plain_bank(xk, pb) for r, xk in zip(R, x)]
    return P
WT = windows(); P14 = light_power_14(WT); OFF14 = {}
for n in NAMES:
    lp = halfplace(np.maximum(P14[n], 1)) >> 1; OFF14[n] = np.clip(mode(lp) - lp, -60, 40)
    if n in ("qkv.0", "fc.5", "proj.5", "mproj.11"):
        print(n, "reading-position offsets:", dict(zip(*np.unique(OFF14[n], return_counts=True))), flush=True)
_terms_rowsize = terms
import diffuse_lib as _dl
def terms14(n, q, rule):
    t, _ = _terms_rowsize(n, q, rule); return t, OFF14[n]
_dl.terms = terms14
args = sys.argv[1:] or ["v12:8", "v12:7", "v12:6"]
for a in args:
    rule, A = a.split(":"); A = int(A)
    run(f"v13 light meter on positions 1-4, plain nearest, {rule} A={A}", rule, A, lambda n, q, s: develop_plain(q, s),
        note="line exposure from the reading positions' light power (calibration text)")
