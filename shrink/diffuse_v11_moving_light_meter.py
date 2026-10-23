"""DIFFUSE v11 -- meter the MOVING light, not all of it. v10 metered each line by its whole light power and lost
(59.3 MB 29, 54.2 MB 27, 48.6 MB 27 against plain v12's 56.7 30, 51.8 28, 46.1 29): it gave the brightest lines of
the stream (a few lines carry hundreds of times the average power) steps 4-4.5 places finer -- but that light stands
still (it is nearly the same on every word), and v8/v9 showed the standing light's error does not reach the words.
So the meter here reads only the part of each line's light that changes from text to text: for each position (0-4)
the spread across the 128 calibration windows, N x sum G^2 - (sum G)^2 (whole numbers; no division -- the factor N^2
is the same on every line and drops out of the offsets), summed over positions. Same file, decoder, neuron terms and
plain nearest development as v10.
Prediction before running: better than v10 at the same A; whether it beats round one's row-size meter at matched
size is open: +1 word at ~46 and ~51 MB if the moving light differs from the row-size guess."""
import sys
sys.argv_saved = sys.argv; sys.argv = sys.argv[:1]
src = open(__file__.replace("diffuse_v11_moving_light_meter.py", "diffuse_v10_light_meter.py")).read()
src = src[src.index("from diffuse_dark import"):src.index("WT = windows()")]
exec(src)
sys.argv = sys.argv_saved

def moving_power(wins):
    NW = len(wins); P = {}
    R = [_tb.M["wte"][ids] + _tb.M["wpe"][:len(ids)] for ids in wins]
    def gauged_light(r):
        x16, ex, LS = _tb.gauge_log(r)
        return _tb.power(x16 << 8, (ex * 65536 - LS)[:, None])
    def spread(X):                                          # X: list over windows of (positions, lines)
        S = np.stack(X); s1 = S.sum(0); s2 = (S * S).sum(0)   # per position, per line
        return (NW * s2 - s1 * s1).sum(0)
    for l in range(12):
        for bank, pb, nxt in ((f"qkv.{l}", f"proj.{l}", _tb.attention), (f"fc.{l}", f"mproj.{l}", _tb.edge)):
            P[bank] = spread([gauged_light(r) for r in R])
            o = [_tb.gauged_bank(r, bank) for r in R]; x = [nxt(ok) for ok in o]
            P[pb] = spread([xk >> 8 for xk in x])
            R = [r + _tb.plain_bank(xk, pb) for r, xk in zip(R, x)]
    return P

WT = windows()
PM = moving_power(WT)
OFFM = {}
for n in NAMES:
    lp = halfplace(np.maximum(PM[n], 1)) >> 1
    OFFM[n] = np.clip(mode(lp) - lp, -60, 40)
    if n in ("qkv.0", "fc.5", "proj.5", "mproj.11"):
        print(n, "moving-light offsets (half places):", dict(zip(*np.unique(OFFM[n], return_counts=True))), flush=True)
_terms_rowsize = terms
import diffuse_lib as _dl
def terms_moving(n, q, rule):
    t, _ = _terms_rowsize(n, q, rule); return t, OFFM[n]
_dl.terms = terms_moving
if __name__ == "__main__":
    args = sys.argv[1:] or ["v12:8", "v12:7", "v12:6"]
    for a in args:
        rule, A = a.split(":"); A = int(A)
        run(f"v11 moving-light-metered lines, plain nearest, {rule} A={A}", rule, A, lambda n, q, s: develop_plain(q, s),
            note="line offsets from the light's spread across texts (calibration), neuron term round one's")
