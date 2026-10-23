"""DIFFUSE v10 -- what v1-v9 leave as the only lever: expose each LINE for the light that actually falls on it.
The ledger from v1-v9 (all judged by words):
  - a carry along any chain multiplies each pile's error power by sum h^2 of the carry (2 first order, 6 second
    order), measured 1.75-2.04 on every kind of plate (diffuse_look_standing.py): the light on consecutive lines is
    unrelated, in any order, so nothing cancels;
  - the part of the error that is the same on every word (the standing light: 25-58% of each bank's error power)
    can be removed for one constant per neuron (v8, v9) and the words do not move: the words are read by the
    MOVING light;
  - what is left is each pile's moving error, sum_i t_i^2 P_i, P_i the light's power on line i. Nearest rounding
    already makes each t_i as small as its step allows, so the only freedom is the step itself: for a fixed number of
    stored places, the scatter is least when every line's step D_i makes D_i^2 P_i the same (a printer lays coarser
    dots where the eye is least sensitive). Round one guessed P_i from the plate's own row sizes; here a light meter
    reads it.
The meter (whole numbers): the original machine is lit by the calibration windows (diffuse_dark.py: public-domain
text, not the judged prompts, positions 0-4); at qkv/fc the light a line gives the pile is the gauged stream,
G = x16 x 2^(8 + ex - log2 size) (the machine's own power(): a table read and a shift); at proj the arrival's output,
at mproj the bend's output (>> 8). P_i = sum over positions of G^2. In half places of the step: off_i = most common
lp - lp_i, lp_i = halfplace(P_i) >> 1 (whole places of power = half places of the light). The neuron term is round
one's (v12 own size, or v14 next path); the file and decoder are round one's unchanged (off_i is the stored per-line
byte). Development is plain nearest.
Prediction before running: if the row-size guess was already close to the light, the same words at the same size;
if not, at matched size (A chosen so the file is ~46 and ~51-57 MB) one or more words more than plain v12 (29 at
46.1 MB, 28 at 51.8, 30 at 56.7)."""
from diffuse_dark import *
import table_bits as _tb

def light_power(wins):
    P = {}; R = [_tb.M["wte"][ids] + _tb.M["wpe"][:len(ids)] for ids in wins]
    def gauged_light(r):
        x16, ex, LS = _tb.gauge_log(r)
        return _tb.power(x16 << 8, (ex * 65536 - LS)[:, None])
    for l in range(12):
        for bank, pb, nxt in ((f"qkv.{l}", f"proj.{l}", _tb.attention), (f"fc.{l}", f"mproj.{l}", _tb.edge)):
            P[bank] = sum((gauged_light(r) ** 2).sum(0) for r in R)
            o = [_tb.gauged_bank(r, bank) for r in R]; x = [nxt(ok) for ok in o]
            P[pb] = sum(((xk >> 8) ** 2).sum(0) for xk in x)
            R = [r + _tb.plain_bank(xk, pb) for r, xk in zip(R, x)]
    return P

WT = windows()
PW = light_power(WT)
OFF = {}
for n in NAMES:
    lp = halfplace(np.maximum(PW[n], 1)) >> 1
    off = mode(lp) - lp
    OFF[n] = np.clip(off, -60, 40)
    if n in ("qkv.0", "fc.5", "proj.5", "mproj.11"):
        print(n, "light-meter offsets (half places):", dict(zip(*np.unique(OFF[n], return_counts=True))), flush=True)

_terms_rowsize = terms
def terms_light(n, q, rule):
    t, _ = _terms_rowsize(n, q, rule); return t, OFF[n]
terms = terms_light                                         # write_file (diffuse_lib) reads `terms` from its module
import diffuse_lib as _dl; _dl.terms = terms_light
if __name__ == "__main__":
    args = sys.argv[1:] or ["v12:8", "v12:7", "v12:6"]
    for a in args:
        rule, A = a.split(":"); A = int(A)
        run(f"v10 light-metered lines, plain nearest, {rule} A={A}", rule, A, lambda n, q, s: develop_plain(q, s),
            note="line offsets from the measured light power (calibration text), neuron term round one's")
