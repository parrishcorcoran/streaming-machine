"""DIFFUSE look (a diagnostic, not a storable version): does the first word's error reach the judged word?
The first position sees nearly the same bright light in every text (its error is 95-97% standing: the same offset
for every text). Here the machine runs with the developed plates everywhere EXCEPT that position 0's piles come from
the original plates (as if position 0's error were cancelled exactly), and the reverse. Words out of 30."""
from diffuse_dark import *
import table_bits as _tb
from diffuse_dark import _swap
A_ = [int(a) for a in sys.argv[1:]] or [5, 6]
def dev_all(A, rule="v12"):
    out = {}
    for n in NAMES:
        q = plate(n); t, off = terms(n, q, rule); s = steps(t, off, A); out[n] = shift_back(develop_plain(q, s), s)
    return out
def run_mixed(ids, Wd, exact_rows):
    orig = {n: _tb.W[n] for n in NAMES}
    def bank(f, X, n):
        o = f(X, n); _swap(n, Wd[n]); d = f(X, n); _tb.W[n] = orig[n]
        d[exact_rows] = o[exact_rows]; return d
    P = len(ids); R = _tb.M["wte"][ids] + _tb.M["wpe"][:P]
    for l in range(12):
        R = R + bank(_tb.plain_bank, _tb.attention(bank(_tb.gauged_bank, R, f"qkv.{l}")), f"proj.{l}")
        R = R + bank(_tb.plain_bank, _tb.edge(bank(_tb.gauged_bank, R, f"fc.{l}")), f"mproj.{l}")
    w, we, b = _tb.W["head"]; x16, ex, LS = _tb.gauge_log(R[-1:])
    return _tb.race(_tb.power(_tb.product(x16, w), (we[None, :] + ex[:, None] + _tb.ES) * 65536 - LS[:, None])[0] + b)
assert words() == 30
for A in A_:
    Wd = dev_all(A)
    for label, rows in (("all developed", []), ("position 0 exact", [0]), ("positions 1-4 exact (only position 0 developed)", [1, 2, 3, 4]),
                        ("last position exact", [-1])):
        wd = sum(run_mixed(i, Wd, rows) == t for i, t in zip(_tb.inputs, _tb._TRUTH))
        print(f"v12 A={A}: {label:50s} words {wd}/30", flush=True)
