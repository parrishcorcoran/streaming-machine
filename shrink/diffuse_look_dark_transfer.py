"""DIFFUSE look (analysis of v8/v9's negative, not a judge): do the dark-frame constants, measured on the calibration
text, remove the standing error on the JUDGED prompts too? For plain v12 A=5 with and without v9's constants
(positions 1-4, local), each bank lit by the original machine's light on the 30 judged prompts: the error power at
the last position (where the word is read) and at positions 1-4, summed per kind. Whole-number errors; ratios printed."""
from diffuse_dark import *
import table_bits as _tb
import diffuse_dark as dd
from diffuse_dark import _swap
dd.POS = 1
A = int(sys.argv[1]) if len(sys.argv) > 1 else 5
Wd = {}
for n in NAMES:
    q = plate(n); t, off = terms(n, q, "v12"); s = steps(t, off, A); Wd[n] = shift_back(develop_plain(q, s), s)
C = corrections(Wd, windows())
orig = {n: _tb.W[n] for n in NAMES}
acc = {}
for ids in _tb.inputs:
    R = _tb.M["wte"][ids] + _tb.M["wpe"][:len(ids)]
    for l in range(12):
        for bank, pb, nxt in ((f"qkv.{l}", f"proj.{l}", _tb.attention), (f"fc.{l}", f"mproj.{l}", _tb.edge)):
            o = _tb.gauged_bank(R, bank); _swap(bank, Wd[bank]); d = _tb.gauged_bank(R, bank); _tb.W[bank] = orig[bank]
            x = nxt(o); po = _tb.plain_bank(x, pb); _swap(pb, Wd[pb]); pd = _tb.plain_bank(x, pb); _tb.W[pb] = orig[pb]
            for n, D in ((bank, (d - o).astype(float)), (pb, (pd - po).astype(float))):
                k = n.split(".")[0]; a = acc.setdefault(k, np.zeros(4))
                a[0] += (D[-1] ** 2).sum(); a[1] += ((D[-1] + C[n]) ** 2).sum()
                a[2] += (D[1:] ** 2).sum(); a[3] += ((D[1:] + C[n][None, :]) ** 2).sum()
            R = R + po
for k, a in acc.items():
    print(f"{k:6s} judged prompts, A={A}: error power with constants / without -- last position {a[1]/a[0]:.3f}, positions 1-4 {a[3]/a[2]:.3f}")
