"""JUDGE: the plate's sign run on top of the confirmed per-place machine, counted from the plates alone (no light).
Prediction in PREDICTION.txt beside this file, written before this ran. Whole numbers only.

Per (gate j, group k) cell, groups keyed [1, slab, layer, g] (the confirmed machine's own draw), g by the readers
relation (12 at qkv/fc, 11 at proj/mproj) and also g = 12 everywhere:
  A  confirmed per-place: reads = h (the plate's up/hole places, PP.rule), top digits = lit signed digits of w_hi
  B  A plus the run inside places 0..h-1: reads = c + 1, c = the cell's top run start in the centred h-form
  C  design 2's form: the full 16 places with the run: reads = c16 + 1, no top digits
  BC per cell the cheaper of B and C (cost = 100 reads + 87 top digits; 87/100 ~ s_x / s_V)
  Dh per cell the cheapest split place h' in 8..16 with the run (the floor of any per-cell split)
Each cell part is exact by the centred identity: sum_{b<h} 2^b (2 bit_b - 1) = 2 w_lo + 1, for any h, so the cells may
use different h and share the one -X.  Checked here: for every plate, sum over cells of each form's digits gives back
2 w + 1 per line and gate (asserted on the cells' own numbers).
"""
import sys, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"
sys.path.insert(0, f"{HERE}/groups2")
_argv = sys.argv; sys.argv = [sys.argv[0]]
import uh_perplace as PP
import uh_rebuild as R
sys.argv = _argv
SLABS = ["qkv", "proj", "fc", "mproj", "o", "gu", "down"]
GREL = {"qkv": 12, "proj": 11, "fc": 12, "mproj": 11}

def sdc(a):
    a = np.abs(np.asarray(a, np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)
def blen(v):                          # bit length of non-negative whole numbers
    v = np.asarray(v, np.int64); out = np.zeros(v.shape, np.int64)
    for s in range(17): out += (v >> s) > 0
    return out
def split(w, h):                      # centred: w = wlo + 2^h whi, wlo in [-2^(h-1), 2^(h-1))
    C = 1 << (h - 1); wlo = ((w + C) & ((1 << h) - 1)) - C; whi = (w - wlo) >> h
    assert np.array_equal(wlo + (whi << h), w); return wlo, whi
def runstart(wlo):                    # smallest c with wlo in [-2^c, 2^c)
    return blen(np.where(wlo >= 0, wlo, ~wlo))

def cells(w, glines, goff, h):
    """per cell (G, M): reads with the run (c + 1, c <= h - 1) and top digits"""
    wlo, whi = split(w, h); c = np.minimum(runstart(wlo), h - 1); td = sdc(whi)
    sizes = np.diff(goff); G = len(sizes); M = w.shape[1]
    cmax = np.zeros((G, M), np.int64); tdig = np.zeros((G, M), np.int64)
    for k in range(G):
        L = glines[goff[k]:goff[k + 1]]
        cmax[k] = c[L].max(0); tdig[k] = td[L].sum(0)
    # exact: the read places 0..c-1 bitwise, then one read at place c of the top pattern, give 2 wlo + 1 per line
    if h >= 2:
        E = wlo + (1 << (h - 1)); cl = np.zeros_like(E);
        for k in range(G):
            L = glines[goff[k]:goff[k + 1]]; cl[L] = cmax[k][None, :]
        part = np.zeros_like(E)
        for b in range(h - 1):
            bit = (E >> b) & 1; part += np.where(b < cl, (2 * bit - 1) << b, 0)
        topb = (E >> (h - 1)) & 1; part += (2 * topb - 1) << cl
        assert np.array_equal(part, 2 * wlo + 1), "the run does not give the cell back"
    return cmax + 1, tdig

def main():
    Z = np.load(f"{HERE}/base2_gpt2.npz"); tot = {}
    for gmode in ("readers", "12"):
        acc = {}
        for l in range(12):
            for kind in ("qkv", "proj", "fc", "mproj"):
                w = np.asarray(Z[f"{kind}.{l}.w"]).astype(np.int64); N, M = w.shape
                g = GREL[kind] if gmode == "readers" else 12
                up, _ = PP.rule(w, [(kind, 0, M)], g); u = int(up[0]); assert (up == u).all()
                h = u.bit_length(); assert u == (1 << h) - 1, "up/hole places not 0..h-1"
                glines, goff = R.make_groups(N, g, [1, SLABS.index(kind), l, g])
                rA = h; _, whiA = split(w, h)
                rB, tB = cells(w, glines, goff, h)                      # tB = A's top digits too
                rC, tC = cells(w, glines, goff, 16); assert not tC.any()
                costA = 100 * rA + 87 * tB; costB = 100 * rB + 87 * tB; costC = 100 * rC
                costBC = np.minimum(costB, costC)
                costD = costBC.copy()
                for hh in range(8, 16):
                    r_, t_ = cells(w, glines, goff, hh); costD = np.minimum(costD, 100 * r_ + 87 * t_)
                a = acc.setdefault(kind, dict(cells=0, A=0, B=0, C=0, BC=0, D=0, readsA=0, readsB=0, readsC=0, topA=0))
                a["cells"] += costA.size; a["A"] += int(costA.sum()); a["B"] += int(costB.sum()); a["C"] += int(costC.sum())
                a["BC"] += int(costBC.sum()); a["D"] += int(costD.sum()); a["readsA"] += rA * costA.size
                a["readsB"] += int(rB.sum()); a["readsC"] += int(rC.sum()); a["topA"] += int(tB.sum())
        tot[gmode] = acc
        print(f"g by {gmode}:")
        for kind, a in acc.items():
            n = a["cells"]
            print(f"  {kind:6s} reads a cell A {a['readsA'] / n:6.3f}  B {a['readsB'] / n:6.3f}  C {a['readsC'] / n:6.3f}; "
                  f"top digits a cell (A, B) {a['topA'] / n:5.3f} | read-equivalents vs A: B {100 * a['B'] / a['A']:6.2f}%  "
                  f"C {100 * a['C'] / a['A']:6.2f}%  BC {100 * a['BC'] / a['A']:6.2f}%  D {100 * a['D'] / a['A']:6.2f}%", flush=True)

if __name__ == "__main__":
    main()
