"""THE FLAT LEFT INSIDE EACH GATE: after the table has paired the light inside every random group of lines, each gate
adds one packet a sheet; at one place the packets of all the groups then pair off with each other inside the gate.
Measured here on the sample of sizes.py (layer 5, 48 positions, g by the relation): the photons the gate adds (its
up/hole packets' signed digits) against the signed digits of what is left at each place after the groups pair off, and
of the whole doubled pile after the places pair off too.

PREDICTION (written before the run): a place's 1/g x N packets of about 4.65 photons pair off to a number about
sqrt(N/g) times one packet, about 1.5-2 more places, so about 6 photons: 98% of the packet photons at the 768-line
plates and 99.5% at mproj (256 groups) are the flat between groups. The shuffle control: the same within 0.2 point.

    python3 flat_design/1/flat_between.py  -> printed, and flat_design/1/flat_between.json
"""
import sys, json, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/flat_design/1"
sys.path.insert(0, f"{HERE}/groups2")
_argv = sys.argv; sys.argv = [sys.argv[0]]
import gbank as GB
sys.argv = _argv
SLABS = ["qkv", "proj", "fc", "mproj", "o", "gu", "down"]
LAYER, NPOS = 5, 48
GREL = {"qkv": 12, "proj": 11, "fc": 12, "mproj": 11}
def naf(a): a = np.abs(np.asarray(a, np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)

def main():
    Z = np.load(f"{HERE}/base2_gpt2.npz"); out = {}
    for kind, g in GREL.items():
        L = np.load(f"{HERE}/groups/light_gpt2/{kind}.npy", mmap_mode="r")
        pos = np.linspace(0, L.shape[1] - 1, NPOS).round().astype(int)
        x0 = np.asarray(L[LAYER, pos]).astype(np.int64); w = Z[f"{kind}.{LAYER}.w"].astype(np.int64)
        gp = GB.GPlate(w, [(kind, 0, w.shape[1])], g, [1, SLABS.index(kind), LAYER, g])
        for tag, x in (("real", x0), ("shuffle", np.random.default_rng([31337, SLABS.index(kind), LAYER]).permuted(x0, axis=1))):
            pile, C, parts = gp.read(x, keep_parts=True)
            assert np.array_equal(pile, x @ w)
            A = np.concatenate([a for a, _ in parts]); H = np.concatenate([h for _, h in parts])
            UPM = gp.UPM[None]                                         # (1, M, 18) the up/hole places
            added = int(np.where(UPM, H, 0).sum())                     # photons the gate adds at its up/hole places
            left_place = int(np.where(UPM, naf(A), 0).sum())           # what is left at each place after the groups pair off
            D = sum(A[:, :, b] << b for b in range(A.shape[2])) + gp.kcorr[None, :] * x.sum(1)[:, None]
            left_pile = int(naf(D).sum())                              # the doubled pile, after the places pair off too
            r = dict(added=added, left_place=left_place, left_pile=left_pile)
            out[f"{kind}.{LAYER}.{tag}"] = r
            print(f"{kind}.{LAYER} g={g} {tag:7s}: packets added {added:,}; left after the groups pair off at each place "
                  f"{left_place:,} ({100 * left_place / added:.2f}%); left in the whole pile {left_pile:,} "
                  f"({100 * left_pile / added:.3f}%)", flush=True)
    json.dump(out, open(f"{OUT}/flat_between.json", "w"), indent=1)

if __name__ == "__main__":
    main()
