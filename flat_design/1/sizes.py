"""SIZES: the table sized by the gates that read it (PREDICTION.txt, written before this ran). Each plate of layer 5 is
read through groups2/gbank.py's group bank (the confirmed machine's own bank, unchanged) at the two sizes the relation
names for it, on the 48 positions of excess_sample.py. Every pile is checked against the direct whole-number product.
Not a sweep: the relation fixed g before the run (11 at the 768-gate plates, 12 at qkv and fc); the neighbour size is
run only to test that prediction.

    python3 flat_design/1/sizes.py  -> flat_design/1/sizes.json
"""
import sys, json, time, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/flat_design/1"
sys.path.insert(0, f"{HERE}/groups2")
_argv = sys.argv; sys.argv = [sys.argv[0]]
import gbank as GB
sys.argv = _argv
SLABS = ["qkv", "proj", "fc", "mproj", "o", "gu", "down"]
LAYER, NPOS = 5, 48
PAIRS = {"qkv": (12, 13), "proj": (11, 12), "fc": (12, 13), "mproj": (11, 12)}
def ones(a): return np.bitwise_count(np.abs(np.asarray(a, np.int64))).astype(np.int64)

def run(x, w, kind, g):
    gp = GB.GPlate(w, [(kind, 0, w.shape[1])], g, [1, SLABS.index(kind), LAYER, g])
    pile, C = gp.read(x)
    bad = int((pile != x.astype(np.int64) @ w.astype(np.int64)).any(1).sum())
    assert bad == 0, (kind, g, bad)
    tot = {k: int(v.sum()) for k, v in C.items()}
    return dict(items=tot, total=sum(tot.values()), up_places=bin(int(gp.up[0])).count("1"))

def main():
    t0 = time.time(); Z = np.load(f"{HERE}/base2_gpt2.npz"); out = {}
    for kind, gs in PAIRS.items():
        L = np.load(f"{HERE}/groups/light_gpt2/{kind}.npy", mmap_mode="r")
        pos = np.linspace(0, L.shape[1] - 1, NPOS).round().astype(int)
        x = np.asarray(L[LAYER, pos]).astype(np.int64); w = Z[f"{kind}.{LAYER}.w"].astype(np.int64)
        xs = np.random.default_rng([31337, SLABS.index(kind), LAYER]).permuted(x, axis=1)
        plain = int(ones(x).sum(0) @ ones(w).sum(1)); plain_s = int(ones(xs).sum(0) @ ones(w).sum(1))
        rec = dict(plain=plain, plain_shuffle=plain_s)
        for g in gs:
            r = run(x, w, kind, g); s = run(xs, w, kind, g)
            rec[f"g{g}"] = r; rec[f"g{g}_shuffle"] = s
            it = r["items"]; reads = sum(v for k, v in it.items() if k.startswith("reads|"))
            sd = sum(v for k, v in it.items() if k.startswith("sdreads|"))
            print(f"{kind}.{LAYER} g={g:2d}: {100 * r['total'] / plain:6.2f}% of plain (shuffle {100 * s['total'] / plain_s:6.2f}%) | "
                  f"reads {100 * reads / plain:5.2f} sd {100 * sd / plain:4.2f} table {100 * it['table'] / plain:4.2f} "
                  f"write {100 * it['pwrite'] / plain:4.2f} light {100 * it['light'] / plain:4.2f} | up/hole places {r['up_places']}",
                  flush=True)
        out[f"{kind}.{LAYER}"] = rec
    json.dump(out, open(f"{OUT}/sizes.json", "w"), indent=1)
    print(f"done ({time.time() - t0:.0f}s); every pile identical at every size, real and shuffled light")

if __name__ == "__main__":
    main()
