"""BUILD THE ENTRY FILES FOR THE WHOLE VOCABULARY: the stored meetings with the qkv.0 plate for every word and every
position, at every shift s the water level can give there (entry/survey_s.py found them, over all 51,463,168 pairs):
    s = 4  only at position 0, for every word
    s = 2  at positions 1..1023, every word
    s = 1  at 663 positions (7..1023), 50,121 words
    s = 0  only at position 1023, 17,994 words
    s = 3  never

LAYOUT (chosen as the cheapest exact one; see the manifest's "why"):
  ZP0[w]     the whole qkv.0 pile of word w at position 0:     rshift(wte[w] + wpe[0], 4) @ plate     (all 50,257 words)
  Z1023[w]   the whole qkv.0 pile of word w at position 1023 when s = 0:  (wte[w] + wpe[1023]) @ plate (17,994 words)
             (s = 4 and s = 0 each belong to one position only, so the word's rounded row there is only ever met
             together with that one position's row: storing the pair's pile costs the same one row per word as the
             pilot's word tip, and leaves no carry to meet.)
  T2[w]      rshift(wte[w], 2) @ plate      (all 50,257 words)        P2[p]  rshift(wpe[p], 2) @ plate  (positions 1..1023)
  T1[w]      rshift(wte[w], 1) @ plate      (50,121 words)            P1[p]  rshift(wpe[p], 1) @ plate  (663 positions)
             per token at s = 1 or 2: pile = T_s[w] + P_s[p] + the carry (-1/0/+1 per line) met with the plate.
  Each table has an ids file (which words / positions it holds, int32). Its values are whole numbers stored at the
  fewest bytes that hold them: int32 when every value is below 2^31; otherwise a 5-byte split, lo (uint32, the low
  32 digits) and hi (int8, the rest: value = hi x 2^32 + lo) when below 2^39; otherwise int64.

EXACTNESS OF THE BUILD: the meetings are float64 products of whole numbers. Every row met here is below 2^19 in size
and every plate column's sum of sizes is below 768 x 2^15 < 2^25, so every partial sum is below 2^44 < 2^53: exact in
any order. Checked as numbers below, not assumed. Built in chunks of 4,096 rows; memory stays under 1 GB.
    python3 entry/build.py
"""
import os, sys, json, time, hashlib, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")    # Accelerate's spurious flags; the bound is checked
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/entry"
# base2.py's own rounding (rshift), read from its source, with nothing else of the machine loaded
_src = open(f"{HERE}/base2.py").read(); _i = _src.index("def rshift"); _j = _src.index("def to16")
NS = {"np": np}; exec(_src[_i:_j], NS); rshift = NS["rshift"]

t0 = time.time()
M = np.load(f"{HERE}/base2_gpt2.npz")
wte, wpe = M["wte"], M["wpe"]                          # int64 at 2^16
plate16 = M["qkv.0.w"]; PLATE = plate16.astype(np.float64)   # (768 lines, 2304 neurons)
S = np.load(f"{OUT}/survey_s.npz"); count, wmask, pmask = S["count"], S["wmask"], S["pmask"]
assert count.sum() == 50257 * 1024 and count[3].sum() == 0 and count[5:].sum() == 0
assert set(np.flatnonzero(count[4]).tolist()) == {0} and set(np.flatnonzero(count[0]).tolist()) == {1023}
COLSUM = np.abs(plate16.astype(np.int64)).sum(0).max()       # the largest column's sum of sizes
s0_words = np.flatnonzero((wmask >> 0) & 1)
TABLES = {   # name: (ids, light rows for a chunk of ids)
    "ZP0":   (np.arange(50257),               lambda ids: rshift(wte[ids] + wpe[0][None, :], 4)),
    "Z1023": (s0_words,                       lambda ids: wte[ids] + wpe[1023][None, :]),
    "T2":    (np.flatnonzero((wmask >> 2) & 1), lambda ids: rshift(wte[ids], 2)),
    "T1":    (np.flatnonzero((wmask >> 1) & 1), lambda ids: rshift(wte[ids], 1)),
    "P2":    (np.flatnonzero((pmask >> 2) & 1), lambda ids: rshift(wpe[ids], 2)),
    "P1":    (np.flatnonzero((pmask >> 1) & 1), lambda ids: rshift(wpe[ids], 1)),
}
assert len(TABLES["ZP0"][0]) == 50257 and len(TABLES["T2"][0]) == 50257
assert TABLES["P2"][0].tolist() == list(range(1, 1024))      # s = 2 at every position but 0
# Z1023 at s = 0: the light is not rounded at all, and every one of its lines is below 2^15 (that is what s = 0 means)
CH = 4096
def meet(ids, light):                                  # whole-number meetings of a chunk of rows with the plate
    X = light(ids); big = int(np.abs(X).max())
    assert big * COLSUM < 2 ** 53, "partial sums would not be exact"
    return (X.astype(np.float64) @ PLATE).astype(np.int64), big
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""): h.update(blk)
    return h.hexdigest()
manifest = {"what": "qkv.0 entry for every word and position (see entry/build.py docstring and base2_entry.py)",
            "plate": "base2_gpt2.npz qkv.0.w (int16, 768 x 2304)", "neurons": 2304, "files": {}, "tables": {}}
def add_file(path, arr_meta):
    manifest["files"][os.path.basename(path)] = dict(arr_meta, bytes=os.path.getsize(path), sha256=sha(path))
for name, (ids, light) in TABLES.items():
    t1 = time.time(); ids = np.asarray(ids, np.int64)
    # pass 1: the largest value (sets the bytes per value)
    top = 0; bigl = 0
    for i in range(0, len(ids), CH):
        v, big = meet(ids[i:i + CH], light); top = max(top, int(np.abs(v).max())); bigl = max(bigl, big)
    if top < 2 ** 31: kind = "int32"
    elif top < 2 ** 39: kind = "split40"
    else: kind = "int64"
    # pass 2: write
    idp = f"{OUT}/{name}_ids.npy"; np.save(idp, ids.astype(np.int32)); add_file(idp, {"shape": [len(ids)], "dtype": "int32"})
    shape = (len(ids), 2304)
    if kind == "int32": outs = {"v": np.lib.format.open_memmap(f"{OUT}/{name}.npy", "w+", np.int32, shape)}
    elif kind == "int64": outs = {"v": np.lib.format.open_memmap(f"{OUT}/{name}.npy", "w+", np.int64, shape)}
    else: outs = {"lo": np.lib.format.open_memmap(f"{OUT}/{name}_lo.npy", "w+", np.uint32, shape),
                  "hi": np.lib.format.open_memmap(f"{OUT}/{name}_hi.npy", "w+", np.int8, shape)}
    for i in range(0, len(ids), CH):
        v, _ = meet(ids[i:i + CH], light)
        if kind == "split40":
            hi = np.right_shift(v, 32); lo = np.bitwise_and(v, (1 << 32) - 1)
            assert hi.min() >= -128 and hi.max() <= 127
            outs["hi"][i:i + CH] = hi.astype(np.int8); outs["lo"][i:i + CH] = lo.astype(np.uint32)
            assert np.array_equal(hi.astype(np.int8).astype(np.int64) * (1 << 32) + lo.astype(np.uint32).astype(np.int64), v)
        else:
            outs["v"][i:i + CH] = v.astype(outs["v"].dtype)
    files = []
    for part, arr in outs.items():
        arr.flush(); files.append((os.path.basename(arr.filename), str(arr.dtype)))
    outs.clear()
    for f, dt in files: add_file(f"{OUT}/{f}", {"shape": list(shape), "dtype": dt})
    files = [f for f, _ in files]
    manifest["tables"][name] = {"rows": len(ids), "kind": kind, "files": files, "ids": os.path.basename(idp),
                                "largest value": top, "bits": top.bit_length() + 1, "largest light line": bigl}
    print(f"  {name:6s} {len(ids):6,} rows  largest value {top:,} ({top.bit_length() + 1} bits signed) -> {kind}  "
          f"largest light {bigl:,}  ({time.time() - t1:.0f}s, {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB)", flush=True)
manifest["survey"] = {"pairs": 50257 * 1024,
                      "s": {str(v): {"pairs": int(count[v].sum()), "words": int(((wmask >> v) & 1).sum()),
                                     "positions": int((count[v] > 0).sum())} for v in range(8) if count[v].sum()},
                      "never": [v for v in range(5) if count[v].sum() == 0]}
manifest["exactness bound"] = f"largest light line x largest plate column sum of sizes ({COLSUM:,}) < 2^53 checked per chunk"
add_file(f"{OUT}/survey_s.npz", {"shape": "count (8,1024) int64, wmask (50257,) uint8, pmask (1024,) uint8", "dtype": "npz"})
manifest["bytes on disk, tables and ids"] = sum(f["bytes"] for k, f in manifest["files"].items() if k != "survey_s.npz")
manifest["why"] = ("Chosen: rounded-row tips (the pilot's layout) at s = 1, 2, and the whole pair pile at the two "
                   "one-position shifts (s = 4 at position 0, s = 0 at position 1023). Per token only the carry, lit "
                   "on about a quarter of the lines with one photon each, meets the plate (nothing at positions 0 and "
                   "1023 with s = 0). The alternative, one unrounded tip per word and position with the residue "
                   "(s digits on every line) meeting the plate per token, needs fewer bytes but meets more light per "
                   "token; entry/measure.py prints both.")
json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1)
print(f"\nbytes on disk (tables + ids): {manifest['bytes on disk, tables and ids']:,}")
for k, f in manifest["files"].items(): print(f"  {k:22s} {str(f['shape']):>16s} {f['dtype']:8s} {f['bytes']:>14,}  {f['sha256'][:16]}")
print(f"max memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB, {time.time() - t0:.0f}s")
