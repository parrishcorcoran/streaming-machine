"""LAYER 0, STEP 2: BUILD THE STORED TABLES (layer0/, manifest in layer0/manifest.json).

What at layer 0 is fixed by (word, position), and what of it is stored here:
  Z0[w]        position 0, every word: the WHOLE stream after layer 0 (768 values). At position 0 nothing arrives
               but the token itself, so every pile of layer 0 there is fixed by the word. Built by running base2.py's
               own layer 0 (its functions, through judge5) on each word alone. All 50,257 words.
  FT5[w] FT6[w]   fc.0, every word: rshift(wte[w], s1) met with the fc.0 plate, s1 = 5 and 6 (layer0/survey.py: the
               only shifts fc.0's water level gives on the 30 refs texts and the 300 judge windows). All 50,257 words.
  FP5[p] FP6[p]   fc.0, every position: rshift(wpe[p] + proj.0's bias, s1) met with the fc.0 plate. All 1,024.
               Per token at fc.0: pile = FT_s1[w] + FP_s1[p] + (x1 - the two rounded rows) met with the plate, where
               x1 - the two rows is the arrival write's part (rounded at s1) plus a -1/0/+1 carry.
  OWN1/2/3[(w, p)]  proj.0, the token's own value: rshift(v(w, p), so) met with the proj.0 plate, so = 1, 2, 3 (the
               survey's shifts of the arrival's output). Per token at proj.0: pile = OWN_so[(w, p)] + (o16 - the own
               value rounded at so) met with the plate (the pilot's V6). This table is per PAIR: for the whole
               vocabulary it would be 51,412,911 pairs x 768 values x 3 shifts, far past the disk, so it is built
               ONLY for the pairs (positions 1 and on) of the 30 refs texts and the 300 judge windows; a pair not in
               it meets the whole light at proj.0, exactly as base2.py does.
What is fixed by the pair but NOT stored: q, k, v themselves (51,463,168 pairs x 2,304 values: about 593 GB); they
come from step 1's tables (entry/) with the -1/0/+1 carry met per token.

RULE for whole-vocabulary tables (written before building): built when they fit (this step writes at most 6 GB; the
disk has about 15 GB free) and pay (on the refs texts the pilot's split lowered the per-token meetings at the plate,
stored adds and the split counted: fc.0 61,305,666 -> 58,303,923). Z0 and FT/FP fit and pay; OWN does not fit.

Formats as entry/build.py: int32 when every value is below 2^31; a 5-byte split (lo uint32 + hi int8) below 2^39;
otherwise int64. Every meeting is a float64 product of whole numbers with every partial sum below 2^53 (checked per
chunk as numbers, not assumed).

PREDICTION (the session's, before running): FT5/FT6 need 33-35 bits -> 5 bytes, about 772 MB each; FP5/FP6 about
16 MB each; Z0 fits int32 (the stream below 2^31), 154 MB; OWN needs 40-42 bits -> int64. Everything together under
2 GB. At position 0, over the whole vocabulary, fc.0's shift is 5 or 6 for every word.
    python3 layer0/build.py
"""
import os, sys, json, time, hashlib, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/layer0"; sys.path.insert(0, HERE); os.chdir(HERE)
import judge5 as J
M, W, ES = J.M, J.W, J.ES
t0 = time.time()
def gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
WTE, WPE = M["wte"], M["wpe"]; BP = W["proj.0"][2]
PF = M["fc.0.w"]; PP = M["proj.0.w"]                    # int16 plates (lines in, neurons out)
COL = {"fc": int(np.abs(PF.astype(np.int64)).sum(0).max()), "proj": int(np.abs(PP.astype(np.int64)).sum(0).max())}
REFS = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in REFS]
WIN = [w["ids"] for w in J.WINDOWS]
CH = 2048

def meet(X, plate, key):                               # whole-number rows met with an int16 plate, exact
    big = int(np.abs(X).max()); assert big * COL[key] < 2 ** 53, "partial sums would not be exact"
    return (X.astype(np.float64) @ plate.astype(np.float64)).astype(np.int64), big
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""): h.update(blk)
    return h.hexdigest()
manifest = {"what": "layer 0 step 2 tables (see layer0/build.py docstring and base2_layer0.py)", "files": {}, "tables": {}}
def add_file(path, meta): manifest["files"][os.path.basename(path)] = dict(meta, bytes=os.path.getsize(path), sha256=sha(path))

def write_table(name, ids, rows, note, values=None):
    """rows(ids_chunk) -> (int64 values, largest light line); values: the whole table already in hand (else two passes)"""
    t1 = time.time(); ids = np.asarray(ids, np.int64)
    top = 0; bigl = 0
    if values is None:
        for i in range(0, len(ids), CH):
            v, b = rows(ids[i:i + CH]); top = max(top, int(np.abs(v).max())); bigl = max(bigl, b)
    else: top = int(np.abs(values).max())
    kind = "int32" if top < 2 ** 31 else ("split40" if top < 2 ** 39 else "int64")
    idp = f"{OUT}/{name}_ids.npy"; np.save(idp, ids.astype(np.int32)); add_file(idp, {"shape": [len(ids)], "dtype": "int32"})
    ncol = (values if values is not None else rows(ids[:1])[0]).shape[1]; shape = (len(ids), ncol)
    if kind in ("int32", "int64"):
        outs = {"v": np.lib.format.open_memmap(f"{OUT}/{name}.npy", "w+", np.int32 if kind == "int32" else np.int64, shape)}
    else: outs = {"lo": np.lib.format.open_memmap(f"{OUT}/{name}_lo.npy", "w+", np.uint32, shape),
                  "hi": np.lib.format.open_memmap(f"{OUT}/{name}_hi.npy", "w+", np.int8, shape)}
    for i in range(0, len(ids), CH):
        v = values[i:i + CH] if values is not None else rows(ids[i:i + CH])[0]
        if kind == "split40":
            hi = np.right_shift(v, 32); lo = np.bitwise_and(v, (1 << 32) - 1); assert hi.min() >= -128 and hi.max() <= 127
            outs["hi"][i:i + CH] = hi.astype(np.int8); outs["lo"][i:i + CH] = lo.astype(np.uint32)
            assert np.array_equal(hi.astype(np.int8).astype(np.int64) * (1 << 32) + lo.astype(np.uint32).astype(np.int64), v)
        else: outs["v"][i:i + CH] = v.astype(outs["v"].dtype)
    files = []
    for part, arr in outs.items(): arr.flush(); files.append((os.path.basename(arr.filename), str(arr.dtype)))
    outs.clear()
    for f, dt in files: add_file(f"{OUT}/{f}", {"shape": list(shape), "dtype": dt})
    manifest["tables"][name] = {"rows": len(ids), "cols": ncol, "kind": kind, "files": [f for f, _ in files],
                                "ids": os.path.basename(idp), "largest value": top, "bits": top.bit_length() + 1,
                                "largest light line": bigl, "what": note}
    b = sum(manifest["files"][f]["bytes"] for f, _ in files) + manifest["files"][os.path.basename(idp)]["bytes"]
    print(f"  {name:5s} {len(ids):6,} rows x {ncol:,}  largest value {top:,} ({top.bit_length() + 1} bits signed) -> {kind}, "
          f"{b:,} bytes  ({time.time() - t1:.0f}s, {gb():.2f} GB)", flush=True)

# ---------------------------------------------------------------- Z0: the whole layer 0 at position 0, every word
print("Z0: base2.py's own layer 0 on every word alone at position 0", flush=True)
Z0 = np.zeros((50257, 768), np.int64); S1_0 = np.zeros(50257, np.int64); SO_0 = np.zeros(50257, np.int64); SM_0 = np.zeros(50257, np.int64)
for i in range(0, 50257, CH):
    ids = np.arange(i, min(i + CH, 50257)); R = WTE[ids] + WPE[0][None, :]           # each word alone, position 0
    QKV = J.gauged_bank(R, "qkv.0")                                                    # row by row (its own water level)
    o = np.concatenate([J.attention(QKV[j:j + 1]) for j in range(len(ids))])          # the arrival: each word alone
    SO_0[ids] = J.to16(o, -ES)[1] + ES
    R1 = R + J.plain_bank(o, "proj.0"); S1_0[ids] = J.gauge_log(R1)[1] + ES
    G = J.edge(J.gauged_bank(R1, "fc.0")); SM_0[ids] = J.to16(G, -ES)[1] + ES
    Z0[ids] = R1 + J.plain_bank(G, "mproj.0")
    if i % 16384 == 0: print(f"   words {i:,}..  ({time.time() - t0:.0f}s, {gb():.2f} GB)", flush=True)
for key, arr in (("fc.0 shift s1", S1_0), ("proj.0 shift so", SO_0), ("mproj.0 shift sm", SM_0)):
    v, c = np.unique(arr, return_counts=True)
    print(f"   position 0, all 50,257 words, {key}: " + ", ".join(f"{int(a)}: {int(b):,}" for a, b in zip(v, c)))
manifest["position 0 shifts, every word"] = {k: {str(int(a)): int(b) for a, b in zip(*np.unique(arr, return_counts=True))}
                                            for k, arr in (("s1", S1_0), ("so", SO_0), ("sm", SM_0))}
write_table("Z0", np.arange(50257), None, "the stream after layer 0 at position 0, word alone (base2.py's layer 0)", values=Z0)
del Z0

# ---------------------------------------------------------------- FT, FP: fc.0's word and position parts
for s1 in (5, 6):
    write_table(f"FT{s1}", np.arange(50257), lambda ids, s1=s1: meet(J.rshift(WTE[ids], s1), PF, "fc"),
                f"rshift(wte[w], {s1}) met with fc.0")
    write_table(f"FP{s1}", np.arange(1024), lambda ids, s1=s1: meet(J.rshift(WPE[ids] + BP[None, :], s1), PF, "fc"),
                f"rshift(wpe[p] + proj.0 bias, {s1}) met with fc.0")

# ---------------------------------------------------------------- OWN: proj.0's own-value tips, the texts' pairs only
pairs = sorted({int(w) * 1024 + p for t in TEXTS + WIN for p, w in enumerate(t) if p >= 1})
pairs = np.array(pairs, np.int64)
print(f"OWN: {len(pairs):,} (word, position) pairs at positions >= 1 in the 30 refs texts and the 300 judge windows", flush=True)
def own_rows(keys, so):
    w, p = keys // 1024, keys % 1024
    v = J.gauged_bank(WTE[w] + WPE[p], "qkv.0")[:, 1536:]                             # the pair's own value, row by row
    return meet(J.rshift(v, so), PP, "proj")
for so in (1, 2, 3):
    write_table(f"OWN{so}", pairs, lambda k, so=so: own_rows(k, so), f"rshift(v(w, p), {so}) met with proj.0; ids = w x 1024 + p")

bw = {n: manifest["files"][t["files"][0]]["bytes"] / t["rows"] + (manifest["files"][t["files"][1]]["bytes"] / t["rows"] if len(t["files"]) > 1 else 0)
      for n, t in manifest["tables"].items()}
manifest["whole vocabulary, not built"] = {
    "OWN (per pair, 3 shifts)": int(51412911 * 3 * bw["OWN2"]),
    "q, k, v per pair (2,304 values, 5 bytes)": 51463168 * 2304 * 5}
manifest["bytes on disk, tables and ids"] = sum(f["bytes"] for f in manifest["files"].values())
manifest["rule"] = ("whole-vocabulary tables built when they fit (at most 6 GB written in this step) and pay (lower the "
                    "per-token meetings at their plate, stored adds and the split counted). OWN is per pair: built for "
                    "the pairs of the 30 refs texts and the 300 judge windows only.")
json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1)
print(f"\nbytes on disk (tables + ids): {manifest['bytes on disk, tables and ids']:,}")
for k, f in manifest["files"].items(): print(f"  {k:16s} {str(f['shape']):>16s} {f['dtype']:8s} {f['bytes']:>14,}  {f['sha256'][:16]}")
for k, v in manifest["whole vocabulary, not built"].items(): print(f"  whole vocabulary, not built: {k}: {v:,} bytes")
print(f"max memory {gb():.2f} GB, {time.time() - t0:.0f}s")
