"""PHOTONS family: the codings, each run on the SAME developed values as round one, each decoded back and checked
np.array_equal (so the words cannot change: 30/30 plates v12 A=8, 30/30 tables c=32, 29/30 tables c=34).
    python3 shrink/photons_versions.py v1 v2 ...      (versions below; each adds rows to photons_results.json)
Sizes: real bytes out of photons_rc.c + every piece of side information:
  plates: round one's own per-plate side bytes (header, neuron powers e_j, step terms t_j, line offsets off_i)
          = 231,072 bytes; the reader needs them anyway to put each stored number back at its place.
  tables: shape 12 + the 768 line cut places + 16 option bytes, + the 768 lens gains T is printed through (2,548).
Bits per photon = stored bits / photons in the writing that coder uses (single wave: up and holes; plain: 1s).
"""
import sys, time
from photons_lib import *

DATA = {}
def plates():
    if "plates" not in DATA: DATA["plates"] = read_round_one_plates()
    return DATA["plates"]
def tables(c):
    k = f"c{c}"
    if k not in DATA:
        Q, e, h = tables_Q(c); assert h == ROUND1_TABLE_HASH[c], (c, h); DATA[k] = (Q, e)
    return DATA[k]

PLATE_SIDE = None
def plate_side():
    return sum(len(p["head"]) for p in plates())
TABLE_SIDE = 12 + 768 + 16 + 2548
ROUND1_PLATES_FILE = 56742543
PH = {}
def photons_of(key):
    if key not in PH:
        if key == "plates":
            pp = pw = 0
            for p in plates(): a, b = count_photons(p["v"]); pp += a; pw += b
            PH[key] = (pp, pw)
        else: PH[key] = count_photons(tables(int(key[1:]))[0])
    return PH[key]

def run_plates(version, what, o, cls=None, shuffle=None):
    """cls: None, or 'residual' (the plates' exposure residual from t, off, A, known to the reader)"""
    t0 = time.time(); tot = 0; ok = True; by = {}
    for p in plates():
        v = p["v"]
        if shuffle is not None:                       # CONTROL: the plate's own numbers on random places
            rng = np.random.default_rng(shuffle + NAMES.index(p["name"]))
            v = rng.permutation(v.ravel()).reshape(v.shape).astype(np.int32)
        kw = {}
        if cls == "residual": kw = dict(colterm=p["t"], rowterm=p["off"])
        n, k = ph_store(v, o, **kw); tot += n; ok &= k
        kd = p["name"].split(".")[0]; by[kd] = by.get(kd, 0) + n
    side = plate_side(); total = tot + side; pp, pw = photons_of("plates"); nval = sum(p["v"].size for p in plates())
    rec = {"version": version, "data": "plates v12 A=8", "what": what, "opts": [int(x) for x in o[:11]],
           "coded_bytes": tot, "side_bytes": side, "total_bytes": total, "total_MB": round(total / 1e6, 3),
           "bits_per_value": round(8 * total / nval, 4), "roundtrip_ok": ok,
           "plain_photons_per_value": round(pp / nval, 4), "wave_photons_per_value": round(pw / nval, 4),
           "bits_per_photon_in_its_writing": round(8 * total / (pw if o[1] else pp), 4),
           "bits_per_wave_photon": round(8 * total / pw, 4), "bits_per_plain_photon": round(8 * total / pp, 4),
           "round_one_MB": round(ROUND1_PLATES_FILE / 1e6, 3), "minus_round_one_MB": round((total - ROUND1_PLATES_FILE) / 1e6, 3),
           "by_kind_MB": {k: round(b / 1e6, 3) for k, b in by.items()}, "shuffle_control": shuffle, "secs": round(time.time() - t0, 1)}
    log(rec); return rec

def run_tables(version, what, o, c, cls=None, shuffle=None, dark=None):
    """tables: rows = words, columns = lines. cls 'line': each line its own class (the reader knows the line).
    dark: subtract each line's most common stored number first (the dark frame, 768 numbers of side info)."""
    t0 = time.time(); Q, e = tables(c); V = Q; extra = 0
    if shuffle is not None:                           # CONTROL: each line's numbers moved to random words
        rng = np.random.default_rng(shuffle); V = np.empty_like(Q)
        for j in range(Q.shape[1]): V[:, j] = rng.permutation(Q[:, j])
    if dark is not None:
        V = V.copy() if V is Q else V
        d = np.zeros(Q.shape[1], np.int32)
        for j in range(Q.shape[1]):
            vals, cnt = np.unique(V[:, j], return_counts=True); d[j] = vals[np.argmax(cnt)]
        V -= d[None, :]; extra = 2 * 768
    kw = dict(colcls=np.arange(Q.shape[1])) if cls == "line" else {}
    n, ok = ph_store(V, o, **kw)
    if dark is not None:                              # the stored numbers come back with the dark frame added
        ok &= True
    side = TABLE_SIDE + extra; total = n + side; pp, pw = count_photons(V) if (dark is not None) else photons_of(f"c{c}")
    nval = Q.size; r1 = ROUND1_TABLE_BYTES[c]
    rec = {"version": version, "data": f"tables c={c}", "what": what, "opts": [int(x) for x in o[:11]],
           "coded_bytes": n, "side_bytes": side, "total_bytes": total, "total_MB": round(total / 1e6, 3),
           "bits_per_value": round(8 * total / nval, 4), "roundtrip_ok": ok,
           "plain_photons_per_value": round(pp / nval, 4), "wave_photons_per_value": round(pw / nval, 4),
           "bits_per_photon_in_its_writing": round(8 * total / (pw if o[1] else pp), 4),
           "bits_per_wave_photon": round(8 * total / pw, 4), "bits_per_plain_photon": round(8 * total / pp, 4),
           "round_one_MB": round(r1 / 1e6, 3), "minus_round_one_MB": round((total - r1) / 1e6, 3),
           "shuffle_control": shuffle, "secs": round(time.time() - t0, 1)}
    if V is not Q: del V
    log(rec); return rec

def both(version, what, op, ot, plates_kw=None, tables_kw=None, cs=(32, 34)):
    run_plates(version, what, op, **(plates_kw or {}))
    for c in cs: run_tables(version, what, ot, c, **(tables_kw or {}))

V = {
  # round one's coders, re-run here on the same values (plates: byte-for-byte round one's plate coder)
  "v0": lambda: both("v0", "round one's fax photon coder (single wave, row+column meters)", opts("fax", 1, 1), opts("fax", 1, 1)),
  # 1. positions with no learning: Elias gamma
  "v1": lambda: (both("v1 wave", "Elias gamma: top place, raw sign, each next photon's distance (no learning)",
                      opts("elias", 1, 0), opts("elias", 1, 0)),
                 both("v1 plain", "Elias gamma, plain base 2 photons (no learning)", opts("elias", 0, 0), opts("elias", 0, 0))),
  # 2. Golomb-Rice the camera way (JPEG-LS running k), top place against the two meters
  "v2": lambda: (both("v2 wave", "Rice: top place minus the meters' reading, runs between photons; k from running counts",
                      opts("rice", 1, 1), opts("rice", 1, 1)),
                 both("v2 plain", "Rice, plain base 2 photons", opts("rice", 0, 1), opts("rice", 0, 1))),
  # 3. fax runs + adaptive range coder, plain base 2 instead of the single wave
  "v3": lambda: both("v3 plain", "fax runs + range coder, PLAIN base 2 photons (row+column meters)", opts("fax", 0, 1), opts("fax", 0, 1)),
  # 4. the whole photon pattern as one symbol (binary tree of the pattern, range coder, meters)
  "v4": lambda: (run_plates("v4 pattern", "whole pattern as one symbol (8-place tree, escape to photons), meters", opts("pattern", 1, 1, tbits=8)),
                 [run_tables("v4 pattern", "whole pattern as one symbol (10-place tree, escape), meters", opts("pattern", 1, 1, tbits=10), c) for c in (32, 34)]),
  # 5. the top photon read against what the reader already knows about the exposure
  "v5p": lambda: (run_plates("v5 residual", "fax wave + exposure residual of the stored step (5 classes) x meters", opts("fax", 1, 1, 2, 5, 8), cls="residual"),
                  run_plates("v5 two meters", "fax wave, row and column meters apart (8 x 8)", opts("fax", 1, 2)),
                  run_plates("v5 two meters + residual", "fax wave, meters apart x exposure residual", opts("fax", 1, 2, 2, 5, 8), cls="residual")),
  "v5t": lambda: [(run_tables("v5 line", "fax wave, top place and sign per line (768 classes), no meter", opts("fax", 1, 0, 1, 768, signcls=1), c, cls="line"),
                   run_tables("v5 line + word meter", "fax wave, per line x word meter, sign per line", opts("fax", 1, 3, 1, 768, signcls=1), c, cls="line"),
                   run_tables("v5 pattern line", "whole pattern (10-place tree) per line, no meter", opts("pattern", 1, 0, 1, 768, tbits=10), c, cls="line"))
                  for c in (32, 34)],
  # controls: the same values on random places (plates: anywhere in the plate; tables: each line's numbers to random words)
  "c": lambda: (run_plates("control v0 shuffled", "round one's coder on each plate's own numbers put on random places", opts("fax", 1, 1), shuffle=1000),
                run_plates("control v5 residual shuffled", "v5 residual coder, numbers on random places (residual stays put)", opts("fax", 1, 1, 2, 5, 8), cls="residual", shuffle=1000),
                [run_tables("control v5 pattern line shuffled", "pattern per line, each line's numbers moved to random words", opts("pattern", 1, 0, 1, 768, tbits=10), c, cls="line", shuffle=2000) for c in (32,)],
                [run_tables("control v5 line + word meter shuffled", "fax per line x word meter, line numbers moved to random words", opts("fax", 1, 3, 1, 768, signcls=1), c, cls="line", shuffle=2000) for c in (32,)]),
  # is the tables' gain the coder's learning rule? round one's table coder moved every chance by 1/32 (fast, noisy)
  "lim5": lambda: [run_tables("check pattern line, chance never slower than 1/32", "v5 pattern line with the floor of the chance's step at 1/32 (like round one's coder)", opts("pattern", 1, 0, 1, 768, tbits=10, lim=5), c, cls="line") for c in (32,)],
  # 6. the same knowledge, the chances allowed to settle (a plate does not drift: the step floor 1/2^16), the pattern
  #    as one symbol with the exposure residual, the photons under the top read with the residual too
  "v6p": lambda: (run_plates("v6 residual, settled chances", "v5 residual, chance steps down to 1/65536", opts("fax", 1, 1, 2, 5, 8, lim=16), cls="residual"),
                  run_plates("v6 pattern + residual", "whole pattern (8-place tree) x meters x exposure residual", opts("pattern", 1, 1, 2, 5, 8, tbits=8), cls="residual"),
                  run_plates("v6 residual + under-top by residual", "v5 residual, photons under the top also read by the residual", opts("fax", 1, 1, 2, 5, 8, lowcls=1), cls="residual")),
  "v6t": lambda: [run_tables("v6 pattern line, settled chances", "v5 pattern line, chance steps down to 1/65536", opts("pattern", 1, 0, 1, 768, tbits=10, lim=16), c, cls="line") for c in (32, 34)],
  # 7. the photons as pages, one page per place from the top down (fax: page by page, line by line), each value's
  #    "lit here?" read against its left and upper neighbours on the same page (the page's own texture)
  "v7": lambda: (run_plates("v7 pages", "photon pages from the top place down, neighbours on the page", opts("pages", 1, 0)),
                 run_plates("v7 pages + residual", "photon pages x exposure residual", opts("pages", 1, 0, 2, 5, 8), cls="residual"),
                 run_plates("v7 pages plain", "plain base 2 photon pages", opts("pages", 0, 0)),
                 [run_tables("v7 pages line", "photon pages, start and sign per line, neighbours on the page", opts("pages", 1, 0, 1, 768, signcls=1), c, cls="line") for c in (32, 34)]),
}

if __name__ == "__main__":
    for name in sys.argv[1:]: V[name]()
