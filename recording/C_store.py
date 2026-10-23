"""PHASE C, STEP 3(a): STORAGE OF THE RECORDING (read from recording/C/ alone).
  RAW: the bytes as written: the up and down sheets (17 places x 2 sheets, 1 bit each a weight), the order, the powers.
  PACKED: the same sheets coded by the best exact coder already used in shrink/: planes_rc.c, the planes family's
    adaptive binary range coder on bit planes (whole numbers only), with the contexts of its version 9 (higher places
    of the same value + the row's and the column's exposure meters, M_HIGH | M_WEXP | M_LEXP), which gave the 153.06 MB
    exact floor of the 48 plates (shrink/planes_out_v9_plates.json; shrink/RESULTS.md: 153.1 MB). Its single-wave
    entry (naf_planes, used in planes v4a) codes exactly these sheets: place by place from the top, lit? then up or
    down? (a lit place right above forces this one dark and is not coded). The library shrink/planes_rc.dylib is
    loaded as it is (not rebuilt). Coded: each plate's sheets in the order they are stored (line-up rank in rows,
    neurons in columns); decoded back and checked equal.
  THE ORDER: raw as written (10 bits an entry for 768 lines, 12 for 3,072), and its exact floor
    log2(lines! / product over equal weights of count!) per neuron, in whole numbers (Python integers), which no
    coder of the order can go below.
CONTROLS on the 48 plates: (i) the same coder on the same sheets in the plate's own line order (no line-up);
  (ii) SHUFFLE: the plate's values shuffled inside the plate (same values, every pairing broken), then lined up and
  coded the same way.
PREDICTIONS (written before the run)
  The task's: raw sheets larger than 16 bits a weight; packed sheets close to the 153 MB floor, not below it.
  Claude's:
  - raw sheets exactly 34 bits a weight: 360.97 MB for the 48 plates; the order 10.67 bits a weight (113.3 MB);
    raw total about 474 MB against 169.9 MB plain.
  - packed sheets in line-up order: 9-12 bits a weight (95-127 MB). In rank order the top place and the sign of each
    weight are nearly given by the rank, but every place below about the third under the top stays a coin flip for
    this coder, whose contexts never look at the neighbour's same place.
  - the order: floor about 8.1 bits a weight on 768-line plates and 10.1 on 3,072-line plates, about 93 MB in all.
  - total packed sheets + order: 190-240 MB, above the 153.1 MB floor and above the plain 169.9 MB. The information
    does not shrink; the line-up moves part of it from the sheets into the order and the coder cannot take it back.
  - control (i), the plate's own line order: 152-156 MB, at the floor (version 4a: single-wave coding costs what
    plain planes cost).
  - control (ii), values shuffled then lined up: within 2% of the real line-up's packed sheets (in rank order the
    sheets are set mostly by the plate's spread of values).
    python3 recording/C_store.py part=0 ... part=4 [controls=0]   -> recording/C_logs/store_part{k}.json, C_store_part{k}.log
    python3 recording/C_store.py join                             -> recording/C_store.log, recording/C_logs/store.json
"""
import sys, os, json, time, math, ctypes, resource, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; CDIR = f"{HERE}/recording/C"; PLACES = 17
ARGS = dict(a.split("=") for a in sys.argv[1:] if "=" in a); CONTROLS = ARGS.get("controls", "1") == "1"
JOIN = "join" in sys.argv[1:]; PART = int(ARGS.get("part", -1))  # part=0..4 (12 plates each, the top table last), then join
LOG = open(f"{HERE}/recording/C_store{'' if JOIN else f'_part{PART}'}.log", "w"); t0 = time.time()
def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
MAN = json.load(open(f"{CDIR}/manifest.json"))
BANKS = [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")]
M_HIGH, M_WEXP, M_LEXP = 1, 32, 64; MODE = M_HIGH | M_WEXP | M_LEXP          # planes version 9's contexts
LIB = ctypes.CDLL(f"{HERE}/shrink/planes_rc.dylib")
_P, _I, _i = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
LIB.naf_planes.argtypes = [_i, _P, _I, _I, _i, _i, _P, _I, _P]; LIB.naf_planes.restype = _I
SIDE_BITS = 2 * 32 + 5 + 16                                  # per coded image: its shape, its number of places, the mode

def code(A):
    """the planes coder's single-wave entry, both ways. Returns (bytes coded, decoded back equal)."""
    A = np.ascontiguousarray(A, np.int32); R, C = A.shape; nb = int(np.abs(A).max()).bit_length() + 1
    cap = A.size * 3 + (1 << 20); buf = np.zeros(cap, np.uint8); mark = np.zeros(nb + 1, np.int64)
    n = LIB.naf_planes(0, A.ctypes.data, R, C, nb, MODE, buf.ctypes.data, cap, mark.ctypes.data); assert n <= cap
    data = buf[:n].copy(); del buf
    Bk = np.zeros_like(A); m2 = np.zeros(nb + 1, np.int64)
    LIB.naf_planes(1, Bk.ctypes.data, R, C, nb, MODE, data.ctypes.data, n, m2.ctypes.data)
    return int(n), bool(np.array_equal(Bk, A))

def signed_digits(v):
    v = np.asarray(v, np.int64); a = np.abs(v); t = 3 * a; up, dn = (t & ~a) >> 1, (a & ~t) >> 1; neg = v < 0
    return np.where(neg, dn, up), np.where(neg, up, dn)
def unpack_order(buf, lines, neurons, nb):
    n = lines * neurons; out = np.empty(n, np.int64); sh = np.arange(nb - 1, -1, -1, dtype=np.int64); step = 1 << 20
    for i in range(0, n, step):
        m = min(step, n - i); b0 = i * nb // 8
        bits = np.unpackbits(buf[b0:b0 + (m * nb + 7) // 8])[:m * nb].reshape(m, nb).astype(np.int64)
        out[i:i + m] = (bits << sh[None, :]).sum(1)
    return out.reshape(lines, neurons)
def stored(name, need_order=True):                           # the sheets as stored -> the weights in line-up rank order
    r = MAN["plates"][name]; L, N = r["lines"], r["neurons"]
    up = np.fromfile(f"{CDIR}/{name}.up", np.uint8).reshape(PLACES, L // 8, N)
    dn = np.fromfile(f"{CDIR}/{name}.down", np.uint8).reshape(PLACES, L // 8, N)
    ws = np.empty((L, N), np.int32)
    for c0 in range(0, N, 8192):
        sl = slice(c0, c0 + 8192); n = min(8192, N - c0)
        w = np.zeros((L, n), np.int64); U = np.zeros((L, n), np.int64); D = np.zeros((L, n), np.int64)
        for p in range(PLACES):
            u, d = np.unpackbits(up[p][:, sl], axis=0).astype(np.int64), np.unpackbits(dn[p][:, sl], axis=0).astype(np.int64)
            w += (u - d) << p; U |= u << p; D |= d << p
        su, sd = signed_digits(w); assert np.array_equal(su, U) and np.array_equal(sd, D)   # the coder's writing = the sheets
        ws[:, sl] = w
    o = unpack_order(np.fromfile(f"{CDIR}/{name}.order", np.uint8), L, N, r["order_bits"]) if need_order else None
    return ws, o
FACT = [1]
for i in range(1, 3073): FACT.append(FACT[-1] * i)
def order_floor_bits(ws):                                    # sum over neurons of log2(L! / prod of ties!), whole numbers
    L, N = ws.shape; tot = 0
    brk = np.concatenate([np.ones((1, N), bool), ws[1:] != ws[:-1]], 0)            # a new value starts (rank order)
    for n in range(N):
        starts = np.flatnonzero(brk[:, n]); runs = np.diff(np.append(starts, L)); den = 1
        for t in runs[runs > 1].tolist(): den *= FACT[t]
        tot += (FACT[L] // den).bit_length() - 1                                    # floor of log2 (the floor of the floor)
    return tot

res = dict(plates={}, controls={}); acc = dict(weights=0, raw_sheets=0, raw_order=0, raw_power=0, packed=0, order_floor_bits=0,
                                               natural=0, shuffled=0)
ALL = BANKS + ["head"]; PARTS = [ALL[0:12], ALL[12:24], ALL[24:36], ALL[36:48], ALL[48:]]
if JOIN:
    for k in range(len(PARTS)): res["plates"].update(json.load(open(f"{HERE}/recording/C_logs/store_part{k}.json")))
    assert sorted(res["plates"]) == sorted(ALL)
for name in (ALL if JOIN else PARTS[PART]):
    r = MAN["plates"][name]; L, N = r["lines"], r["neurons"]; f = r["files"]
    if JOIN:
        row = res["plates"][name]
        if name != "head":
            acc["weights"] += L * N; acc["raw_sheets"] += row["raw_sheets"]; acc["raw_order"] += row["raw_order"]
            acc["raw_power"] += row["raw_power"]; acc["packed"] += row["packed_sheets"]; acc["order_floor_bits"] += row["order_floor_bits"]
            acc["natural"] += row.get("natural_order_packed", 0); acc["shuffled"] += row.get("shuffled_packed", 0)
        continue
    rng = np.random.default_rng(17 + ALL.index(name))
    ws, o = stored(name, need_order=CONTROLS and name != "head")
    nbytes, ok = code(ws); assert ok, name
    ofl = order_floor_bits(ws)
    row = dict(weights=L * N, raw_sheets=f["up"]["bytes"] + f["down"]["bytes"], raw_order=f["order"]["bytes"],
               raw_power=f["power"]["bytes"], packed_sheets=nbytes + (SIDE_BITS + 7) // 8, roundtrip=ok, order_floor_bits=ofl)
    if CONTROLS and name != "head":
        nat = np.zeros_like(ws); np.put_along_axis(nat, o, ws, 0)                # each weight back on its own line
        row["natural_order_packed"], ok1 = code(nat); assert ok1
        sh = rng.permutation(ws.reshape(-1)).reshape(L, N)                        # SHUFFLE: same values, any neuron
        shs = -np.sort(-sh, axis=0)                                               # then lined up, largest first
        row["shuffled_packed"], ok2 = code(shs); assert ok2
        row["natural_order_packed"] += (SIDE_BITS + 7) // 8; row["shuffled_packed"] += (SIDE_BITS + 7) // 8
        del nat, sh, shs
    res["plates"][name] = row
    b = lambda x: 8 * x / (L * N)
    log(f"  {name:9s} raw sheets {b(row['raw_sheets']):5.2f} b/w | packed sheets {b(row['packed_sheets']):6.3f} b/w (round trip {ok}) | "
        f"order raw {b(row['raw_order']):5.2f}, floor {ofl / (L * N):5.2f} b/w"
        + (f" | CONTROLS: own line order {b(row['natural_order_packed']):6.3f}, values shuffled {b(row['shuffled_packed']):6.3f}" if 'shuffled_packed' in row else "")
        + f"  ({time.time() - t0:.0f}s, {rss()} MB)")
    if name != "head":
        acc["weights"] += L * N; acc["raw_sheets"] += row["raw_sheets"]; acc["raw_order"] += row["raw_order"]
        acc["raw_power"] += row["raw_power"]; acc["packed"] += row["packed_sheets"]; acc["order_floor_bits"] += ofl
        acc["natural"] += row.get("natural_order_packed", 0); acc["shuffled"] += row.get("shuffled_packed", 0)
    del ws, o
if not JOIN:
    json.dump(res["plates"], open(f"{HERE}/recording/C_logs/store_part{PART}.json", "w"), indent=1)
    log(f"saved recording/C_logs/store_part{PART}.json ({time.time() - t0:.0f}s, max memory {rss()} MB)"); sys.exit()
n = acc["weights"]; MB = lambda x: x / 1e6
log(f"\nTHE 48 PLATES ({n:,} weights; plain 16-bit {MB(2 * n):.2f} MB; exact floor shrink/RESULTS.md 153.1 MB)")
log(f"  RAW:    sheets {MB(acc['raw_sheets']):.2f} MB ({8 * acc['raw_sheets'] / n:.2f} bits a weight); order {MB(acc['raw_order']):.2f} MB "
    f"({8 * acc['raw_order'] / n:.2f}); powers {MB(acc['raw_power']):.3f} MB; total {MB(acc['raw_sheets'] + acc['raw_order'] + acc['raw_power']):.2f} MB")
log(f"  PACKED: sheets {MB(acc['packed']):.2f} MB ({8 * acc['packed'] / n:.3f} bits a weight; planes coder, single-wave entry, v9 contexts)")
log(f"          + order raw {MB(acc['raw_order']):.2f} MB + powers {MB(acc['raw_power']):.3f} MB = {MB(acc['packed'] + acc['raw_order'] + acc['raw_power']):.2f} MB")
log(f"          + order at its floor {MB(acc['order_floor_bits'] / 8):.2f} MB ({acc['order_floor_bits'] / n:.2f} bits a weight) instead: "
    f"{MB(acc['packed'] + acc['order_floor_bits'] / 8 + acc['raw_power']):.2f} MB")
if CONTROLS:
    log(f"  CONTROL (i) the same coder on the same sheets in each plate's own line order (no line-up, no order stored): "
        f"{MB(acc['natural']):.2f} MB ({8 * acc['natural'] / n:.3f} bits a weight) + powers = {MB(acc['natural'] + acc['raw_power']):.2f} MB")
    log(f"  CONTROL (ii) SHUFFLE, values shuffled inside each plate, then lined up: sheets {MB(acc['shuffled']):.2f} MB "
        f"({8 * acc['shuffled'] / n:.3f} bits a weight), against {MB(acc['packed']):.2f} MB for the real line-up")
h = res["plates"]["head"]; hw = h["weights"]
log(f"\nTHE TOP TABLE ({hw:,} weights; plain 16-bit {MB(2 * hw):.2f} MB; shrink: T alone 54.8 MB, T given the entry table 0.34 MB)")
log(f"  RAW: sheets {MB(h['raw_sheets']):.2f} MB, order {MB(h['raw_order']):.2f} MB, power {h['raw_power']} bytes; "
    f"PACKED sheets {MB(h['packed_sheets']):.2f} MB ({8 * h['packed_sheets'] / hw:.3f} bits a weight); order floor {MB(h['order_floor_bits'] / 8):.2f} MB")
res["totals"] = acc; json.dump(res, open(f"{HERE}/recording/C_logs/store.json", "w"), indent=1)
log(f"saved recording/C_logs/store.json ({time.time() - t0:.0f}s, max memory {rss()} MB)")
