"""PHASE C, STEP 1: WRITE THE RECORDING. Move 1 alone (the efficient format), no turn (phase A: a single turn set by
the foundation does not pay, recording/why.log).

Every one of the 48 plates, and the top table, is written to files under recording/C/:
  - each weight in signed digits: 17 places (0..16), each place up, flat or down, no two lit places side by side;
    17 places allow every 16-place whole number (the plates are int16, so place 16 is dark on every weight here);
  - stored as an UP sheet and a DOWN sheet per place, packed bits over the lines (8 lines a byte, top line first);
    flat is dark on both sheets;
  - the lines of each neuron in its line-up order (largest weight first, ties in line order, as combined/machine.py's
    line-up has it); the order is stored, each entry in just enough bits for the plate's line count (10 bits for 768
    lines, 12 bits for 3,072);
  - each neuron's own power of two, as the machine has it (the plate's ".e").
Files per plate N (shapes in the manifest):
  N.up, N.down   uint8 (17, lines/8, neurons): sheet of place p, bytes of 8 lines (line-up rank), neuron
  N.order        packed bits: rank k, neuron n -> the line that sits at rank k (row-major, rank outer)
  N.power        the neuron's power of two, the plate's own dtype
Also written, unchanged, so the machine can run from this folder alone: rest.npz (every number of base2_gpt2.npz that
is not a plate or the top table: the entry and position tables, biases, the top's standing amounts and power, the
bend, arrival, log and power tables). manifest.json lists every file, its size and its sha256.

How the digits are found (whole numbers only): a = |v|, up = (3a AND NOT a) >> 1, down = (a AND NOT 3a) >> 1, and the
two swap where v is below zero. Checked on every weight: up AND down = 0, no two lit side by side, the sum over places
of 2^p (up - down) = the weight, and the files read back from disk give the plate exactly.
    python3 recording/C_write.py      -> recording/C/ and recording/C_write.log
"""
import os, sys, json, time, hashlib, resource, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/recording/C"
NPZ = f"{HERE}/base2_gpt2.npz"; PLACES = 17
BANKS = [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")]
LOG = open(f"{HERE}/recording/C_write.log", "w"); t0 = time.time()
def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""): h.update(blk)
    return h.hexdigest()

def signed_digits(v):                                        # v: whole numbers -> (up, down) as place masks
    v = np.asarray(v, np.int64); a = np.abs(v); t = 3 * a
    up, dn = (t & ~a) >> 1, (a & ~t) >> 1
    neg = v < 0
    return np.where(neg, dn, up), np.where(neg, up, dn)
def obits(lines): return int(lines - 1).bit_length()       # bits for one order entry

def pack_order(o, nb):                                       # (lines, neurons) whole numbers -> packed bits, rank outer
    out = []; flat = o.reshape(-1).astype(np.int64); sh = np.arange(nb - 1, -1, -1, dtype=np.int64)
    step = 1 << 20                                           # a multiple of 8, so every piece packs to whole bytes
    for i in range(0, len(flat), step):
        out.append(np.packbits(((flat[i:i + step, None] >> sh[None, :]) & 1).astype(np.uint8).reshape(-1)))
    return np.concatenate(out)
def unpack_order(buf, lines, neurons, nb):
    n = lines * neurons; out = np.empty(n, np.int64); sh = np.arange(nb - 1, -1, -1, dtype=np.int64); step = 1 << 20
    for i in range(0, n, step):
        m = min(step, n - i); b0 = i * nb // 8
        bits = np.unpackbits(buf[b0:b0 + (m * nb + 7) // 8])[:m * nb].reshape(m, nb).astype(np.int64)
        out[i:i + m] = (bits << sh[None, :]).sum(1)
    return out.reshape(lines, neurons)

def write_plate(name, w16, e):
    """w16: (lines, neurons) int16 as the machine stores it; e: each neuron's power of two."""
    lines, N = w16.shape; nb = obits(lines); assert lines % 8 == 0
    UP = np.zeros((PLACES, lines // 8, N), np.uint8); DN = np.zeros_like(UP); O = np.zeros((lines, N), np.int64)
    lit_plain = lit_signed = 0; top_place = -1
    for c0 in range(0, N, 8192):                             # neurons a piece at a time (the top table is wide)
        w = w16[:, c0:c0 + 8192].astype(np.int64)
        o = np.argsort(~w16[:, c0:c0 + 8192], axis=0, kind="stable")          # largest first (~w = -w - 1), stable
        ws = np.take_along_axis(w, o, 0)                                       # the weights in line-up order
        assert (ws[:-1] >= ws[1:]).all()
        up, dn = signed_digits(ws)
        lit = up | dn
        assert not (up & dn).any() and not (lit & (lit >> 1)).any() and not (lit >> PLACES).any()
        back = np.zeros_like(ws)
        for p in range(PLACES):
            u, d = (up >> p) & 1, (dn >> p) & 1
            back += (u - d) << p
            UP[p, :, c0:c0 + 8192] = np.packbits(u.astype(np.uint8), axis=0)    # 8 lines (by rank) a byte
            DN[p, :, c0:c0 + 8192] = np.packbits(d.astype(np.uint8), axis=0)
        assert np.array_equal(back, ws)
        O[:, c0:c0 + 8192] = o
        lit_plain += int(np.bitwise_count(np.abs(w)).sum()); lit_signed += int(np.bitwise_count(lit).sum())
        if lit.any(): top_place = max(top_place, int(lit.max()).bit_length() - 1)
    po = pack_order(O, nb)
    files = {}
    for kind, arr in (("up", UP), ("down", DN), ("order", po), ("power", np.ascontiguousarray(e))):
        path = f"{OUT}/{name}.{kind}"; arr.tofile(path)
        files[kind] = dict(file=os.path.basename(path), bytes=os.path.getsize(path), sha256=sha(path), dtype=str(arr.dtype),
                           shape=list(arr.shape))
    # read back from disk, alone, and rebuild the plate in its own line order
    up = np.fromfile(f"{OUT}/{name}.up", np.uint8).reshape(PLACES, lines // 8, N)
    dn = np.fromfile(f"{OUT}/{name}.down", np.uint8).reshape(PLACES, lines // 8, N)
    o = unpack_order(np.fromfile(f"{OUT}/{name}.order", np.uint8), lines, N, nb)
    assert np.array_equal(np.sort(o, axis=0), np.broadcast_to(np.arange(lines)[:, None], (lines, N)))   # each a full order
    ok = True
    for c0 in range(0, N, 8192):
        sl = slice(c0, c0 + 8192); ws = np.zeros((lines, min(8192, N - c0)), np.int64)
        for p in range(PLACES):
            ws += (np.unpackbits(up[p][:, sl], axis=0).astype(np.int64) - np.unpackbits(dn[p][:, sl], axis=0).astype(np.int64)) << p
        nat = np.zeros_like(ws); np.put_along_axis(nat, o[:, sl], ws, 0)              # rank k -> its own line
        ok &= bool(np.array_equal(nat, w16[:, sl].astype(np.int64)))
    e_back = np.fromfile(f"{OUT}/{name}.power", e.dtype); ok &= bool(np.array_equal(e_back, e))
    assert ok, name
    return dict(name=name, lines=lines, neurons=N, places=PLACES, order_bits=nb, files=files, readback_exact=ok,
                lit_plain=lit_plain, lit_signed=lit_signed, highest_lit_place=top_place,
                power_dtype=str(e.dtype), layout=dict(up_down="(place, lines/8, neuron) uint8, np.packbits over lines in line-up rank, top line in the high bit",
                                                       order=f"(rank, neuron) row-major, {nb} bits an entry, np.packbits, high bit first"))

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    M = np.load(NPZ)
    plates = BANKS + ["head"]
    man = dict(what="Phase C recording: the 48 plates and the top table in signed digits (up/down sheets per place, packed "
                    "over the lines, lines in each neuron's line-up order, order stored, each neuron's power of two). "
                    "Move 1 alone, no turn.", places=PLACES, plates={}, rest=None)
    for name in plates:
        w16 = M[name + ".w"]; e = M[name + ".e"]
        assert w16.dtype == np.int16, (name, w16.dtype)
        r = write_plate(name, w16, e); man["plates"][name] = r; del w16
        log(f"  {name:9s} {r['lines']:5d} lines x {r['neurons']:6d} neurons; order {r['order_bits']} bits; lit places a weight "
            f"plain {r['lit_plain'] / (r['lines'] * r['neurons']):.3f} -> signed {r['lit_signed'] / (r['lines'] * r['neurons']):.3f}; "
            f"highest lit place {r['highest_lit_place']}; read back exact {r['readback_exact']} ({time.time() - t0:.0f}s, {rss()} MB)")
    rest = {k: M[k] for k in M.files if not k.endswith(".w")}
    np.savez(f"{OUT}/rest.npz", **rest)
    R2 = np.load(f"{OUT}/rest.npz"); assert all(np.array_equal(R2[k], M[k]) and R2[k].dtype == M[k].dtype for k in rest)
    man["rest"] = dict(file="rest.npz", bytes=os.path.getsize(f"{OUT}/rest.npz"), sha256=sha(f"{OUT}/rest.npz"),
                       keys=sorted(rest), note="unchanged numbers of base2_gpt2.npz that are not plates (no '.w' key)")
    assert sorted(set(M.files) - set(rest)) == sorted(p + ".w" for p in plates)        # every '.w' is a recorded plate
    tot = {}
    for nm, r in man["plates"].items():
        g = "top table" if nm == "head" else "48 plates"
        t = tot.setdefault(g, dict(weights=0, up_down_bytes=0, order_bytes=0, power_bytes=0, lit_plain=0, lit_signed=0))
        t["weights"] += r["lines"] * r["neurons"]; t["up_down_bytes"] += r["files"]["up"]["bytes"] + r["files"]["down"]["bytes"]
        t["order_bytes"] += r["files"]["order"]["bytes"]; t["power_bytes"] += r["files"]["power"]["bytes"]
        t["lit_plain"] += r["lit_plain"]; t["lit_signed"] += r["lit_signed"]
    man["totals"] = tot
    json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=1)
    for g, t in tot.items():
        n = t["weights"]
        log(f"\n{g}: {n:,} weights; sheets {t['up_down_bytes']:,} bytes ({8 * t['up_down_bytes'] / n:.2f} bits a weight); "
            f"order {t['order_bytes']:,} bytes ({8 * t['order_bytes'] / n:.2f} bits a weight); powers {t['power_bytes']:,} bytes; "
            f"plain 16-bit {2 * n:,} bytes")
        log(f"  lit places a weight: plain {t['lit_plain'] / n:.4f}, signed digits {t['lit_signed'] / n:.4f} "
            f"({100 * t['lit_signed'] / t['lit_plain']:.1f}%)")
    log(f"rest.npz {man['rest']['bytes']:,} bytes ({len(rest)} arrays, read back equal)")
    log(f"manifest.json written ({time.time() - t0:.0f}s, max memory {rss()} MB)")
