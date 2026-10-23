"""Independent recount, pythia-70m, three plates (qkv.0, mlp_in.5, words_out) at step0 and step143000,
plus every 70m plate's whole-plate net, plus per-neuron / per-line shares beyond 3 widths on the three plates.

Written separately from balance_count.py and balance_checker.py: reads the safetensors file by its own header
(np.memmap, read-only), finds the power of two by trying each power, takes the places by whole-number division,
and writes the signed digits by the digit-by-digit rule on the signed number itself (mod 4), not by the bit trick.
Compares every count with balance_70m.json.

    python3 recheck_70m_3plates.py
"""
import os, json, struct, math, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); MD = os.path.join(os.path.dirname(HERE), "models")
STEPS = ("step0", "step143000")


def open_st(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    base = 8 + n
    def get(key):
        h = hdr[key]; assert h["dtype"] == "F16", h["dtype"]
        a, b = h["data_offsets"]
        m = np.memmap(path, dtype=np.float16, mode="r", offset=base + a, shape=tuple(h["shape"]))
        return m
    return get


def power_for(top):
    e = int(math.floor(math.log2(top))) - 20
    while np.rint(top / 2.0 ** e) > 32767:
        e += 1
    return e


def grid(W, e):
    q = np.rint(W.astype(np.float64) / 2.0 ** e)          # rint = half to even
    assert np.abs(q).max() <= 32767
    return q.astype(np.int64)


def places_ss(q):
    """sign and size: places 0-14 of |q| by whole-number division"""
    a = np.abs(q); ups = []; dns = []; pos = q > 0; neg = q < 0
    for p in range(15):
        b = (a // (2 ** p)) % 2
        ups.append(int(b[pos].sum())); dns.append(int(b[neg].sum()))
    assert (a // 2 ** 15 == 0).all()
    return np.array(ups), np.array(dns)


def places_sd(q):
    """signed digits, no two non-zero side by side, built from the low end on the signed number itself"""
    n = q.copy(); ups = []; dns = []; prev = np.zeros(q.shape, np.int8)
    for p in range(17):
        r = np.mod(n, 4)                                   # 0..3, floor rule, works for negatives
        d = np.where(n % 2 != 0, 2 - r, 0).astype(np.int8)  # +1 if n = 1 mod 4, -1 if n = 3 mod 4
        assert not np.any((d != 0) & (prev != 0)), "two non-zero digits side by side"
        ups.append(int((d == 1).sum())); dns.append(int((d == -1).sum()))
        n = (n - d) // 2; prev = d
    assert (n == 0).all()
    assert ups[16] == 0 and dns[16] == 0
    return np.array(ups[:16]), np.array(dns[:16])


def zz(u, d):
    t = [a + b for a, b in zip(u, d)]
    return [None if x == 0 else round((a - b) / math.sqrt(x), 3) for a, b, x in zip(u, d, t)]


def plate_names(L):
    out = []
    for l in range(L):
        p = f"gpt_neox.layers.{l}."
        out += [(f"qkv.{l}", p + "attention.query_key_value.weight"), (f"attn_out.{l}", p + "attention.dense.weight"),
                (f"mlp_in.{l}", p + "mlp.dense_h_to_4h.weight"), (f"mlp_out.{l}", p + "mlp.dense_4h_to_h.weight")]
    return out + [("words_in", "gpt_neox.embed_in.weight"), ("words_out", "embed_out.weight")]


G = {st: open_st(os.path.join(MD, f"pythia-70m-{st}", "model.safetensors")) for st in STEPS}
claim = {r["plate"]: r for r in json.load(open(os.path.join(HERE, "balance_70m.json")))}
bad = 0

print("== whole-plate nets, all 26 plates (net / sqrt(sum of squares)) ==")
for name, key in plate_names(6):
    top = max(float(np.abs(G[st](key).astype(np.float64)).max()) for st in STEPS)
    e = power_for(top)
    row = [f"{name:10s} e={e:+d} (claimed {claim[name]['power']:+d})"]
    if e != claim[name]["power"]: bad += 1; row.append("POWER MISMATCH")
    for st in STEPS:
        q = grid(np.asarray(G[st](key)), e)
        net = int(q.sum()); sq = int((q * q).sum()); z = net / math.sqrt(sq)
        c = claim[name]["steps"][st]["ss"]["plate"]
        ok = (net == c["plate_net"]) and abs(z - c["plate_z"]) < 0.001
        bad += (not ok)
        row.append(f"{st[4:]:>6s} net z {z:+8.3f} (claimed {c['plate_z']:+8.3f}) {'ok' if ok else 'MISMATCH'}")
        del q
    print("  ".join(row), flush=True)

print("\n== per place, three plates ==")
for name, key in [(n, k) for n, k in plate_names(6) if n in ("qkv.0", "mlp_in.5", "words_out")]:
    e = claim[name]["power"]
    for st in STEPS:
        q = grid(np.asarray(G[st](key)), e)
        for form, fn in (("ss", places_ss), ("sd", places_sd)):
            u, d = fn(q)
            assert int(((u - d) * 2 ** np.arange(len(u))).sum()) == int(q.sum())
            c = claim[name]["steps"][st][form]["plate"]
            ok = u.tolist() == c["ups"] and d.tolist() == c["downs"]
            bad += (not ok)
            z = zz(u.tolist(), d.tolist())
            zl = [abs(v) for v in z if v is not None]
            print(f"{name:10s} {st[4:]:>6s} {form} ups/downs {'IDENTICAL' if ok else 'MISMATCH'} | "
                  f"max|z| {max(zl):6.2f} at p{[abs(v) if v is not None else -1 for v in z].index(max(zl))} "
                  f"| >2: {sum(v > 2 for v in zl)} >3: {sum(v > 3 for v in zl)} of {len(zl)}", flush=True)
            print("    z:", z, flush=True)
            if name == "words_out" and form == "ss":
                for p in (12, 13):
                    print(f"    place {p}: ups {u[p]:,} downs {d[p]:,} diff {u[p]-d[p]:+,} width {math.sqrt(u[p]+d[p]):.0f}")
                print("    down/up - 1 by place:", [round(float(d[p] / u[p] - 1) * 100, 2) if u[p] else None for p in range(15)])
        # per neuron and per line: net beyond 3 widths. Pythia plates are [out][in]: neuron = row, line = column.
        qf = q.astype(np.float64)
        for ax, lab in ((1, "per row (neuron / word)"), (0, "per column (line / lane)")):
            zn = qf.sum(axis=ax) / np.sqrt((qf * qf).sum(axis=ax))
            share = float((np.abs(zn) > 3).mean())
            cl = claim[name]["steps"][st]["ss"]["plate"]["neuron_net3" if ax == 1 else "line_net3"]
            ok = abs(share - cl) < 1e-12; bad += (not ok)
            print(f"    {lab}: {100*share:.2f}% beyond 3 (claimed {100*cl:.2f}%) {'ok' if ok else 'MISMATCH'}", flush=True)
        # an own fresh bell-curve draw, same spread, same shape, same power, other seed: the coin toss
        rng = np.random.default_rng(20261006 + len(name) + (1 if st == "step143000" else 0))
        W = np.asarray(G[st](key)).astype(np.float64); rms = math.sqrt(float((W * W).mean())); del W
        fq = np.rint(rng.standard_normal(q.shape) * rms / 2.0 ** e).astype(np.int64)
        if np.abs(fq).max() > 32767: fq = np.rint(fq / 2.0).astype(np.int64)
        for form, fn in (("ss", places_ss), ("sd", places_sd)):
            u, d = fn(fq); zl = [abs(v) for v in zz(u.tolist(), d.tolist()) if v is not None]
            print(f"    my fresh draw {form}: max|z| {max(zl):.2f}, >3: {sum(v > 3 for v in zl)} of {len(zl)}; "
                  f"net z {fq.sum() / math.sqrt(float((fq.astype(np.float64)**2).sum())):+.2f}", flush=True)
        del q, qf, fq

print(f"\nMISMATCHES: {bad}")
