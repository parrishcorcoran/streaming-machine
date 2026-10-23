"""Ups and downs on the blank start and on the finished plates, in base 2.

His question (2026-10-06), his exact words: "The base blank slate should be equal in ups and downs right? In base 2?"

Reads the Pythia plates read-only, plate by plate (safetensors, mmap), for the starting weights (step0) and the
finished weights (step143000). The prediction is in PREDICTION.txt, written before this ran.

THE GRID. One power of two per plate, the same for start and finished of that plate: the smallest power that holds
the larger of the two plates' largest values in 16 places (|value| <= 32767, the machine's 16 places: 15 of size and
the sign). Values are rounded to that grid half-to-even (the same for + and -, so rounding cannot tip a balance); the
count of values that sit off the grid is reported.

TWO FORMS of each whole number
  ss  sign and size: an UP is a 1 in the size of a positive weight, a DOWN a 1 in the size of a negative weight.
      Places 0-14 carry size (place 15 is the sign's place and holds no size).
  sd  signed digits: each whole number written with +1 / -1 digits, no two non-zero digits side by side (the one such
      writing of each number); an UP is a +1 digit, a DOWN a -1 digit. Places 0-15.
  Either form: sum over places of 2^p (ups_p - downs_p) = the plate's net, checked exactly for every plate.

WHAT IS COUNTED, per plate and step
  1. per place p: ups, downs, ups - downs, z = (ups - downs)/sqrt(ups + downs)   (a fair coin: |z| about 1)
  2. the plate's net: sum of all values, z = net / sqrt(sum of squares)   (spread x sqrt(count))
  3. per neuron (each output neuron's own weights) and per line (each input line's weights): the share with net
     beyond 3 widths; the share of (neuron, place) cells with |z| beyond 3; the share of neurons with any place beyond 3.
     Plates [out][in] as Pythia holds them: neuron = row, line = column. Word tables: the input word table's neurons
     are the hidden lanes (one per column) and its lines are the words; the output word table's neurons are the words
     and its lines are the lanes. The log names them "lane" and "word".
  4. the reference: a fresh bell-curve draw of the same spread (root mean square) and shape, on the same power of two
     (one place coarser only if the draw would pass 16 places), counted the same way.
  shuffle control (per neuron and per line): the plate's own values shuffled over the plate. The whole-plate counts
  are the same for the shuffle as for the plate (they hold every value): asserted, and that is the order-free check.

    python3 balance_count.py [sizes=70m,160m]
"""
import os, sys, json, math, time, numpy as np
from safetensors import safe_open

A = dict(a.split("=") for a in sys.argv[1:])
SIZES = A.get("sizes", "70m,160m").split(",")
HERE = os.path.dirname(os.path.abspath(__file__)); MD = os.path.join(os.path.dirname(HERE), "models")
STEPS = ("step0", "step143000")
PL = {"ss": 15, "sd": 16}                    # places counted in each form


def masks(q, form):
    """two whole-number masks: P holds the UP 1s of each value, M the DOWN 1s, place by place"""
    a = np.abs(q).astype(np.uint16)                                     # |q| <= 32767; 3|q|/2 < 2^16 too
    if form == "ss":
        z = np.zeros_like(a)
        return np.where(q > 0, a, z), np.where(q < 0, a, z)
    xh = a >> 1; x3 = a + xh; c = xh ^ x3; pp = x3 & c; mm = xh & c   # signed digits of the size: a = pp - mm
    del xh, x3, c
    return np.where(q > 0, pp, mm), np.where(q > 0, mm, pp)            # a negative value turns every digit over


def count(q, form, nax, lax):
    """ups/downs per place over the plate, per neuron (reduce nax) and per line (reduce lax)"""
    P, M = masks(q, form); n = PL[form]
    assert not np.any(P & M)                                            # never an up and a down in the same place
    out = dict(up=np.zeros(n, np.int64), dn=np.zeros(n, np.int64))
    nn, nl = q.shape[1 - nax], q.shape[1 - lax]
    out.update(n_d=np.zeros((n, nn), np.int64), n_t=np.zeros((n, nn), np.int64),
               l_d=np.zeros((n, nl), np.int64), l_t=np.zeros((n, nl), np.int64))
    for p in range(n):
        bu = ((P >> p) & 1).astype(np.int8); bd = ((M >> p) & 1).astype(np.int8)
        d = bu - bd; t = bu + bd
        out["up"][p] = bu.sum(dtype=np.int64); out["dn"][p] = bd.sum(dtype=np.int64)
        out["n_d"][p] = d.sum(axis=nax, dtype=np.int64); out["n_t"][p] = t.sum(axis=nax, dtype=np.int64)
        out["l_d"][p] = d.sum(axis=lax, dtype=np.int64); out["l_t"][p] = t.sum(axis=lax, dtype=np.int64)
        del bu, bd, d, t
    assert P.max(initial=0) < 2 ** n and M.max(initial=0) < 2 ** n
    net = int(q.sum(dtype=np.int64))
    w = 2 ** np.arange(n, dtype=np.int64)
    assert int((w * (out["up"] - out["dn"])).sum()) == net, "sum of 2^p (ups - downs) is not the net"
    return out


def nets(q, nax, lax):
    q2 = q.astype(np.int32); q2 *= q2                                   # q^2 <= 32767^2 < 2^31
    I = np.int64
    r = dict(net=int(q.sum(dtype=I)), sq=int(q2.sum(dtype=I)),
             n_net=q.sum(axis=nax, dtype=I), n_sq=q2.sum(axis=nax, dtype=I),
             l_net=q.sum(axis=lax, dtype=I), l_sq=q2.sum(axis=lax, dtype=I))
    del q2
    return r


def zs(d, t):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(t > 0, d / np.sqrt(np.maximum(t, 1)), np.nan)


def summary(c, nt):
    """the numbers the log reports, from one count of one form"""
    zp = zs(c["up"] - c["dn"], c["up"] + c["dn"])
    zn = zs(c["n_d"], c["n_t"]); zl = zs(c["l_d"], c["l_t"])
    nz = nt["n_net"] / np.sqrt(np.maximum(nt["n_sq"], 1)); lz = nt["l_net"] / np.sqrt(np.maximum(nt["l_sq"], 1))
    live = ~np.isnan(zp)
    i = int(np.nanargmax(np.abs(zp))) if live.any() else -1
    def cells(z):
        ok = ~np.isnan(z)
        return float((np.abs(z[ok]) > 3).mean()) if ok.any() else float("nan")
    def anyp(z):
        return float((np.nan_to_num(np.abs(z)) > 3).any(axis=0).mean())
    return dict(place_z=[None if np.isnan(v) else round(float(v), 3) for v in zp],
                ups=c["up"].tolist(), downs=c["dn"].tolist(),
                maxz=round(float(abs(zp[i])), 3) if i >= 0 else None, maxz_place=i,
                n_over2=int((np.abs(zp[live]) > 2).sum()), n_over3=int((np.abs(zp[live]) > 3).sum()), n_live=int(live.sum()),
                plate_net=nt["net"], plate_z=round(nt["net"] / math.sqrt(max(nt["sq"], 1)), 3),
                neuron_net3=float((np.abs(nz) > 3).mean()), neuron_cell3=cells(zn), neuron_any3=anyp(zn),
                neuron_maxnetz=round(float(np.abs(nz).max()), 2),
                line_net3=float((np.abs(lz) > 3).mean()), line_cell3=cells(zl), line_any3=anyp(zl),
                line_maxnetz=round(float(np.abs(lz).max()), 2),
                top_place_neuron_cell3=[cells(zn[p]) for p in range(zn.shape[0])],
                top_place_line_cell3=[cells(zl[p]) for p in range(zl.shape[0])])


def quant(W, e):
    q = np.round(W / np.float32(2.0 ** e))
    assert np.abs(q).max() <= 32767
    return q.astype(np.int16)


def plates(L):
    for l in range(L):
        p = f"gpt_neox.layers.{l}."
        yield f"qkv.{l}", p + "attention.query_key_value.weight", 1, 0
        yield f"attn_out.{l}", p + "attention.dense.weight", 1, 0
        yield f"mlp_in.{l}", p + "mlp.dense_h_to_4h.weight", 1, 0
        yield f"mlp_out.{l}", p + "mlp.dense_4h_to_h.weight", 1, 0
    yield "words_in", "gpt_neox.embed_in.weight", 0, 1        # neuron = lane (column), line = word (row)
    yield "words_out", "embed_out.weight", 1, 0               # neuron = word (row), line = lane (column)


for SIZE in SIZES:
    t0 = time.time()
    cfg = json.load(open(os.path.join(MD, f"pythia-{SIZE}-step143000", "config.json")))
    F = {st: safe_open(os.path.join(MD, f"pythia-{SIZE}-{st}", "model.safetensors"), "np") for st in STEPS}
    res = []; logl = []
    for pi, (name, key, nax, lax) in enumerate(plates(cfg["num_hidden_layers"])):
        top = max(float(np.abs(F[st].get_tensor(key)).max()) for st in STEPS)   # fp16 values exactly
        e = math.ceil(math.log2(top / 32767))
        rec = dict(plate=name, shape=list(F["step0"].get_slice(key).get_shape()), power=e, steps={})
        for si, st in enumerate(STEPS):
            W = F[st].get_tensor(key).astype(np.float32)                         # fp16 -> fp32 is exact
            q = quant(W, e)
            off = int((q.astype(np.float32) * np.float32(2.0 ** e) != W).sum())  # exact: both sides are fp32 grid values
            rms = float(np.sqrt(np.mean(np.square(W, dtype=np.float64))))
            del W
            rng = np.random.default_rng(1000 * pi + si + (0 if SIZE == "70m" else 7_000_000 if SIZE == "160m" else 13_000_000))
            fr = rng.standard_normal(q.shape, dtype=np.float32) * np.float32(rms)
            ef = e if float(np.abs(fr).max()) / 2.0 ** e <= 32767 - 0.5 else e + 1
            qf = quant(fr, ef); del fr
            qs = rng.permutation(q.ravel()).reshape(q.shape)
            ntr, nfr, nsh = nets(q, nax, lax), nets(qf, nax, lax), nets(qs, nax, lax)
            S = dict(off_grid=off, values=int(q.size), rms=rms, fresh_power=ef, top=int(np.abs(q).max()))
            for form in ("ss", "sd"):
                cr = count(q, form, nax, lax); cs = count(qs, form, nax, lax); cf = count(qf, form, nax, lax)
                assert np.array_equal(cr["up"], cs["up"]) and np.array_equal(cr["dn"], cs["dn"])   # order-free
                S[form] = dict(plate=summary(cr, ntr), fresh=summary(cf, nfr), shuffle=summary(cs, nsh))
                del cr, cs, cf
            rec["steps"][st] = S
            del q, qf, qs
        res.append(rec)
        for st in STEPS:
            S = rec["steps"][st]
            for form in ("ss", "sd"):
                a, f, s = S[form]["plate"], S[form]["fresh"], S[form]["shuffle"]
                nm = ("lane", "word") if name == "words_in" else ("word", "lane") if name == "words_out" else ("neuron", "line")
                line = (f"{name:11s} {st[4:]:>6s} {form} e={e:+d} | places: max|z| {a['maxz']:6.2f}@p{a['maxz_place']:<2d} "
                        f">2:{a['n_over2']:2d} >3:{a['n_over3']:2d} of {a['n_live']:2d} (fresh max {f['maxz']:5.2f} >2:{f['n_over2']} >3:{f['n_over3']}) "
                        f"| net z {a['plate_z']:+9.2f} (fresh {f['plate_z']:+5.2f}) "
                        f"| per {nm[0]}: net>3 {100*a['neuron_net3']:6.2f}% (f {100*f['neuron_net3']:.2f} s {100*s['neuron_net3']:.2f}) "
                        f"cells>3 {100*a['neuron_cell3']:6.2f}% (f {100*f['neuron_cell3']:.2f} s {100*s['neuron_cell3']:.2f}) "
                        f"| per {nm[1]}: net>3 {100*a['line_net3']:6.2f}% (f {100*f['line_net3']:.2f} s {100*s['line_net3']:.2f}) "
                        f"cells>3 {100*a['line_cell3']:6.2f}% (f {100*f['line_cell3']:.2f} s {100*s['line_cell3']:.2f})")
                print(line, flush=True); logl.append(line)
        print(f"  [{name} done, {time.time() - t0:.0f}s, off grid start {rec['steps']['step0']['off_grid']:,} "
              f"finished {rec['steps']['step143000']['off_grid']:,} of {rec['steps']['step0']['values']:,}]", flush=True)
    json.dump(res, open(os.path.join(HERE, f"balance_{SIZE}.json"), "w"))
    print(f"wrote balance_{SIZE}.json, {time.time() - t0:.0f}s", flush=True)
