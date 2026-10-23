"""flat_build/1/check_readers.py -- A SECOND CODE PATH for flat build 1 (the readers-sized table), seeded from
flat_design/4/check_indep.py. It imports nothing from base2_groups*.py, base2_fringes2.py, groups2/ (gbank, uh_rebuild,
uh_perplace, bgm.c) or groups/. It reads only:
  base2_gpt2.npz                  the plates' whole numbers (the machine rebuilds its plates from recording/C's sheets)
  groups/light_gpt2/{kind}.npy    base2's own light at every bank, every refs position (captured by a separate run)
  recording/C_ref/plain_refs.npz  base2's own piles at every bank, every refs position
  flat_build/1/machine/gpt2_refs.json   the machine's counts and the layer-0 rows it met with the whole light (to compare)
  groups2/machine/gpt2_refs.json        the confirmed g = 12 run's counts (to compare the g = 12 recount)
Written by the same session as the machine, so under rule 10 it is NOT a separate checker: it is a second code path.

THE RULES AS WRITTEN (gbank.py's header and the build's docstring), rebuilt here from the words:
  - g = 11 at proj and mproj, 12 at qkv and fc; groups: numpy's generator keyed [1, slab, layer, g] (slab qkv 0, proj 1,
    fc 2, mproj 3), a random line order cut into runs of g (the last run may be short).
  - per place b of a plate: L_b = how many weights have a lit signed digit at b (digit 2 - (r mod 4) where r is odd);
    U = groups x neurons; place b is up/hole when L_b >= U, signed digits otherwise.
  - digits, bottom place up, from r = w + C (C = 2^(top up/hole place)): at an up/hole place the binary digit of r, at
    every other place (and 16, 17) a signed digit; 2 x.w = sum over up/hole b of 2^b V_b + 2 x (signed part) + kcorr X,
    V_b = sum over the group's lines of x (up) or -x (hole), kcorr = (sum of 2^b over up/hole places) - 2C.
  - work a position: light = the light's 1s; table = each line's signed digits x (1 if it is not the group's first line,
    plus 2^(k-2-t) for line t <= k-2); writing = the 1s of every half-table entry read (the entry with the group's last
    line up: the pattern itself or its flip); reads = the signed digits of every packet read (one a neuron, group and
    up/hole place); signed-digit reads = each lit signed digit's light value's signed digits; X = the light's signed
    digits on lines 1..N-1; X off = per neuron, kcorr X's signed digits plus (kcorr's signed digits - 1) x X's.
  - adds a position: one a packet read, one a lit signed digit on a nonzero light value, the table's (k - 1) + 2^(k-1) - 1
    a group, X's N - 1, X off's kcorr signed digits a neuron; plain adds: each nonzero light value x the 1s of its line.
Every pile is rebuilt from this code's own digits and packets and ASSERTED equal to base2's own pile at all 648 positions,
all 48 banks. Then every count is compared with the machine's, number for number.
    python3 flat_build/1/check_readers.py   -> flat_build/1/check_readers.log, .json
"""
import json, time, resource, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/flat_build/1"
SLAB = dict(qkv=0, proj=1, fc=2, mproj=3)
GREAD = dict(qkv=12, proj=11, fc=12, mproj=11)
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def onec(a): return np.bitwise_count(np.abs(np.asarray(a, np.int64))).astype(np.int64)
def sdc(a):                                          # signed digits, by the step rule (not the 3v trick), whole numbers
    v = np.abs(np.asarray(a, np.int64)).copy(); n = np.zeros(v.shape, np.int64)
    while v.any():
        odd = (v & 1) == 1; d = np.where(odd, 2 - (v & 3), 0); n += (d != 0); v = (v - d) >> 1
    return n
def sd1(v): return int(sdc(np.array([v]))[0])
LOG = open(f"{OUT}/check_readers.log", "w")
def say(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

def plate_digits(w, g):
    """the per-place rule and the digits of one plate (plate only)"""
    N, M = w.shape; G = -(-N // g); U = G * M
    r = w.copy(); lit = []
    for b in range(18):
        odd = (r & 1) == 1; d = np.where(odd, 2 - (r & 3), 0); lit.append(int((d != 0).sum())); r = (r - d) >> 1
    assert not r.any() and not any(lit[16:])
    up = [b for b in range(16) if lit[b] >= U]
    K = sum(1 << b for b in up); C = (1 << max(up)) if up else 0
    r = w + C; E = {}; Wsd = np.zeros((N, M), np.int64); nsd_line = np.zeros(N, np.int64)
    for b in range(18):
        if b in up:
            e = r & 1; E[b] = e.astype(np.int64); r = (r - e) >> 1
        else:
            odd = (r & 1) == 1; d = np.where(odd, 2 - (r & 3), 0); Wsd += d << b; nsd_line += (d != 0).sum(1); r = (r - d) >> 1
    assert not r.any(), "digits left above place 17"
    back = sum((E[b] << b) for b in up) - C + Wsd if up else Wsd
    assert np.array_equal(back, w), "the digits do not give the plate back"
    return dict(up=up, K=K, C=C, kcorr=K - 2 * C, E=E, Wsd=Wsd, nsd_line=nsd_line, G=G, U=U, lit=lit)

def recount(w, x, kind, l, g, D):
    """every pile and every count a position, at group size g, from this code's own digits"""
    N, M = w.shape; P = len(x); up = D["up"]
    order = np.random.default_rng([1, SLAB[kind], l, g]).permutation(N); groups = [order[s:s + g] for s in range(0, N, g)]
    assert len(groups) == D["G"]
    sx = sdc(x); X = x.sum(1)
    c = dict(light=onec(x).sum(1), table=np.zeros(P, np.int64), pwrite=np.zeros(P, np.int64), reads=np.zeros(P, np.int64),
             sdreads=sx @ D["nsd_line"], xsum=sx[:, 1:].sum(1) if up else np.zeros(P, np.int64), xoff=np.zeros(P, np.int64))
    a = dict(reads=np.zeros(P, np.int64), sd=((x != 0).astype(np.int64) @ D["nsd_line"]), table=np.zeros(P, np.int64),
             xsum=np.full(P, N - 1 if up else 0, np.int64), xoff=np.zeros(P, np.int64))
    D2 = 2 * (x @ D["Wsd"]) + D["kcorr"] * X[:, None]
    for gr in groups:
        k = len(gr); n = 1 << k; xg = x[:, gr]
        bits = ((np.arange(n)[:, None] >> np.arange(k)[None, :]) & 1) * 2 - 1        # (n, k): +1 up, -1 hole
        V = xg @ bits.T                                                                 # (P, n): every pattern's packet
        ch = np.array([(1 if t >= 1 else 0) + ((1 << (k - 2 - t)) if t <= k - 2 else 0) for t in range(k)], np.int64)
        if up:
            c["table"] += sx[:, gr] @ ch; a["table"] += (k - 1) + (1 << (k - 1)) - 1
        read_count = np.zeros(n, np.int64); entries = set()
        for b in up:
            pat = (D["E"][b][gr] << np.arange(k)[:, None]).sum(0)                      # (M,): bit t = line t's binary digit at b
            D2 += np.take(V, pat, axis=1) << b
            read_count += np.bincount(pat, minlength=n)
            hi = 1 << (k - 1); full = n - 1
            entries.update(np.unique(np.where(pat & hi, pat, pat ^ full)).tolist())
        if up:
            c["reads"] += sdc(V) @ read_count; a["reads"] += int(read_count.sum())
            ent = np.array(sorted(entries), np.int64); c["pwrite"] += onec(V[:, ent]).sum(1)
    if up and D["kcorr"] != 0:
        dk = sd1(D["kcorr"]); c["xoff"] = M * (sdc(D["kcorr"] * X) + (dk - 1) * sdc(X)); a["xoff"][:] = M * dk
    assert not (D2 & 1).any(); pile = D2 >> 1
    plain = onec(x) @ onec(w).sum(1); padds = (x != 0).astype(np.int64) @ onec(w).sum(1)
    return pile, c, a, plain, padds

LABEL = dict(light="light written in signed digits", table="half table (by the flip)", pwrite="ready piles written in signed digits",
             xsum="X (the light summed)", xoff="X taken from each neuron", reads="reads, up/hole places", sdreads="reads, signed-digit places")
def main():
    t0 = time.time()
    Z = np.load(f"{HERE}/base2_gpt2.npz"); REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")
    LG = {k: np.load(f"{HERE}/groups/light_gpt2/{k}.npy", mmap_mode="r") for k in SLAB}
    J = json.load(open(f"{OUT}/machine/gpt2_refs.json")); MC = J["counts"]; npos = J["npos"]; assert npos == 648
    CC = json.load(open(f"{HERE}/groups2/machine/gpt2_refs.json"))["counts"]
    cr, r1 = np.array(J["rows"]["L0_carried"]), np.array(J["rows"]["r1"])
    say("SECOND CODE PATH (flat_build/1/check_readers.py): every bank rebuilt from the written rules at the readers g; "
        "piles against base2's own run, counts against the machine's")
    res = dict(banks={}); piles_bad = 0; items_bad = []; n_items = 0; g12_bad = []; n_g12 = 0
    tot = dict(mine={}, mach={}, plain={})
    for l in range(12):
        for kind in ("qkv", "proj", "fc", "mproj"):
            tl = time.time(); w = Z[f"{kind}.{l}.w"].astype(np.int64); x = np.asarray(LG[kind][l]).astype(np.int64)
            ref = REF[f"{kind}_pile_{l}"].astype(np.int64); g = GREAD[kind]
            D = plate_digits(w, g)
            pile, c, a, plain, padds = recount(w, x, kind, l, g, D)
            bad = int((pile != ref).any(1).sum()); piles_bad += bad
            assert bad == 0, (kind, l, bad)
            # which rows the machine met with the whole light, and under which keys it counted them
            lay = f"L{l}"
            if l == 0 and kind in ("qkv", "proj", "fc"):
                rows = cr if kind == "qkv" else r1
                keyf = lambda it: f"sidei|L0|{kind}|storage: the whole light in the group form: {LABEL[it]}"
            else:
                rows = r1 if l == 0 else np.arange(npos)
                keyf = lambda it: (f"fcw|{lay}|fc|{LABEL[it]}" if kind == "fc" and it in ("reads", "sdreads") else f"fr|{lay}|{kind}|{LABEL[it]}")
            rec = {}
            for it in LABEL:
                mine = int(c[it][rows].sum()); mach = MC.get(keyf(it)); n_items += 1
                rec[it] = (mine, mach)
                if mine != mach: items_bad.append((f"{kind}.{l}", it, mine, mach))
            if not (l == 0 and kind in ("qkv", "proj", "fc")):
                pm = int(plain[rows].sum()); pk = MC.get(f"plain|{lay}|{kind}|meetings, plain"); n_items += 1
                if pm != pk: items_bad.append((f"{kind}.{l}", "plain", pm, pk))
                for it in a:
                    am = int(a[it][rows].sum()); ak = MC.get(f"adds|{lay}|{kind}|adds: {it}"); n_items += 1
                    if am != ak: items_bad.append((f"{kind}.{l}", f"adds {it}", am, ak))
                pa = int(padds[rows].sum()); pk = MC.get(f"adds|{lay}|{kind}|adds: plain"); n_items += 1
                if pa != pk: items_bad.append((f"{kind}.{l}", "plain adds", pa, pk))
            mine_tot = sum(int(c[it][rows].sum()) for it in LABEL); mach_tot = sum((MC.get(keyf(it)) or 0) for it in LABEL)
            tot["mine"][f"{kind}.{l}"] = mine_tot; tot["mach"][f"{kind}.{l}"] = mach_tot; tot["plain"][f"{kind}.{l}"] = int(plain[rows].sum())
            # the g = 12 recount at proj and mproj (layers 1-11: the whole light, all rows) against the machine's side count
            # and the confirmed run
            extra = ""
            all_mine = sum(int(c[it].sum()) for it in LABEL); t12 = all_mine
            if kind in ("proj", "mproj"):
                D12 = plate_digits(w, 12); p12, c12, a12, _, _ = recount(w, x, kind, l, 12, D12)
                assert int((p12 != ref).any(1).sum()) == 0
                rows12 = np.arange(npos) if l >= 1 else (r1 if kind == "mproj" else None)    # proj.0's side ran on the rest light
                if rows12 is not None:
                    for it in LABEL:
                        v = int(c12[it][rows12].sum()); n_g12 += 2
                        if v != MC.get(f"g12|{lay}|{kind}|{LABEL[it]}"): g12_bad.append((f"{kind}.{l}", it, "machine side", v, MC.get(f"g12|{lay}|{kind}|{LABEL[it]}")))
                        if v != CC.get(f"fr|{lay}|{kind}|{LABEL[it]}"): g12_bad.append((f"{kind}.{l}", it, "confirmed", v, CC.get(f"fr|{lay}|{kind}|{LABEL[it]}")))
                t12 = sum(int(c12[it].sum()) for it in LABEL); extra = f"; all rows: g = {g} {all_mine / npos:,.0f}, g = 12 {t12 / npos:,.0f} ({100 * (all_mine - t12) / int(plain.sum()):+.2f} points)"
                tot.setdefault("g12", {})[f"{kind}.{l}"] = int(sum(int(c12[it][rows].sum()) for it in LABEL))
            tot.setdefault("all_mine", {})[f"{kind}.{l}"] = all_mine; tot.setdefault("all_g12", {})[f"{kind}.{l}"] = t12
            tot.setdefault("all_plain", {})[f"{kind}.{l}"] = int(plain.sum())
            res["banks"][f"{kind}.{l}"] = dict(g=g, up=D["up"], G=D["G"], U=D["U"], rows=int(len(rows)), items={k: list(v) for k, v in rec.items()})
            say(f"  {kind}.{l:<2d} g={g} groups {D['G']:4d} up/hole places {len(D['up'])}: piles = base2's at 648/648; "
                f"work {mine_tot / npos:>12,.0f} a token on {len(rows)} rows ({100 * mine_tot / max(1, int(plain[rows].sum())):.2f}% of plain); "
                f"machine {mach_tot / npos:>12,.0f}: {'SAME' if mine_tot == mach_tot else 'DIFFERENT'}{extra}  ({time.time() - tl:.1f}s, {rss()} MB)")
    say("")
    say(f"PILES: 48 banks x 648 positions rebuilt from this code's own digits and packets: rows differing from base2's own {piles_bad}")
    say(f"COUNTS against the machine (readers g): {n_items - len(items_bad)}/{n_items} numbers identical" + ("" if not items_bad else f"; DIFFERENT: {items_bad[:20]}"))
    say(f"THE g = 12 RECOUNT at proj and mproj, layers 1-11: {n_g12 - len(g12_bad)}/{n_g12} numbers identical to the machine's side count and the confirmed run"
        + ("" if not g12_bad else f"; DIFFERENT: {g12_bad[:20]}"))
    for kind in ("proj", "mproj"):
        L_ = [f"{kind}.{l}" for l in range(1, 12)]
        m11 = sum(tot["mine"][k] for k in L_); m12 = sum(tot["g12"][k] for k in L_); pl = sum(tot["plain"][k] for k in L_)
        say(f"  {kind} layers 1-11: g = 11 {m11 / npos:,.0f} ({100 * m11 / pl:.3f}% of plain), g = 12 {m12 / npos:,.0f} ({100 * m12 / pl:.3f}%): "
            f"{(m11 - m12) / npos:+,.0f} a token")
    am, a12, ap = (sum(tot[k].values()) for k in ("all_mine", "all_g12", "all_plain"))
    say(f"THE 48 BANKS ALONE, the whole light at every position (nothing stored; the audit's 'banks 22.779%' at g = 12): plain {ap / npos:,.0f} a token; "
        f"g = 12 {a12 / npos:,.0f} = {100 * a12 / ap:.3f}%; readers {am / npos:,.0f} = {100 * am / ap:.3f}% ({100 * (am - a12) / ap:+.3f} points)")
    for kind in ("qkv", "proj", "fc", "mproj"):
        ks = [f"{kind}.{l}" for l in range(12)]; m_ = sum(tot["all_mine"][k] for k in ks); g_ = sum(tot["all_g12"][k] for k in ks); p_ = sum(tot["all_plain"][k] for k in ks)
        say(f"    {kind:6s} readers {100 * m_ / p_:6.3f}%   g = 12 {100 * g_ / p_:6.3f}%")
    res.update(piles_bad=piles_bad, items=n_items, items_bad=items_bad, g12_items=n_g12, g12_bad=g12_bad, totals=tot)
    json.dump(res, open(f"{OUT}/check_readers.json", "w"), indent=0, default=int)
    say(f"done ({time.time() - t0:.0f}s, max memory {rss()} MB)")
    assert piles_bad == 0 and not items_bad and not g12_bad

if __name__ == "__main__":
    main()
