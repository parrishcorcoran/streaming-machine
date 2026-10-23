"""BASEDOWN/COMBINE: THE LANDINGS THROUGH GROUP PILES -- the line-up of basedown/lineup.py combined with the group
machine (groups2/gbank.py at g = 12, the confirmed groups), at all 48 banks of GPT-2. Whole numbers only.

HIS WORDS (2026-10-07), the only sentences quoted as his:
  "It's removable as in convert them to base 2 and line them all up, any shared 1s in a line cancel, any shared zeros
  cancel"
  "Everything these cancel with ARE the base. So every single weight they touch we can extract those also, and alllllll
  the way down all The layers"

WHAT lineup.py FOUND (basedown/out/lineup.log): one order a line, each product = the previous token's + w x gap, each
product landing in its own token's pile. Every pile exact. The landings (the 1s of every product) are about 31-33% of
plain at every plate, whatever the soup; the walk is about 16-24% (b) and 39-46% (a). In (b) the landings are the
larger part. So the task's next step: land through group piles across lines, as the group machine does.

ANALYSIS, BEFORE ANY RULE (the session's): a product w[j,n] x x[t,j] belongs to ONE token's pile at ONE neuron; no
other pile uses it. So summing the products of a group of lines into a group pile and landing that once costs the
group pile's own adds (every product still lands once, now into the group pile) plus the group pile's landing:
more than one by one, never less. A pile can only be shared where many piles need the same sum. The group machine
shares because what it lands is not a product but a ready pile of the LIGHT (the group's light values, each one up or
a hole), which every neuron at every place of the plate reads; the plate's bits only choose which. So the products of
the per-line walk cannot be landed through group piles; what can go through group piles is the light's line-up
itself. That is the rule below.

THE RULE (the session's; fixed here, before any run of this file; nothing in it is read from results)
  - The group machine's bank as confirmed: groups2/gbank.GPlate(w, [(kind, 0, M)], 12, [1, slab, layer, 12]): the
    confirmed random groups of 12 lines, each place of the plate up/hole or signed digits by the plate alone.
  - EACH GROUP LINES UP ITS OWN TOKENS, from the light alone: the group's values written in base 2 (each offset by
    2^15 so the order runs from the most negative up), their places interleaved from the top (place 16 of every line
    of the group, lines in index order, then place 15 of every line, and so on), and the tokens sorted by that: tokens
    that share the top places of all the group's lines sit next to each other. Soups (a) each text, (b) all 648.
  - Each token's GAP in the group = its group values minus the previous token's (the first token of a soup against
    0: its values whole). The gaps are the light the group machine meets: their ready piles (the half table by the
    flip) and every neuron's reads, exactly as gbank does for the light.
  - The reads land in the group's pile Q' (the doubled part before the offset correction), which walks along the
    group's order: Q' at a token = Q' at the previous token + this gap's reads. Then Q' lands in its token's pile:
    its 1s, once a group, a neuron and a token. Each token's pile then takes kcorr x X(t) once (X(t) its own light
    summed, as the group machine does) and is halved.
  WORK (photons, gbank's unit): on the gaps, the light written (the gaps' 1s), the half table, the ready piles
  written, the up/hole reads and the signed-digit reads; on the light, X summed and taken from each neuron (as the
  group machine); the gaps formed (the previous values' 1s); Q' landed (its 1s). Compares (the order) are a count,
  kept apart. Against: THE GROUP MACHINE REBUILT HERE on the same light (gbank's every item), and plain.

CONTROLS AND CHECKS
  exact: gbank's piles of the gaps = their direct product (lineup's walk.c `direct`); every Q' + kcorr X_G(t) = 2 x
    the group's direct part (gwalk.c counts any that differ: 0); every pile = recording/C_ref/plain_refs.npz's; the
    group machine rebuilt: its piles = plain_refs.
  order-free: each soup's tokens presented in a random order: every pile identical; counts identical unless two
    tokens tie in a whole group (then only the order between them changes).
  shuffle control (count only): the same shuffled light as lineup.py (each position's light shuffled across its lines,
    rng [7, slab, layer]); both the group machine and the combination on it; its piles against its own direct product.

RULE 13, CHECKED BEFORE THE RUN
  13.1 the measure is the work of forming every exact pile.
  13.2 nothing of the machine changes; the plate's places, the groups and their reads are the confirmed ones.
  13.3 no agreement statistic.
  13.4 FIXED BY THE DEFINITIONS: the number of reads (one a neuron, group and up/hole place, whatever the light): the
       line-up can only make each read smaller, and the landing of Q' is new work. The run measures the reads' size on
       the gaps, the table and ready piles on the gaps, and Q' landed.
  13.5 the rule and the restatement are the session's.
  13.6 every bank, every position, both soups.

ANALYSIS BEFORE THE RUN (the session's)
  - With 12 lines interleaved, 648 tokens share about 9 places of the interleaved key with their neighbour: the top
    place of most of the 12 lines (in effect the sign); 19-24 tokens (a) share about 4-5: the sign of 4-5 lines. So a
    group's gaps are differences of values that mostly share the sign and not much more: the per-line line-up's 65-70%
    (b) and 25-30% (a) come from one line at a time, and almost none of it survives 12 lines at once.
  - The reads at the group machine are about 80% of its bank work (base2_groups.py's report); its ready piles are sums
    of 12 values each up or a hole, so their size is set by the group's largest values.
  - Q' is about 2 x (12 products) plus the offset term, about 26-30 places: about 13-15 1s. One a group, neuron and
    token: about 64 x 14 photons a pile at a 768-line plate against about 768 x 31 plain: about 3.5-4% of plain.
PREDICTIONS (written before any run of this file; the session's)
  - Exact: every pile identical; gaps' group piles = their direct product; every Q' check 0.
  - The gaps' lit places against the light's (signed digits, within the groups): (b) 85-97%, (a) 92-100%.
  - The gaps' reads: (b) 88-98% of the group machine's reads, (a) 95-102%.
  - Q' landed: 3-5% of plain at every plate kind.
  - THE COMBINATION DOES NOT PAY: (b) 0 to +4 points of plain above the group machine rebuilt here, (a) +2 to +5;
    against the per-line walk of lineup.py (48-56% (b)) it is far lower, because the landings are 1 a group and not 1
    a line.
  - Shuffle control: the gaps' share of the light's lit places about the same as the real light's (within 3 points):
    with 12 lines at once almost no line-up survives in either.
  - Order-free: piles identical; counts identical.

    python3 basedown/combine.py  -> basedown/out/combine.log, combine.json
"""
import os, sys, json, time, ctypes, resource, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; BD = f"{HERE}/basedown"; OUT = f"{BD}/out"
sys.path.insert(0, f"{HERE}/groups2")
_argv = sys.argv; sys.argv = [sys.argv[0]]
import gbank as GB
sys.argv = _argv
LOG = open(f"{OUT}/combine.log", "w"); T0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6
def ones(A): return np.bitwise_count(np.abs(np.asarray(A, dtype=np.int64))).astype(np.int64)
def naf(A): a = np.abs(np.asarray(A, dtype=np.int64)); return np.bitwise_count(a ^ (3 * a)).astype(np.int64)

def lib(name, fn):
    src, so = f"{BD}/{name}.c", f"{BD}/{name}.dylib"
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        import subprocess; subprocess.check_call(["cc", "-O3", "-mcpu=apple-m1", "-shared", "-o", so, src])
    return ctypes.CDLL(so)
V, I = ctypes.c_void_p, ctypes.c_int64
WL = lib("walk", None); WL.direct.argtypes = [I, I, I, V, V, V]; WL.direct.restype = None
GW = lib("gwalk", None); GW.gwalk.argtypes = [I] * 5 + [V] * 13; GW.gwalk.restype = None

def direct(x, w16):
    x = np.ascontiguousarray(x, np.int32); w16 = np.ascontiguousarray(w16, np.int16); P, N = x.shape; M = w16.shape[1]
    out = np.empty((P, M), np.int64); WL.direct(P, N, M, x.ctypes.data, w16.ctypes.data, out.ctypes.data); return out

def group_lineup(x, soups, glines, goff):
    """every group's own order in every soup (the interleaved places from the top), the gaps D at their ranks, the
    token at each rank (tokmap, G x P), the previous values' 1s (gaps formed), compares (count)"""
    P, N = x.shape; G = len(goff) - 1; D = np.zeros((P, N), np.int64); tokmap = np.zeros((G, P), np.int32)
    formed = 0; compares = 0
    for k in range(G):
        L = np.sort(glines[goff[k]:goff[k + 1]])
        for rows in soups:
            u = x[np.ix_(rows, L)].astype(np.int64) + (1 << 15)                      # (p, gk), 0 .. 2^16
            assert u.min() >= 0 and u.max() < 1 << 17
            keys = [(u[:, i] >> b) & 1 for b in range(17) for i in range(len(L) - 1, -1, -1)]   # lexsort: last key first
            o = np.lexsort(keys)                                                     # stable: ties keep the presented order
            v = x[np.ix_(rows[o], L)].astype(np.int64)
            prev = np.vstack([np.zeros((1, len(L)), np.int64), v[:-1]])
            D[np.ix_(rows, L)] = v - prev; tokmap[k, rows] = rows[o]
            formed += int(ones(prev).sum()); p = len(rows); compares += int(p * np.ceil(np.log2(max(p, 2))))
    assert np.abs(D).max(initial=0) <= 1 << 16
    return np.ascontiguousarray(D, np.int32), tokmap, formed, compares

def gsum(c, keys=None):
    return {k: int(v.sum()) for k, v in c.items()}
def items(c):
    r = dict(light=c["light"], table=c["table"], pwrite=c["pwrite"], xsum=c["xsum"], xoff=c["xoff"])
    r["reads"] = sum(v for k, v in c.items() if k.startswith("reads|")); r["sdreads"] = sum(v for k, v in c.items() if k.startswith("sdreads|"))
    return r

def combination(x, gp, w16, soups, gm_items, ref_check):
    """the combination on light x; returns the pile (for the caller's checks) and the counts"""
    P, N = x.shape; M = w16.shape[1]
    D, tokmap, formed, compares = group_lineup(x, soups, gp.glines, gp.goff)
    pD, cD = gp.read(D); cD = items(gsum(cD))
    gd_bad = int((pD != direct(D, w16)).any(1).sum())                          # the gaps' group piles = their direct product
    soff = np.concatenate([[0], np.cumsum([len(r) for r in soups])]).astype(np.int64)
    assert all(np.array_equal(r, np.arange(soff[i], soff[i + 1])) for i, r in enumerate(soups))
    pile2 = np.zeros((P, M), np.int64); Q = np.zeros(M, np.int64); y = np.zeros(M, np.int64); z = np.zeros(M, np.int64)
    out = np.zeros(3, np.int64); xc = np.ascontiguousarray(x, np.int32); kc = np.ascontiguousarray(gp.kcorr, np.int64)
    GW.gwalk(P, N, M, gp.G, len(soups), gp.glines.ctypes.data, gp.goff.ctypes.data, soff.ctypes.data, tokmap.ctypes.data,
             D.ctypes.data, xc.ctypes.data, w16.ctypes.data, kc.ctypes.data, pile2.ctypes.data, Q.ctypes.data,
             y.ctypes.data, z.ctypes.data, out.ctypes.data)
    X = x.astype(np.int64).sum(1); dbl = pile2 + kc[None, :] * X[:, None]
    assert not (dbl & 1).any(), "the halving met an odd doubled pile"
    pile = dbl >> 1
    c = dict(light=cD["light"], table=cD["table"], pwrite=cD["pwrite"], reads=cD["reads"], sdreads=cD["sdreads"],
             xsum=gm_items["xsum"], xoff=gm_items["xoff"], formed=formed, qland=int(out[0]))
    c["work"] = sum(c.values()); c["qtrue"] = int(out[1]); c["compares"] = compares
    c["qbad"] = int(out[2]); c["gd_bad"] = gd_bad
    c["d_sd"] = int(naf(D).sum()); c["x_sd"] = int(naf(x).sum()); c["d_1"] = int(ones(D).sum()); c["x_1"] = int(ones(x).sum())
    return pile, c

REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz"); NPOS = REF["npos"].astype(int); P = int(NPOS.sum())
Wz = np.load(f"{HERE}/base2_gpt2.npz")
LIGHT = {k: np.load(f"{HERE}/groups/light_gpt2/{k}.npy", mmap_mode="r") for k in ("qkv", "proj", "fc", "mproj")}
starts = np.concatenate([[0], np.cumsum(NPOS)])
SOUPS = {"a": [np.arange(starts[i], starts[i + 1]) for i in range(30)], "b": [np.arange(P)]}
SLAB = ["qkv", "proj", "fc", "mproj"]; SLABS = GB.R.SLABS; G_BANK = 12

log("BASEDOWN/COMBINE: the light's line-up through the group machine's piles (g = 12, the confirmed groups), 48 banks, 648 positions")
REC = []; CHK = dict(gm_ref=0, gm_ctl_direct=0, ref=0, gd_bad=0, qbad=0, of_piles=0, of_counts=0, ctl_direct=0, ctl_qbad=0, ctl_gd_bad=0)
for l in range(12):
    for kind in SLAB:
        tb = time.time(); name = f"{kind}.{l}"
        x = np.ascontiguousarray(LIGHT[kind][l], np.int64); w16 = np.ascontiguousarray(Wz[f"{name}.w"], np.int16)
        ref = np.asarray(REF[f"{kind}_pile_{l}"]); N, M = w16.shape; wrow = ones(w16).sum(1)
        gp = GB.GPlate(w16.astype(np.int64), [(kind, 0, M)], G_BANK, [1, SLABS.index(kind), l, G_BANK])
        rng = np.random.default_rng([7, SLAB.index(kind), l]); xs = rng.permuted(x, axis=1)   # lineup.py's shuffled light
        orng = np.random.default_rng([11, SLAB.index(kind), l])
        rec = dict(bank=name, kind=kind, layer=l, plain=int((ones(x) @ wrow).sum()), plain_ctl=int((ones(xs) @ wrow).sum()))
        pg, cg = gp.read(x); gi = items(gsum(cg)); CHK["gm_ref"] += int((pg != ref).any(1).sum())
        rec["gm"] = dict(gi, work=sum(gi.values()))
        pgs, cgs = gp.read(xs); gis = items(gsum(cgs)); ds = direct(xs, w16); CHK["gm_ctl_direct"] += int((pgs != ds).any(1).sum())
        rec["gm_ctl"] = dict(gis, work=sum(gis.values()))
        for sm in ("a", "b"):
            soups = SOUPS[sm]
            pile, c = combination(x, gp, w16, soups, gi, ref)
            CHK["ref"] += int((pile != ref).any(1).sum()); CHK["gd_bad"] += c["gd_bad"]; CHK["qbad"] += c["qbad"]
            rec[sm] = c
            perm = np.concatenate([orng.permutation(r) for r in soups])            # ORDER-FREE: tokens presented in a random order
            pp, cp = combination(np.ascontiguousarray(x[perm]), gp, w16, soups, gi, None)
            back = np.empty_like(pp); back[perm] = pp; CHK["of_piles"] += int((back != ref).any(1).sum())
            CHK["of_counts"] += sum(int(cp[k] != c[k]) for k in ("light", "table", "pwrite", "reads", "sdreads", "formed", "qland"))
            ps, cs = combination(xs, gp, w16, soups, gis, None)                    # SHUFFLE CONTROL
            CHK["ctl_direct"] += int((ps != ds).any(1).sum()); CHK["ctl_qbad"] += cs["qbad"]; CHK["ctl_gd_bad"] += cs["gd_bad"]
            rec[sm + "_ctl"] = cs
        REC.append(rec); pl = rec["plain"]; g = rec["gm"]
        log(f"  {name:8s} group machine {100 * g['work'] / pl:6.2f}%  combination (a) {100 * rec['a']['work'] / pl:6.2f}% (b) {100 * rec['b']['work'] / pl:6.2f}%"
            f"  [reads on gaps/light (a) {100 * (rec['a']['reads'] + rec['a']['sdreads']) / (g['reads'] + g['sdreads']):5.1f}% (b) {100 * (rec['b']['reads'] + rec['b']['sdreads']) / (g['reads'] + g['sdreads']):5.1f}%;"
            f" Q' landed {100 * rec['b']['qland'] / pl:5.2f}%]  ctl: gm {100 * rec['gm_ctl']['work'] / rec['plain_ctl']:6.2f}% comb (b) {100 * rec['b_ctl']['work'] / rec['plain_ctl']:6.2f}%"
            f"  ({time.time() - tb:.0f}s, {rss()} MB)")
        json.dump(dict(rec=REC, chk=CHK), open(f"{OUT}/combine.json", "w"))

log(""); log("CHECKS: " + ", ".join(f"{k} {v}" for k, v in CHK.items()))
log("  gm_* the group machine rebuilt (piles vs plain_refs; shuffled vs its direct product); ref = the combination's piles vs plain_refs;")
log("  gd_bad = the gaps' group piles vs their direct product; qbad = Q' checks failing; of_* order-free; ctl_* the shuffle control")
ITEMS = ("light", "table", "pwrite", "reads", "sdreads", "xsum", "xoff")
for tag, plk, gmk, sms in (("REAL LIGHT", "plain", "gm", ("a", "b")), ("SHUFFLE CONTROL", "plain_ctl", "gm_ctl", ("a_ctl", "b_ctl"))):
    log(""); log(f"{tag}: % of plain (a token = total / {P})")
    for kinds, kl in [([k], k) for k in SLAB] + [(SLAB, "ALL")]:
        R_ = [r for r in REC if r["kind"] in kinds]; pl = sum(r[plk] for r in R_)
        gm = sum(r[gmk]["work"] for r in R_)
        line = f"  {kl:5s} plain {pl / P:>13,.0f}  group machine {100 * gm / pl:6.2f}%"
        for sm in sms:
            cw = sum(r[sm]["work"] for r in R_); ql = sum(r[sm]["qland"] for r in R_)
            rd = sum(r[sm]["reads"] + r[sm]["sdreads"] for r in R_); grd = sum(r[gmk]["reads"] + r[gmk]["sdreads"] for r in R_)
            dsd = sum(r[sm]["d_sd"] for r in R_); xsd = sum(r[sm]["x_sd"] for r in R_)
            line += (f" | {sm}: comb {100 * cw / pl:6.2f}% ({100 * (cw - gm) / pl:+5.2f} pts), Q' {100 * ql / pl:5.2f}%, reads {100 * rd / grd:5.1f}% of gm's,"
                     f" gaps' signed digits {100 * dsd / xsd:5.1f}% of the light's")
        log(line)
    for kl in ("ALL",):
        R_ = REC; pl = sum(r[plk] for r in R_)
        log("    by item (" + tag + ", ALL), % of plain: group machine " + ", ".join(f"{k} {100 * sum(r[gmk][k] for r in R_) / pl:.2f}" for k in ITEMS))
        for sm in sms:
            log(f"    {sm}: " + ", ".join(f"{k} {100 * sum(r[sm][k] for r in R_) / pl:.2f}" for k in ITEMS + ("formed", "qland"))
                + f"; Q' as the true group part (count only) {100 * sum(r[sm]['qtrue'] for r in R_) / pl:.2f}; compares a token {sum(r[sm]['compares'] for r in R_) / P:,.0f}")
log(f"done ({time.time() - T0:.0f}s, max memory {rss()} MB)")
assert all(v == 0 for v in CHK.values()), CHK
