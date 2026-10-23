"""THE BEND AS A PURE WEDGE. The machine's bend dims each neuron's light by a share read from a table (the trained
share, Phi). Here the table is replaced at every layer by a wedge on the pile p (value = p / 2^16):

    value <= -L          output 0                        (blocked: the neuron pours nothing)
    value >= +L          output = value                  (open: the light passes whole)
    in between           output = value x (value + L) / (2L)   (dimmed in proportion to how far into the wedge)

Two wedges, both fixed before anything ran, no others:
    L1   L = sqrt(2 pi) / 2 = 1.2533   the wedge's share has slope 1/(2L) at 0 = the trained share's slope 1/sqrt(2 pi)
    L99  L = 2.3263                    the trained share's own 1% and 99% points
and, ONLY as a sensitivity reading, LHALF = L1 / 2.

Whole numbers: Lw = L at 2^16 (a fixed whole number); output = (p x (p + Lw) x RECIP) shifted down 40 places, with
RECIP = 2^40 / (2 Lw) rounded (the division by 2L as a multiply by a fixed whole number and one shift).

    python3 wedge/wedge_bend.py                 # 30 refs texts: 16 words each, and the exact counts under each wedge
    python3 wedge/wedge_bend.py judge=L1        # judge5: 300 windows, 6,000 positions, top 5 of the full machine
"""
import sys, os, json, time, math, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")   # Accelerate's spurious flags; products checked whole by J.product
HERE = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, HERE); os.chdir(HERE)
ARGS = dict(a.split("=") for a in sys.argv[1:])
import judge5 as J
M, W, ES, SC, ARR = J.M, J.W, J.ES, J.SC, J.ARR
rshift, to16, product, gauge_log, power, log2fix, race = J.rshift, J.to16, J.product, J.gauge_log, J.power, J.log2fix, J.race
REFS = json.load(open(f"{HERE}/refs_many.json"))

# ---------- the wedge, in whole numbers ----------
S = 40
LW = {"L1": round(math.sqrt(2 * math.pi) / 2 * 65536),        # 82,137
      "L99": round(2.3263478740408408 * 65536),               # 152,460 (Phi's 1% / 99% point)
      "LHALF": round(math.sqrt(2 * math.pi) / 4 * 65536)}     # 41,069  (sensitivity only)
RECIP = {k: round(2 ** S / (2 * v)) for k, v in LW.items()}
for k in LW: assert 2 * LW[k] * LW[k] * RECIP[k] + (1 << (S - 1)) < 2 ** 62     # the largest ramp product stays inside 64 bits
def make_wedge(name):
    Lw, R = LW[name], RECIP[name]
    def wedge(H):
        Hc = np.clip(H, -Lw, Lw)                                # only the ramp is multiplied; the ends are set outright
        ramp = rshift(Hc * (Hc + Lw) * R, S)
        return np.where(H <= -Lw, 0, np.where(H >= Lw, H, ramp))
    wedge.Lw = Lw; return wedge
BENDS = {"trained": J.edge, **{k: make_wedge(k) for k in LW}}

# ---------- the forward for judge5 (J.scores_all, step for step, with the bend swapped) ----------
def forward_for(bend):
    def fwd(ids):
        P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
        for l in range(12):
            R = R + J.plain_bank(J.attention(J.gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
            R = R + J.plain_bank(bend(J.gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
        w, we, b = W["head"]; Rj = R[J.FIRST:J.LAST]; x16, ex, LS = gauge_log(Rj)
        return power(product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b
    return fwd

# ---------- photons ----------
_POP = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.int64)
def ones(A):                                         # photons (1s) per value, sign apart
    A = np.abs(np.asarray(A).astype(np.int64)); n = np.zeros(A.shape, np.int64)
    while A.any(): n += _POP[A & 0xFFFF]; A = A >> 16
    return n
PLATES = [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")]
WROW = {n: ones(M[n + ".w"]).sum(1) for n in PLATES}   # photons of each plate line, summed over the neurons it feeds

# ---------- the machine, counted (mirrors J step for step) ----------
def attention_counted(QKV, C):                       # J.attention, counting light-with-light meetings
    P = QKV.shape[0]; out = np.zeros((P, 768), dtype=np.int64)
    for h in range(12):
        q, k, v = QKV[:, h*64:(h+1)*64], QKV[:, 768+h*64:768+(h+1)*64], QKV[:, 1536+h*64:1536+(h+1)*64]
        q16, eq = to16(q, -ES); k16, ek = to16(k, -ES)
        sc = rshift(product(q16, k16.T.astype(np.float64)), -(eq[:, None] + ek[None, :] - 3 + SC))
        C["arrival scores"] += int(np.tril(ones(q16).astype(np.float64) @ ones(k16).T.astype(np.float64)).sum())
        ov = ones(v).sum(1)
        for p in range(P):
            row = sc[p, :p+1]; d = row[race(row)] - row
            a = np.where(d < len(ARR), ARR[np.minimum(d, len(ARR) - 1)], 0)
            num = product(a[None, :], v[:p+1].astype(np.float64))[0]
            out[p, h*64:(h+1)*64] = power(num, -log2fix([a.sum()])[0])
            C["arrival mix"] += int((ones(a) * ov[:p+1]).sum())
    return out

def run_counted(ids, bend, C, L, Lw=None):
    """one text, every position. C: photon meetings by kind; L: per-layer counts. Returns R after layer 11."""
    P = len(ids); R = M["wte"][ids] + M["wpe"][:P]
    for l in range(12):
        x16, ex, LS = gauge_log(R); C["plates"] += int((ones(x16) @ WROW[f"qkv.{l}"]).sum())
        QKV = J.gauged_bank(R, f"qkv.{l}")
        o = attention_counted(QKV, C)
        o16, _ = to16(o, -ES); C["plates"] += int((ones(o16) @ WROW[f"proj.{l}"]).sum())
        R = R + J.plain_bank(o, f"proj.{l}")
        x1, _, _ = gauge_log(R); C["plates"] += int((ones(x1) @ WROW[f"fc.{l}"]).sum())
        H = J.gauged_bank(R, f"fc.{l}")
        out = bend(H); m16, _ = to16(out, -ES)
        mo = ones(m16) @ WROW[f"mproj.{l}"]
        C["plates"] += int(mo.sum()); C["out-bank"] += int(mo.sum())
        d = L[l]; d["piles"] += H.size; d["out zero"] += int((out == 0).sum()); d["light zero"] += int((m16 == 0).sum())
        if Lw is not None:                           # the wedge's three parts, and what the trained bend pours on the same piles
            blk, opn = H <= -Lw, H >= Lw
            d["blocked"] += int(blk.sum()); d["open"] += int(opn.sum()); d["ramp"] += int((~blk & ~opn).sum())
            d["ramp out zero"] += int(((~blk & ~opn) & (out == 0)).sum())
            assert np.array_equal(out[opn], H[opn]) and not out[blk].any()     # open passes whole, blocked pours nothing
            t16, _ = to16(J.edge(H), -ES); tw = ones(t16) * WROW[f"mproj.{l}"][None, :]
            d["out-bank, trained light"] += int(tw.sum()); d["dropped"] += int(tw[blk].sum())
            C["out-bank, trained light on the same piles"] += int(tw.sum()); C["dropped"] += int(tw[blk].sum())
        R = R + J.plain_bank(out, f"mproj.{l}")
    return R

def read_words(R, n0):                               # the word at each of the 16 positions: whoever arrives first at the top
    w, we, b = W["head"]; Rj = R[n0 - 1:n0 + 15]; x16, ex, LS = gauge_log(Rj)
    S_ = power(product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b
    return [race(s) for s in S_]

if __name__ == "__main__":
    t0 = time.time()
    if "judge" in ARGS:
        name = ARGS["judge"]; n = int(ARGS.get("n", 300))
        fwd = forward_for(BENDS[name])
        if ARGS.get("mirror", "1") == "1":           # the swapped forward with the trained bend IS the machine
            f0 = forward_for(J.edge)
            for wdw in J.WINDOWS[:2]: assert np.array_equal(f0(wdw["ids"]), J.scores_all(wdw["ids"]))
            print("mirror check: the forward with the trained bend gives the machine's own scores on 2 judge windows: yes", flush=True)
        inn, same, tot = J.judge(forward=fwd, n=n)
        label = "SENSITIVITY ONLY, L halved" if name == "LHALF" else name
        Lw = LW.get(name)
        print(f"judge5 {label} (Lw = {Lw} at 2^16 = {Lw / 65536 if Lw else float('nan'):.4f}, RECIP {RECIP.get(name)}): "
              f"in top 5 {inn}/{tot} ({100 * inn / tot:.2f}%), same top word {same}/{tot} ({100 * same / tot:.2f}%)   ({time.time() - t0:.0f}s)")
        print(f"max memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB"); sys.exit()

    print("wedges: " + ", ".join(f"{k}: Lw {v} (L = {v / 65536:.5f}), RECIP {RECIP[k]}" for k, v in LW.items()) + f", shift {S}")
    # mirror check: the counted machine with the trained bend gives the plain machine's stream
    for r in REFS[:2]:
        ids = r["ids"] + r["ref"]; P = len(ids); R0 = M["wte"][ids] + M["wpe"][:P]
        for l in range(12):
            R0 = R0 + J.plain_bank(J.attention(J.gauged_bank(R0, f"qkv.{l}")), f"proj.{l}")
            R0 = R0 + J.plain_bank(J.edge(J.gauged_bank(R0, f"fc.{l}")), f"mproj.{l}")
        Cx = {"plates": 0, "arrival scores": 0, "arrival mix": 0, "out-bank": 0}; Lx = [dict.fromkeys(["piles", "out zero", "light zero"], 0) for _ in range(12)]
        assert np.array_equal(run_counted(ids, J.edge, Cx, Lx), R0)
    print("mirror check: the counted machine with the trained bend gives the plain machine's stream on 2 texts: yes", flush=True)
    KEYS_L = ["piles", "out zero", "light zero", "blocked", "open", "ramp", "ramp out zero", "out-bank, trained light", "dropped"]
    RES, WORDS = {}, {}
    for name, bend in BENDS.items():
        C = dict.fromkeys(["plates", "arrival scores", "arrival mix", "out-bank", "out-bank, trained light on the same piles", "dropped"], 0)
        L = [dict.fromkeys(KEYS_L, 0) for _ in range(12)]; got = []
        for r in REFS:
            ids = r["ids"] + r["ref"]; R = run_counted(ids, bend, C, L, getattr(bend, "Lw", None))
            got.append(read_words(R, len(r["ids"])))
        RES[name] = (C, L); WORDS[name] = got
        print(f"  {name} done ({time.time() - t0:.0f}s)", flush=True)

    ref = [r["ref"] for r in REFS]; base = WORDS["trained"]
    def score(g, t): return sum(int(a == b) for gg, tt in zip(g, t) for a, b in zip(gg, tt))
    def first_departure(g, t):                       # texts whose 16 words all hold
        return sum(int(gg == tt) for gg, tt in zip(g, t))
    print(f"\nTHE 30 REFS TEXTS, 16 WORDS EACH (480 words, every word read with the true words before it)")
    print(f"  {'bend':28s} {'= fp32 GPT-2 words':>20s} {'= unchanged base-2 machine':>28s} {'texts with all 16 = machine':>30s}")
    for name in BENDS:
        lab = "LHALF (sensitivity only)" if name == "LHALF" else ("trained (the machine)" if name == "trained" else name)
        print(f"  {lab:28s} {score(WORDS[name], ref):>14d}/480 {score(WORDS[name], base):>22d}/480 {first_departure(WORDS[name], base):>24d}/30")

    Ct, Lt = RES["trained"]
    whole_t = Ct["plates"] + Ct["arrival scores"] + Ct["arrival mix"]
    print(f"\nTHE TRAINED BEND (the machine itself, baseline) on the same texts:")
    print("  per layer, neuron outputs exactly 0 / out-bank light exactly 0 (after the 16-bit water level):")
    print("   " + "  ".join(f"L{l}: {100 * d['out zero'] / d['piles']:.2f}/{100 * d['light zero'] / d['piles']:.2f}%" for l, d in enumerate(Lt)))
    print(f"  photon meetings, whole machine (48 plates + the arrival's scores and mixes; the top read not counted): {whole_t:,}")
    print(f"     plates {Ct['plates']:,}  (out-bank {Ct['out-bank']:,}),  arrival scores {Ct['arrival scores']:,},  arrival mix {Ct['arrival mix']:,}")
    for name in LW:
        C, L = RES[name]; Lw = LW[name]
        lab = f"{name}, L = {Lw / 65536:.4f}" + ("   [SENSITIVITY ONLY: L halved]" if name == "LHALF" else "")
        print(f"\nWEDGE {lab}")
        print(f"  {'layer':>5s} {'blocked':>8s} {'open':>8s} {'ramp':>8s} {'ramp->0':>8s} {'out 0':>8s} {'light 0':>8s} {'trained light 0':>16s}  out-bank photon pairs dropped exactly (same piles, trained bend)")
        tb = to = tr = tp = 0
        for l, d in enumerate(L):
            p = d["piles"]; tb += d["blocked"]; to += d["open"]; tr += d["ramp"]; tp += p
            print(f"  {l:5d} {100 * d['blocked'] / p:7.2f}% {100 * d['open'] / p:7.2f}% {100 * d['ramp'] / p:7.2f}% {100 * d['ramp out zero'] / p:7.3f}% "
                  f"{100 * d['out zero'] / p:7.2f}% {100 * d['light zero'] / p:7.2f}% {100 * Lt[l]['light zero'] / Lt[l]['piles']:15.2f}%  "
                  f"{100 * d['dropped'] / d['out-bank, trained light']:6.2f}% of the out-bank's pairs")
        print(f"  {'all':>5s} {100 * tb / tp:7.2f}% {100 * to / tp:7.2f}% {100 * tr / tp:7.2f}%")
        zl = sum(d["light zero"] for d in L); zt = sum(d["light zero"] for d in Lt)
        print(f"  out-bank meetings (position x neuron x out-line) with no light, exactly zero: {100 * zl / tp:.2f}%  "
              f"(blocked alone {100 * tb / tp:.2f}%; the trained bend: {100 * zt / tp:.2f}%)")
        same_piles_whole = C["plates"] - C["out-bank"] + C["out-bank, trained light on the same piles"] + C["arrival scores"] + C["arrival mix"]
        print(f"  photon meetings that drop out exactly (blocked neurons, counted as the trained bend would pour on the same piles):")
        print(f"     {C['dropped']:,} = {100 * C['dropped'] / C['out-bank, trained light on the same piles']:.2f}% of the out-bank's, "
              f"{100 * C['dropped'] / same_piles_whole:.2f}% of the whole machine's ({same_piles_whole:,})")
        whole_w = C["plates"] + C["arrival scores"] + C["arrival mix"]
        print(f"  whole-run comparison, wedge machine vs the machine: photon meetings {whole_w:,} vs {whole_t:,} "
              f"({100 * (whole_w - whole_t) / whole_t:+.2f}%); out-bank {C['out-bank']:,} vs {Ct['out-bank']:,} ({100 * (C['out-bank'] - Ct['out-bank']) / Ct['out-bank']:+.2f}%)")
    print(f"\nmax memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB, {time.time() - t0:.0f}s")
