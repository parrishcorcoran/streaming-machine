"""THE BEND AS A PURE WEDGE, and what it buys.

The machine's bend dims each neuron's light by a share read from a table (the trained share). Here that table is
replaced at every layer by a wedge on the pile p (the bend's input, in the machine's own units, value = p / 2^16):

    value <= -L      output 0                               blocked: the neuron pours nothing
    value >= +L      output = value                         open: the light passes whole
    in between       output = value x (value + L) / (2L)    the light dimmed in proportion to how far into the wedge

Two wedges, both fixed before anything ran, no others:
    L1    L = sqrt(2 pi) / 2 = 1.25331   the wedge's share has slope 1/(2L) at 0 = the trained share's slope 1/sqrt(2 pi)
    L99   L = 2.32635                    the trained share's own 1% and 99% points
and, ONLY as a sensitivity reading (L is one number for every neuron, so a shuffle of per-neuron constants does not
apply), LHALF = L1 / 2.

Whole numbers: Lw = L at 2^16, a fixed whole number. In the ramp, output = (p x (p + Lw) x RECIP) shifted down 40
places, rounded, with RECIP = 2^40 / (2 Lw) rounded once: the division by 2L as a multiply by a fixed whole number and
one shift. Blocked and open are set outright, no product.

    python3 wedge/wedge_machine.py                  30 refs texts, 16 words each (each read with the true words before
                                                    it), and the exact counts under each wedge
    python3 wedge/wedge_machine.py judge=L1         judge5: 300 windows, 6,000 positions, top 5 of the full machine
"""
import sys, os, json, time, math, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")   # spurious flags from the float matmul; J.product checks every sum whole
HERE = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, HERE); os.chdir(HERE)
ARGS = dict(a.split("=") for a in sys.argv[1:])
import judge5 as J
M, W, ES = J.M, J.W, J.ES
to16, gauge_log, power, race = J.to16, J.gauge_log, J.power, J.race
REFS = json.load(open(f"{HERE}/refs_many.json"))
WTE, WPE = M["wte"], M["wpe"]

# ---------------- the wedge, in whole numbers ----------------
SHIFT = 40
L_REAL = {"L1": math.sqrt(2 * math.pi) / 2, "L99": 2.3263478740408408, "LHALF": math.sqrt(2 * math.pi) / 4}
LW = {k: round(v * 65536) for k, v in L_REAL.items()}
RECIP = {k: round(2 ** SHIFT / (2 * v)) for k, v in LW.items()}
for k in LW: assert 2 * LW[k] * LW[k] * RECIP[k] + (1 << (SHIFT - 1)) < 2 ** 62     # the largest ramp product stays inside 64 bits
def make_wedge(name):
    Lw, R = LW[name], RECIP[name]
    def wedge(H):
        Hc = np.clip(H, -Lw, Lw)                                     # only the ramp is multiplied; the ends are set outright
        ramp = J.rshift(Hc * (Hc + Lw) * R, SHIFT)
        return np.where(H <= -Lw, 0, np.where(H >= Lw, H, ramp))
    wedge.Lw = Lw; wedge.R = R; return wedge
BENDS = {"trained": J.edge, **{k: make_wedge(k) for k in LW}}
LABEL = {"trained": "trained bend (the machine)", "L1": "wedge L1", "L99": "wedge L99", "LHALF": "wedge LHALF [SENSITIVITY ONLY: L halved]"}

def checks():                                         # the two constants against the machine's own share table
    PHI = J.PHI; c = 32768                           # PHI index = value x 4096 + 32768, share at 2^16
    slope = (int(PHI[c + 64]) - int(PHI[c - 64])) / (128 / 4096) / 65536
    i99 = c + round(L_REAL["L99"] * 4096); i01 = c - round(L_REAL["L99"] * 4096)
    print(f"checks against the machine's share table: its slope at 0 = {slope:.5f} (1/sqrt(2 pi) = {1 / math.sqrt(2 * math.pi):.5f}, "
          f"wedge L1 share slope 1/(2L) = {65536 / (2 * LW['L1']):.5f}); its share at value -{L_REAL['L99']:.4f} / +{L_REAL['L99']:.4f} = "
          f"{PHI[i01] / 65536:.5f} / {PHI[i99] / 65536:.5f}")
    for k in LW:                                      # the whole-number wedge against the exact ratio, every pile value in the ramp
        Lw = LW[k]; H = np.arange(-Lw - 3, Lw + 4, dtype=np.int64); out = BENDS[k](H)
        exact = np.where(H <= -Lw, 0.0, np.where(H >= Lw, H.astype(float), H * (H + Lw) / (2.0 * Lw)))
        print(f"  {k}: Lw = {Lw} (L = {Lw / 65536:.6f}), RECIP = {RECIP[k]}, shift {SHIFT}; largest |whole-number wedge - exact ratio| "
              f"over every pile value -Lw-3..Lw+3: {np.abs(out - exact).max():.3f} of one unit (2^-16)")

# ---------------- the forward for judge5 (J.scores_all, step for step, with the bend swapped) ----------------
def forward_for(bend):
    def fwd(ids):
        P = len(ids); R = WTE[ids] + WPE[:P]
        for l in range(12):
            R = R + J.plain_bank(J.attention(J.gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
            R = R + J.plain_bank(bend(J.gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
        w, we, b = W["head"]; Rj = R[J.FIRST:J.LAST]; x16, ex, LS = gauge_log(Rj)
        return power(J.product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b
    return fwd

# ---------------- photons ----------------
_POP = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.int64)
def ones(A):                                          # photons (1s) per value, direction apart
    A = np.abs(np.asarray(A).astype(np.int64)); n = np.zeros(A.shape, np.int64)
    while A.any(): n += _POP[A & 0xFFFF]; A = A >> 16
    return n
PLATES = [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")] + ["head"]
def _wrow(name):                                      # photons of each plate line, summed over everything it feeds
    w = M[name + ".w"]; out = np.zeros(w.shape[0], np.int64)
    for i in range(0, w.shape[0], 64): out[i:i + 64] = ones(w[i:i + 64]).sum(1)
    return out

# ---------------- counting the machine's own products ----------------
# Every whole-number product of the machine goes through J.product. The wrapper returns the machine's own result
# untouched and, when counting is on, adds the photon pairs of that meeting: sum over lines of ones(light) x ones(plate).
_orig_product = J.product; COUNT = {"on": False, "C": None}; WROW = {}; PLATE_OF = {}
def counting_product(x, w):
    p = _orig_product(x, w)
    if COUNT["on"]:
        C = COUNT["C"]; name = PLATE_OF.get(id(w))
        if name is not None:
            kind = name.split(".")[0]; C[kind] += int((ones(x) @ WROW[name]).sum())
        elif x.shape[0] == 1 and w.shape[1] == 64:    # the arrival's mix: one row of arrival weights meets the values
            C["arrival mix"] += int((ones(x) @ ones(w)).sum())
        else:                                         # the arrival's scores: query light meets key light, causal half
            assert x.shape[1] == 64 and w.shape[0] == 64 and w.shape[1] == x.shape[0]
            C["arrival scores"] += int(np.tril(ones(x).astype(np.float64) @ ones(w).astype(np.float64)).sum())
    return p
J.product = counting_product
KINDS = ["qkv", "proj", "fc", "mproj", "arrival scores", "arrival mix", "head"]

def run_counted(ids, n0, bend, C, L):
    """one refs text, every position, counted. Returns the 16 words read at positions n0-1 .. n0+14."""
    COUNT["on"] = True; COUNT["C"] = C
    Lw = getattr(bend, "Lw", None)
    P = len(ids); R = WTE[ids] + WPE[:P]
    for l in range(12):
        R = R + J.plain_bank(J.attention(J.gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        H = J.gauged_bank(R, f"fc.{l}")
        out = bend(H)
        m16, _ = to16(out, -ES)                       # the light the out-bank actually receives
        t = J.edge(H); t16, _ = to16(t, -ES)          # what the trained bend would pour on these same piles
        wr = WROW[f"mproj.{l}"][None, :]
        tw = ones(t16) * wr                           # out-bank photon pairs per (position, neuron), trained light
        d = L[l]; d["piles"] += H.size
        d["out zero"] += int((out == 0).sum()); d["light zero"] += int((m16 == 0).sum())
        d["trained light zero"] += int((t16 == 0).sum()); d["became zero"] += int(((m16 == 0) & (t16 != 0)).sum())
        C["mproj, trained light on the same piles"] += int(tw.sum())
        C["dropped, light zero"] += int(tw[m16 == 0].sum())
        # the bend's own products, in photon pairs
        if Lw is None:
            share_idx = J.rshift(H, ES - 12) + 32768
            share = np.where(share_idx < 0, 0, np.where(share_idx > 65535, 65536, J.PHI[np.clip(share_idx, 0, 65535)]))
            C["bend"] += int((ones(H) * ones(share)).sum())
        else:
            blk, opn = H <= -Lw, H >= Lw; rmp = ~blk & ~opn
            d["blocked"] += int(blk.sum()); d["open"] += int(opn.sum()); d["ramp"] += int(rmp.sum())
            d["ramp out zero"] += int((rmp & (out == 0)).sum())
            assert np.array_equal(out[opn], H[opn]) and not out[blk].any()     # open passes whole, blocked pours nothing
            d["mproj trained light"] += int(tw.sum()); d["dropped blocked"] += int(tw[blk].sum())
            C["dropped, blocked"] += int(tw[blk].sum())
            Hr = H[rmp]; C["bend"] += int((ones(Hr) * ones(Hr + Lw)).sum() + (ones(Hr * (Hr + Lw)) * ones(bend.R)).sum())
        R = R + J.plain_bank(out, f"mproj.{l}")
    w, we, b = W["head"]; Rj = R[n0 - 1:n0 + 15]; x16, ex, LS = gauge_log(Rj)
    S_ = power(J.product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b
    COUNT["on"] = False
    return [race(s) for s in S_], R

def plain_R(ids):                                     # the machine as judge5 runs it, uncounted
    P = len(ids); R = WTE[ids] + WPE[:P]
    for l in range(12):
        R = R + J.plain_bank(J.attention(J.gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        R = R + J.plain_bank(J.edge(J.gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
    return R

if __name__ == "__main__":
    t0 = time.time()
    if "judge" in ARGS:
        name = ARGS["judge"]; n = int(ARGS.get("n", 300))
        fwd = forward_for(BENDS[name]); f0 = forward_for(J.edge)
        for wdw in J.WINDOWS[:2]: assert np.array_equal(f0(wdw["ids"]), J.scores_all(wdw["ids"]))
        print("mirror check: this forward with the trained bend gives the machine's own scores on 2 judge windows: yes", flush=True)
        inn, same, tot = J.judge(forward=fwd, n=n)
        Lw = LW.get(name)
        print(f"judge5, {LABEL[name]} (Lw = {Lw}, L = {Lw / 65536:.5f}, RECIP {RECIP[name]}): in top 5 of the full machine "
              f"{inn}/{tot} ({100 * inn / tot:.2f}%), same top word {same}/{tot} ({100 * same / tot:.2f}%)   ({time.time() - t0:.0f}s)")
        print(f"max memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB"); sys.exit()

    checks()
    for nm in PLATES: WROW[nm] = _wrow(nm); PLATE_OF[id(W[nm][0])] = nm
    print(f"plate photons counted per line ({time.time() - t0:.0f}s)", flush=True)
    KL = ["piles", "out zero", "light zero", "trained light zero", "became zero", "blocked", "open", "ramp", "ramp out zero", "mproj trained light", "dropped blocked"]
    KC = KINDS + ["bend", "mproj, trained light on the same piles", "dropped, light zero", "dropped, blocked"]
    RES, WORDS = {}, {}
    for name, bend in BENDS.items():
        C = dict.fromkeys(KC, 0); L = [dict.fromkeys(KL, 0) for _ in range(12)]; got = []
        for i, r in enumerate(REFS):
            ids = r["ids"] + r["ref"]; g, R = run_counted(ids, len(r["ids"]), bend, C, L); got.append(g)
            if name == "trained" and i < 2: assert np.array_equal(R, plain_R(ids))     # the counted run IS the machine
        RES[name] = (C, L); WORDS[name] = got
        print(f"  {name} done ({time.time() - t0:.0f}s)" + ("  [mirror check on 2 texts: the counted run gives the machine's own stream: yes]" if name == "trained" else ""), flush=True)

    ref = [r["ref"] for r in REFS]; base = WORDS["trained"]
    def score(g, t): return sum(int(a == b) for gg, tt in zip(g, t) for a, b in zip(gg, tt))
    def run_len(gg, tt):                              # words in a row from the start: what a free-running machine would share
        k = 0
        while k < 16 and gg[k] == tt[k]: k += 1
        return k
    print(f"\nTHE 30 REFS TEXTS, 16 WORDS EACH (480 words; each read with the true words before it)")
    print(f"  {'bend':42s} {'= fp32 GPT-2':>13s} {'= base-2 machine':>17s} {'texts all 16 = machine':>23s} {'words in a row = machine, mean':>31s}")
    for name in BENDS:
        rl = [run_len(g, t) for g, t in zip(WORDS[name], base)]
        print(f"  {LABEL[name]:42s} {score(WORDS[name], ref):>9d}/480 {score(WORDS[name], base):>13d}/480 "
              f"{sum(int(x == 16) for x in rl):>19d}/30 {np.mean(rl):>27.2f}/16")

    Ct, Lt = RES["trained"]
    def whole(C, top=True): return sum(C[k] for k in KINDS if top or k != "head")
    wt, wt0 = whole(Ct), whole(Ct, False)
    print(f"\nTHE TRAINED BEND (the machine itself, baseline), same 30 texts:")
    print("  per layer, neuron outputs exactly 0 / out-bank light exactly 0 after its 16-bit water level:")
    print("   " + "  ".join(f"{l}: {100 * d['out zero'] / d['piles']:.2f}/{100 * d['light zero'] / d['piles']:.2f}%" for l, d in enumerate(Lt)))
    print(f"  photon meetings, whole machine: {wt:,}  ({wt0:,} without the top read)")
    print("     " + ", ".join(f"{k} {Ct[k]:,}" for k in KINDS) + f";  the bend's own products {Ct['bend']:,}")
    for name in LW:
        C, L = RES[name]
        print(f"\n{LABEL[name].upper()}, L = {LW[name] / 65536:.5f}")
        print(f"  {'layer':>5s} {'blocked':>8s} {'open':>8s} {'ramp':>8s} | {'ramp->0':>7s} {'out 0':>7s} {'light 0':>8s} {'trained 0':>10s} {'became 0':>9s} | out-bank photon pairs dropped by blocking")
        tb = to = tr = tp = trz = tz = tlz = tbz = 0
        for l, d in enumerate(L):
            p = d["piles"]; tb += d["blocked"]; to += d["open"]; tr += d["ramp"]; tp += p; trz += d["ramp out zero"]
            tz += d["light zero"]; tlz += d["trained light zero"]; tbz += d["became zero"]
            print(f"  {l:5d} {100 * d['blocked'] / p:7.2f}% {100 * d['open'] / p:7.2f}% {100 * d['ramp'] / p:7.2f}% | {100 * d['ramp out zero'] / p:6.3f}% "
                  f"{100 * d['out zero'] / p:6.2f}% {100 * d['light zero'] / p:7.2f}% {100 * d['trained light zero'] / p:9.2f}% {100 * d['became zero'] / p:8.2f}% | "
                  f"{100 * d['dropped blocked'] / d['mproj trained light']:6.2f}%")
        print(f"  {'all':>5s} {100 * tb / tp:7.2f}% {100 * to / tp:7.2f}% {100 * tr / tp:7.2f}% | {100 * trz / tp:6.3f}% {'':>7s} {100 * tz / tp:7.2f}% {100 * tlz / tp:9.2f}% {100 * tbz / tp:8.2f}%")
        print(f"  (shares of the {tp:,} neuron outputs = positions x 3,072 x 12 layers; 'light 0' and 'trained 0' are after the 16-bit water level)")
        print(f"  OUT-BANK MEETINGS (one neuron's light meeting one plate value; each neuron meets 768): exactly zero under the wedge "
              f"{100 * tz / tp:.2f}% (blocked alone {100 * tb / tp:.2f}%); exactly zero under the trained bend on the same piles {100 * tlz / tp:.2f}%; "
              f"became exactly zero {100 * tbz / tp:.2f}%")
        ob = C["mproj, trained light on the same piles"]
        print(f"  OUT-BANK PHOTON PAIRS (trained light on the same piles = {ob:,}): dropped by blocked neurons {C['dropped, blocked']:,} = "
              f"{100 * C['dropped, blocked'] / ob:.2f}%; dropped by every zero light {100 * C['dropped, light zero'] / ob:.2f}%; "
              f"the wedge's actual out-bank pairs {C['mproj']:,} ({100 * (C['mproj'] - ob) / ob:+.2f}% against the trained light on the same piles)")
        same_whole = whole(C) - C["mproj"] + ob; same_whole0 = whole(C, False) - C["mproj"] + ob
        print(f"  WHOLE MACHINE PHOTON MEETINGS (48 banks + arrival scores and mixes + top read at 16 positions; trained light in the out-bank, same piles): "
              f"{same_whole:,}; dropped exactly by blocking {100 * C['dropped, blocked'] / same_whole:.3f}% "
              f"({100 * C['dropped, blocked'] / same_whole0:.3f}% without the top read)")
        print(f"  the wedge machine run against the machine run (different streams): photon meetings {whole(C):,} vs {wt:,} ({100 * (whole(C) - wt) / wt:+.2f}%); "
              f"out-bank {C['mproj']:,} vs {Ct['mproj']:,} ({100 * (C['mproj'] - Ct['mproj']) / Ct['mproj']:+.2f}%); "
              f"the bend's own products {C['bend']:,} vs {Ct['bend']:,}")
    print(f"\nnot counted anywhere: the per-value products (gauge squares, the power-of-two table scalings); they are per value, not per line")
    print(f"max memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB, {time.time() - t0:.0f}s")
