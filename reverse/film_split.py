"""THE NEWEST FILM, STEP 1: THE EXACT SPLIT. Is the answer written on a few grains of the newest film or on all of them?

The newest film = the bends of the newest column (the last position t, all 12 layers, 3,072 shares each). The lead =
the answer's score minus the 2nd word's score (the machine's own whole numbers at t; the answer and 2nd word are the
machine's own first and second in its race). The lead's row at the top (each word's row with the top's dimming for it)
is sent back down every step with every dimming held (film_lead.py, machine.py's reverse steps, started from the lead's
row). At layer l the light stands at the bend outputs; with every dimming held the lead is exactly

    lead = [sum over n of part_n]  +  [the bend outputs of the earlier positions met with their light]
           +  [the stream after layer l's arrival (Rmid_l) met with the light above layer l]  +  c_now

    part_n = (reverse lead-light at neuron n's bend output at t) x (neuron n's bend output at t)

c_now = every bias and every rounding met so far (top's standing amounts and rounding, every step above layer l,
mproj.l's bias and its two roundings). The first bracket is THE PART OF THE LEAD THAT PASSES THROUGH LAYER l's NEWEST
BEND OUTPUTS: the light leaving layer l's 3,072 newest bends, carried by the held dimmings of everything above to the
lead. It splits over the 3,072 neurons with no remainder.

Checks:
  (a) unrounded whole numbers (texts 0, 1, 2): the copy of the reverse equals machine.py's reverse_exact (words mode),
      the lead's reverse equals the answer's minus the 2nd word's; at every layer: the 3,072 parts + the other three
      terms == the machine's own lead, digit for digit (whole numbers over one power of two), and the stream above
      layer l met with its light + the constants so far == the lead.
  (b) the 16-place style (all 30 texts): the copy equals machine.py's reverse16 (words mode); at every layer the 3,072
      parts add back exactly to the part as met at the out-bank's pile (light at mproj.l's pile met with the pile,
      minus mproj.l's in-rounding met with its light); how far the 16-place part is from the exact one (texts 0-2), and
      how far the 16-place sum of all terms is from the lead.
Saved for steps 2-4 (reverse/film_split.npz): the 16-place light at each newest bend output, the newest bend outputs
and shares, and the token-alone column's newest bend outputs and shares (swap_alone.py's construction: position t's
arrival over only itself, everything else by the light).

PREDICTION (Claude's, written before running, for the whole film_ series): the lead is spread, half of it in about a
tenth of the neurons per layer, only modestly above the shuffle; the top-100 grains share little across texts.
For this step: every check exact by construction (36/36 layers exact, 360/360 16-place splits exact).
    python3 reverse/film_split.py       -> reverse/film_split.npz, reverse/film_split.log
"""
import sys, time; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/reverse")
import numpy as np
np.seterr(all="ignore")                                # numpy's float products on this Mac raise spurious flags; checked exact
from film_lead import *
from fractions import Fraction

t0 = time.time(); NT = int(sys.argv[1]) if len(sys.argv) > 1 else len(TEXTS)
EXACT_TEXTS = (0, 1, 2)
OUT = dict(GboN=np.zeros((NT, 12, 3072), np.int64), Gboe=np.zeros((NT, 12), np.int64),
           Bout=np.zeros((NT, 12, 3072), np.int64), share=np.zeros((NT, 12, 3072), np.int64),
           Bout_al=np.zeros((NT, 12, 3072), np.int64), share_al=np.zeros((NT, 12, 3072), np.int64),
           ans=np.zeros(NT, np.int64), second=np.zeros(NT, np.int64), top5=np.zeros((NT, 5), np.int64),
           lead=np.zeros(NT, np.int64), part16=np.zeros((NT, 12)), gap16_lead=np.zeros(NT), gap16_bo=np.zeros((NT, 12)),
           gap16_in=np.zeros((NT, 12)))
WRITE = {}
EX = {}                                                # exact parts for texts 0-2: (text, layer) -> (list of whole numbers, exponent)

def unrounded(i, rec, S, a, b, res):
    """step (a): exact, no rounding anywhere"""
    P, t = rec["P"], rec["t"]; lead = int(S[a]) - int(S[b])
    # the copy against machine.py's own reverse_exact, words mode
    Gm, cm = reverse_exact(rec, [a, b], res)
    Gw, cw = rev_exact_from(rec, *top_exact(rec, [a, b]), res)
    same_copy = Gm.e == Gw.e and all(int(x) == int(y) for x, y in zip(Gm.N.ravel(), Gw.N.ravel())) and cm == cw
    caps = {}
    def cap(stage, l, **kw): caps[(stage, l)] = kw
    GL, cL = rev_exact_from(rec, *top_exact(rec, [a, b], lead=True), res, cap)
    same_lead = GL.e == Gw.e and all(int(x) == int(y) - int(z) for x, y, z in zip(GL.N[0].ravel(), Gw.N[0].ravel(), Gw.N[1].ravel())) \
        and cL[0] == cw[0] - cw[1]
    bottom = meet_exact(rec["R0"], GL)[0] + cL[0]
    print(f"  text {i}: {''.join(word(x) for x in rec['ids'])!r}  answer {word(a)!r} ({int(S[a])}), 2nd {word(b)!r} ({int(S[b])}),"
          f" lead {lead} units", flush=True)
    print(f"     copy of the reverse = machine.py's reverse_exact (both words, every whole number and constant): {same_copy};"
          f"  lead's reverse = answer's minus 2nd's: {same_lead};  embedding numbers met with R_lead + c_lead = {bottom}"
          f" ({'EXACT' if bottom == lead else 'DIFFERENT'})")
    ok_all = same_copy and same_lead and bottom == lead
    print("     layer | part through the newest bend outputs (units) | part / lead | 3,072 parts + earlier positions + Rmid + c_now"
          " == lead | stream above + c == lead | whole numbers: places of sum of parts, places of the power of two")
    for l in range(12):
        Ly = rec["layers"][l]; Bout = Ly["bend"]["out"]; Rmid = Ly["Rmid"]
        ci = caps[("in", l)]; cb = caps[("bo", l)]
        stream_ok = ci["G"].meet(stream_after(rec, l), np.zeros((P, 768), np.int64))[0] + ci["c"][0] == lead
        Gbo = cb["Gbo"]; E2 = Gbo.e                                       # light = Gbo.N x 2^E2
        pn = [int(g) * int(x) for g, x in zip(Gbo.N[0, t, :], Bout[t])]   # the 3,072 parts' whole numbers (x 2^E2)
        Psum = sum(pn)
        Bz = Bout.copy(); Bz[t] = 0
        earlier = Gbo.meet(Bz, np.zeros((P, 3072), np.int64))[0]
        mid = cb["G"].meet(Rmid, np.zeros((P, 768), np.int64))[0]
        rest = earlier + mid + cb["c"][0]
        part = Fraction(Psum) * Fraction(2) ** E2
        # digit for digit in whole numbers: everything on the power 2^E2 (rest's denominator divides 2^-E2)
        D = max(-E2, rest.denominator.bit_length() - 1, 0)              # every term on the power 2^-D (all denominators are powers of two)
        PsumD = Psum << (E2 + D); restN = rest * (1 << D); assert restN.denominator == 1 and (E2 + D) >= 0
        lhs = PsumD + int(restN); rhs = lead << D
        exact = (lhs == rhs) and (part + rest == lead)
        ok_all &= bool(exact) and bool(stream_ok)
        EX[(i, l)] = (pn, E2)
        print(f"     {l:5d} | {float(part):14.1f} | {float(part) / lead:9.3f} | {'EXACT' if exact else 'DIFFERENT'}"
              f" | {'EXACT' if stream_ok else 'DIFFERENT'} | {Psum.bit_length()} places, over 2^{D}", flush=True)
        if i == 0 and l == 11: WRITE.update(Psum=PsumD, restN=int(restN), E2=-D, lead=lead)
    if i == 0:
        w_ = WRITE
        print(f"     text 0, layer 11, written out in whole numbers (every term over 2^{-w_['E2']}):\n"
              f"        sum of the 3,072 parts      = {w_['Psum']}\n"
              f"        + the other three terms     = {w_['restN']}\n"
              f"        = {w_['Psum'] + w_['restN']}\n"
              f"        lead x 2^{-w_['E2']}            = {w_['lead'] << -w_['E2']}\n"
              f"        equal: {w_['Psum'] + w_['restN'] == w_['lead'] << -w_['E2']}")
    return ok_all

ok_exact = True; ok16 = 0; n16 = 0; copy16 = 0; t1 = time.time()
for i in range(NT):
    ids = TEXTS[i]; rec = record(ids); S = machine_scores(ids)
    assert (rec["top"]["scores"] == S).all() and race(S) == forward(ids)
    order = np.argsort(-S, kind="stable"); a, b = int(order[0]), int(order[1]); assert a == race(S)
    P, t = rec["P"], rec["t"]; lead = int(S[a]) - int(S[b])
    OUT["ans"][i], OUT["second"][i], OUT["top5"][i], OUT["lead"][i] = a, b, order[:5], lead
    res = residues(rec)
    if i in EXACT_TEXTS:
        if i == EXACT_TEXTS[0]: print("================ (a) UNROUNDED WHOLE NUMBERS ================")
        ok_exact &= unrounded(i, rec, S, a, b, res)
        if i == EXACT_TEXTS[-1]: print(f"  all exact on texts {EXACT_TEXTS}: {ok_exact}   ({time.time() - t0:.0f}s)\n"
                                       "================ (b) THE 16-PLACE STYLE, ALL 30 TEXTS ================", flush=True)
    kap = kappas(rec, res)
    (Nm, em), cm = reverse16(rec, [a, b], kap, res)
    (Nw, ew), cw = rev16_from(rec, *top16(rec, [a, b]), kap, res)
    copy16 += bool((Nm == Nw).all() and (em == ew).all() and cm == cw)
    caps = {}
    def cap(stage, l, **kw): caps[(stage, l)] = kw
    (NL, eL), cL = rev16_from(rec, *top16(rec, [a, b], lead=True), kap, res, cap)
    OUT["gap16_lead"][i] = float(frac(meetE(rec["R0"], NL, eL)[0] + cL[0]) - lead)
    al = token_alone(rec); ok_text = 0
    for l in range(12):
        Ly = rec["layers"][l]; Bout = Ly["bend"]["out"]; mp = Ly["mproj"]
        cb = caps[("bo", l)]; (Nb, eb) = cb["Gbo"]; (Np, ep) = cb["Gp"]
        pn = [int(g) * int(x) for g, x in zip(Nb[0, t], Bout[t])]; Psum = sum(pn)
        part = Fraction(Psum) * Fraction(2) ** int(eb[0, t])
        rn, rx = res[0][l]["mproj_x"]; rx = np.broadcast_to(rx, rn.shape)
        at_pile = Fraction(dot_int(Np[0, t], mp["pile"][t])) * Fraction(2) ** int(ep[0, t]) \
            - sum((Fraction(int(g) * int(r)) * Fraction(2) ** (int(ep[0, t]) + int(x)) for g, r, x in zip(Nb[0, t], rn[t], rx[t])), Fraction(0))
        n16 += 1; ok16 += (part == at_pile); ok_text += (part == at_pile)
        OUT["part16"][i, l] = float(part)
        # how far the 16-place sum of all terms is from the lead (the 16 places' own rounding)
        ci = caps[("in", l)]; (Ng, eg) = ci["G"]
        OUT["gap16_in"][i, l] = float(meet16_frac(Ng[0], eg[0], stream_after(rec, l)) + frac(ci["c"][0]) - lead)
        Bz = Bout.copy(); Bz[t] = 0
        tot = part + meet16_frac(Nb[0], eb[0], Bz) + meet16_frac(cb["G"][0][0], cb["G"][1][0], Ly["Rmid"]) + frac(cb["c"][0])
        OUT["gap16_bo"][i, l] = float(tot - lead)
        OUT["GboN"][i, l] = Nb[0, t]; OUT["Gboe"][i, l] = eb[0, t]
        OUT["Bout"][i, l] = Bout[t]; OUT["share"][i, l] = Ly["bend"]["share"][t]
        OUT["Bout_al"][i, l] = al["layers"][l]["bend"]["out"][t]; OUT["share_al"][i, l] = al["layers"][l]["bend"]["share"][t]
        if i in EXACT_TEXTS:
            pe, E2 = EX[(i, l)]; pex = Fraction(sum(pe)) * Fraction(2) ** E2
            top_e = set(sorted(range(3072), key=lambda k: -abs(pe[k]))[:100])
            top_16 = set(sorted(range(3072), key=lambda k: -abs(pn[k]))[:100])
            sg = sum(1 for x, y in zip(pe, pn) if (x > 0) == (y > 0) and (x < 0) == (y < 0))
            if l == 11 or l == 0 or l == 6:
                print(f"  text {i} layer {l:2d}: 16-place part {float(part):12.1f} vs exact {float(pex):12.1f} (gap {float(part - pex):8.1f} units,"
                      f" {abs(float((part - pex) / pex)):.1e} of it);  top-100 neurons by |part| shared 16-place vs exact: {len(top_e & top_16)}/100;"
                      f" parts with the same sign {sg}/3072", flush=True)
    print(f"  text {i:2d} {''.join(word(x) for x in ids)!r:42s} answer {word(a)!r:12s} 2nd {word(b)!r:10s} lead {lead:7d} | copy = reverse16 "
          f"{bool((Nm == Nw).all() and (em == ew).all() and cm == cw)} | parts add back exactly at all 12 layers: "
          f"{ok_text == 12} | 16-place sum of all terms - lead: bottom {OUT['gap16_lead'][i]:.0f}, max over layers "
          f"{np.abs(OUT['gap16_bo'][i]).max():.0f} units  ({time.time() - t0:.0f}s)", flush=True)

np.savez("/Users/abundancemachine/streaming-machine/reverse/film_split.npz", **OUT)
print(f"\n16-place copy = machine.py's reverse16 (words mode, both words, every whole number and constant): {copy16}/{NT}")
print(f"16-place: the 3,072 parts add back exactly to the part met at mproj's pile: {ok16}/{n16}")
ld = OUT["lead"].astype(float)
print(f"16-place sum of all terms - lead (units of 2^-16): at the embedding numbers median {np.median(np.abs(OUT['gap16_lead'])):.0f}, max "
      f"{np.abs(OUT['gap16_lead']).max():.0f};  at the bend outputs, all layers: median {np.median(np.abs(OUT['gap16_bo'])):.0f}, max "
      f"{np.abs(OUT['gap16_bo']).max():.0f};  the lead itself: median {np.median(ld):.0f}, min {ld.min():.0f}")
print("part through the newest bend outputs / lead, per layer (16 places; median over 30 texts, [10th, 90th]):")
r = OUT["part16"] / ld[:, None]
for l in range(12):
    print(f"   layer {l:2d}: {np.median(r[:, l]):8.3f}  [{np.percentile(r[:, l], 10):8.3f}, {np.percentile(r[:, l], 90):8.3f}]   "
          f"positive {int((r[:, l] > 0).sum())}/{NT}")
print(f"({time.time() - t0:.0f}s)")
