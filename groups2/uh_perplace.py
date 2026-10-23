"""EACH PLATE PLACE UP/HOLE OR SIGNED DIGITS, CHOSEN BY THE PLATE ALONE (groups2/, the session's own step; the builder's
proposal, carried out with this file's own rule, written here before any run).

HIS PICTURE, AS THE SESSION RESTATES IT (the session's words, not his)
  In the up/hole form every place of every weight is a photon up or a hole, so every (neuron, group) cell at every place
  is read: one ready pile of all g lines' light. Where a place of the plate is nearly dark in signed digits (the top
  places, where only the few large weights reach, and on Qwen o and down the bottom places too), most cells would hold
  no lit digit at all, and the up/hole form still reads one pile there for every cell. So each place of each plate is
  read one way or the other: up/hole where its cells are lit, signed digits where they are dark. The choice is made by
  the plate alone, before any light is seen.

THE RULE (fixed here, before any light is seen; from the counting relation, not from results)
  For each plate (a GPT-2 bank of one layer; a Qwen3-0.6B q, k, v, o, gate, up or down plate of one layer) and each
  place b = 0..15:
    L_b = how many of the plate's weights have a lit signed digit at place b (the plate's own signed digits, by the
          classic rule: where the running value is odd, the digit is 2 - (value mod 4); take it away; halve);
    U_b = G x M, the reads the place costs in up/hole form (one ready pile per neuron per group; G = the number of
          groups, ceil(N / g); M = the plate's neurons).
  The place is read in SIGNED DIGITS when L_b < U_b, and UP/HOLE otherwise.
  The counting relation behind it: at an up/hole place every cell is one add of one ready pile; at a signed-digit
  place every lit digit is one add of its own light value (the light meets that digit directly, as in the plain
  signed-digit machine). The place takes the form with fewer adds. Equivalently: signed digits where the plate has
  fewer than one lit digit per (neuron, group) cell at that place, a lit share below G / N = 1/g; with digits lit
  independently that is an empty-cell share above (1 - 1/g)^g (34.4% at g = 8, 35.2% at g = 12). The relation counts
  adds, not photons: a ready pile of g lines has somewhat more signed digits than one light value, so the rule leans a
  little toward up/hole. That lean is kept as it is; nothing is tuned.
  No group table is built for the signed-digit places: a cell there holds fewer than one lit digit on average, so a
  ready pile would mostly be one light value. The up/hole half table is built as before (the flip gives the other half).

HOW THE DIGITS STAND (exact)
  Each weight is written once, bottom place to top: at an up/hole place a binary digit of the running value, at a
  signed-digit place a signed digit by the rule above (places 16 and 17 take whatever is left, as signed digits), with
  the offset C = 2^(the plate's top up/hole place). Then 2 x.w = sum over up/hole places of 2^b x (ready pile) +
  2 x (signed-digit places' light) + (K - 2C) X, K = the sum of 2^b over the up/hole places, X the light summed. With
  the up/hole places running from place lo to the top one, K - 2C = -2^lo: X shifted, taken once from each neuron's
  doubled pile (X's signed digits); the halving is a shift. Checked: the digits give every weight back, and every
  pile equals the machine's own pile.

THE WORK COUNTED (photon meetings, as in uh_rebuild.py): the up/hole half table; every up/hole read (the ready pile's
  signed digits); every lit signed digit (its light value's signed digits); X once a position; X taken from each
  neuron's doubled pile. Against plain photon meetings of the same plates and light.

ANALYSIS BEFORE THE RUN (the session's; plate-only counts, and the builder's earlier up/hole records for the read sizes)
  The rule, applied to the plates (no light): GPT-2 at g = 8 sends places 14 and 15 to signed digits on 47 of 48 banks
  (place 15 alone on one); at g = 12 place 15 on every bank and place 14 on 33 of 48. Qwen3-0.6B: q, k, v, gate, up send
  14 and 15 (sometimes 13 too at g = 8, sometimes only 15 at g = 12); o and down send 0, 1, 2 (and 3 at g = 8) and
  14-15 or 15. Each place taken from up/hole saves 1/16 of that plate's up/hole reads and costs its lit digits' light.
PREDICTION (the session's, written before this file ran)
  Whole machine, table included, as a share of plain photon meetings (the up/hole form alone in brackets):
    GPT-2 g = 8   about 26.0% (25.0-27.0)   [28.49%]
    GPT-2 g = 12  about 19.6% (19.0-20.3)   [21.04%]
    Qwen  g = 8   about 28.6% (27.5-30.0)   [33.42%]; o about 32%, down about 34%
    Qwen  g = 12  about 21.3% (20.3-22.3)   [24.01%]; o about 23.5%, down about 25%
  The saving is largest on Qwen o and down (their dark low places), smallest on GPT-2 at g = 12 (only the top place
  or two are dark enough). Exact everywhere. Second seed within 0.02 point; light shuffled across lines within
  0.1 point (the signed-digit places meet single light values, so a line that carries large light and many lit top
  digits could move it a little); order-free: piles, up/hole reads and signed-digit reads identical, the table within
  0.01 point.
RULE 13, CHECKED BEFORE THE RUN
  13.1 the measure is the work of the finished piles, each pile exact; not the light's 1s.
  13.2 the base is the machine's own plates and light; nothing in the machine is reshaped: the plate is only written in
       a different exact digit form place by place.
  13.3 the shuffle control is a count of work, not an equality statistic.
  13.4 the rule fixes which places are read which way before the run (plate only); the run measures the photons of
       the reads, the table, and exactness.
  13.5 the picture and the rule are the session's, labelled so.
  13.6 every GPT-2 bank and every Qwen3-0.6B plate, both g, every position.

    python3 groups2/uh_perplace.py model=gpt2|qwen g=8|12 [layers=a-b] [runs=main,seed2,shuffle,order]
    -> groups2/out/uhp_<model>_g<g>[_L<a>-<b>].json, .log
"""
import os, sys, json, time, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/groups2")
import uh_rebuild as R

def lit_by_place(w):
    """the plate's own signed digits, place by place: how many weights are lit at each place (0..17)"""
    r = np.asarray(w, np.int64).copy(); out = []
    for b in range(18):
        odd = (r & 1) == 1; d = np.where(odd, 2 - (r & 3), 0); out.append(int((d != 0).sum())); r = (r - d) >> 1
    assert not r.any()
    return out

def rule(w, segs, g):
    """the up/hole places of every neuron, fixed by the plate alone (the rule in the header)"""
    N, M = w.shape; G = -(-N // g); up = np.zeros(M, np.int64); chosen = {}
    for name, a, b in segs:
        L = lit_by_place(w[:, a:b]); U = G * (b - a)
        assert all(v == 0 for v in L[16:]), "a signed digit above place 15"
        mask = sum(1 << p for p in range(16) if not L[p] < U); up[a:b] = mask
        chosen[name] = dict(lit=L[:16], U=U, signed_places=[p for p in range(16) if L[p] < U])
    return up, chosen

if __name__ == "__main__":
    A = dict(s.split("=") for s in sys.argv[1:]); MODEL = A.get("model", "gpt2"); g = int(A.get("g", 8))
    RUNS = A.get("runs", "main,seed2,shuffle,order").split(","); lay = A.get("layers")
    L0, L1 = (int(v) for v in lay.split("-")) if lay else (0, 99)
    tag = f"uhp_{MODEL}_g{g}" + (f"_L{L0}-{L1}" if lay else ""); os.makedirs(R.OUT, exist_ok=True)
    LOG = open(f"{R.OUT}/{tag}.log", "w"); t0 = time.time()
    def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
    log(f"EACH PLACE UP/HOLE OR SIGNED DIGITS, BY THE PLATE ALONE (groups2): {MODEL}, g = {g}, runs {RUNS}, layers {L0}..{L1}")
    res = {r: {} for r in RUNS}; RULE = {}
    for l, slab, x, w, segs, ref in (R.gpt2_plates(L0, L1) if MODEL == "gpt2" else R.qwen_plates(L0, L1)):
        tl = time.time(); N, M = w.shape; key = [1, R.SLABS.index(slab), l, g]; main_pile = None
        up, chosen = rule(w, segs, g)
        for name in chosen: RULE[f"{name}.{l}"] = chosen[name]
        prep = R.prepare(w, up, segs); msg = []
        for run in RUNS:
            k = list(key); xr = x; reo = None
            if run == "seed2": k[0] = 2
            if run == "order": reo = [4242] + key
            if run == "shuffle": xr = np.random.default_rng([31337] + key).permuted(x, axis=1)
            glines, goff = R.make_groups(N, g, k, reo)
            pile, recs, sl = R.run_plate(xr, prep, glines, goff)
            if run == "shuffle": bad = int((pile != R.plain_product(xr, w)).any(1).sum())
            else: bad = R.piles_differ(pile, ref)
            assert bad == 0, (run, slab, l, bad)
            if run == "main": main_pile = pile
            if run == "order": assert np.array_equal(pile, main_pile), "order-free piles differ from main"
            for name, r in recs.items():
                assert r["odd"] == 0; r["piles_differ"] = bad; res[run][f"{name}.{l}"] = r
            res[run][f"{slab}.{l}.slab"] = sl
            msg.append(f"{run}: " + " ".join(f"{n} {100 * (recs[n]['read'] + recs[n]['sdread']) / recs[n]['plain']:.2f}" for n in recs))
            del pile
        sp = " ".join(f"{n}:{','.join(map(str, chosen[n]['signed_places']))}" for n in chosen)
        log(f"  layer {l:2d} {slab:5s} exact (every run); signed places {sp}; reads % of plain: " + " | ".join(msg)
            + f"  ({time.time() - tl:.0f}s, {R.rss_mb()} MB)")
        json.dump(dict(model=MODEL, g=g, runs=res, rule=RULE), open(f"{R.OUT}/{tag}.json", "w"))
    log(f"done ({time.time() - t0:.0f}s, max memory {R.rss_mb()} MB)")
