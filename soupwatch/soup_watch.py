"""WATCH THE SOUP (base 2, whole numbers only). For each text of refs_many.json: prompt = its first 5 words, N = 16.

The soup: the prompt and the next 16 words are all in the machine from the start (future slots start as copies of
the last prompt word). Every wave, every layer fires for every position at once, each reading what is below it now;
every future slot's word becomes what the top reads at the position before it, re-read every wave; it runs until no
word changes. Wave w pours the words T_w and reads T_{w+1}.

Predicted before running (each a relation, nothing searched):
  behind the front   d = 0 exactly at all 25 snapshots: each position reads only earlier ones.
  the front          its word is replaced, so its entry d is a different word's light: as thick as the shuffle.
  ahead              entry d = 0 where the word did not change; from the first write on nearly all 768 values move,
                     and d's photons are a large share of the light's (1/2 .. 9/10): a change 5% the size of a
                     value still has every photon below its own top place.
  work               the delta soup can never go below the one-word machine: wave 1 pours all 21 positions whole,
                     already 21/20 of the one-word machine's 20 positions.

Recorded per text (soupwatch/out_<tag>/text_XX.npz):
  W1  the words entering every wave, the one-word machine's words.
  W2  per wave w -> w+1, snapshot, position: values moved, d's photons, the light's photons, the bits that flip
      (|R(w)| XOR |R(w+1)|; a value whose sign flips counts whole: all its old photons and all its new ones), and the
      same against R(w) at a different random position of the same wave and snapshot (shuffle A: any other position;
      shuffle B: any other position except position 0).
  W3  per bank and position: the work of pouring the whole light, of pouring only d (d = the exact change of that
      bank's own input), and of pouring d against a random other position (shuffle A); lines that moved; piles that
      moved; the bend's inputs and outputs that moved; whether the position's gauge moved; the arrival's light-on-light
      meetings, whole and as changes.
  W4  the plain one-word machine run on its own (its own light, its own work) for the same 16 words.

    python3 soupwatch/soup_watch.py A B [lines=1]      texts A..B-1; lines=1 relabels every line and neuron first
"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import soup_lib as L

args = [a for a in sys.argv[1:] if "=" not in a]; opts = dict(a.split("=") for a in sys.argv[1:] if "=" in a)
A, B = int(args[0]), int(args[1]); LINES = opts.get("lines") == "1"; N = 16
tag = "lines" if LINES else "real"; OUT = os.path.join(L.HERE, "soupwatch", f"out_{tag}"); os.makedirs(OUT, exist_ok=True)
refs = json.load(open(L.HERE + "refs_many.json"))
if LINES: L.relabel_lines(11)
elif A == 0:
    pr = (refs[0]["ids"] + refs[0]["ref"])[:5]
    print("light() against base2.py's own functions, bit for bit:", "IDENTICAL" if L.check(pr + [pr[-1]] * N, 5) else "DIFFERENT")

bc = L.bc


def snapstats(new, old):
    d = new - old
    sf = (np.sign(new) * np.sign(old)) < 0
    fl = np.where(sf, bc(new) + bc(old), bc(np.abs(new) ^ np.abs(old)))
    return (d != 0).sum(-1), bc(d).sum(-1), fl.sum(-1), sf.sum(-1)


for t in range(A, B):
    t0 = time.time()
    prompt = (refs[t]["ids"] + refs[t]["ref"])[:5]; S = len(prompt); P = S + N; reads = list(range(S - 1, P - 1))
    ow_words, ow_bank, ow_att, ow_snaps = L.oneword(prompt, N)
    T = prompt + [prompt[-1]] * N; hist = [T]; prev = None; R = {k: [] for k in
        "nzv phd phn pho fl sf nzv_a phd_a fl_a nzv_b phd_b fl_b full delta dsh lmov pmov Hmov Gmov gmov attf attd attsh moving".split()}
    for w in range(1, N + 3):
        cur = L.light(T, reads)
        newT = prompt + cur["tops"]
        full, attf = L.bank_work(cur), L.att_work(cur)
        R["full"].append(full); R["attf"].append(attf)
        if prev is None:
            R["delta"].append(full); R["dsh"].append(full); R["attd"].append(attf); R["attsh"].append(attf); R["moving"].append(np.ones(P, bool))
        else:
            g = np.random.default_rng(1000 * t + w)
            qa = (np.arange(P) + g.integers(1, P, size=P)) % P                         # any other position
            qb = np.array([g.choice([x for x in range(1, P) if x != p]) for p in range(P)])   # any other, never position 0
            a, b = cur["snaps"], prev["snaps"]
            for k, v in zip("nzv phd fl sf".split(), snapstats(a, b)): R[k].append(v)
            R["phn"].append(bc(a).sum(-1)); R["pho"].append(bc(b).sum(-1))
            for sfx, q in (("a", qa), ("b", qb)):
                nz, ph, fl, _ = snapstats(a, b[:, q]); R["nzv_" + sfx].append(nz); R["phd_" + sfx].append(ph); R["fl_" + sfx].append(fl)
            R["delta"].append(L.bank_delta(cur, prev)); R["dsh"].append(L.bank_delta(cur, prev, qa))
            R["lmov"].append(L.lines_moved(cur, prev))
            R["pmov"].append(np.stack([(cur["piles"][n] != prev["piles"][n]).sum(1) for n in L.BANKS[:48]], 1))
            R["Hmov"].append((cur["H"] != prev["H"]).sum(-1)); R["Gmov"].append((cur["G"] != prev["G"]).sum(-1))
            R["gmov"].append(np.stack([(cur["gauge"][f"fc.{l}"][0] != prev["gauge"][f"fc.{l}"][0]) | (cur["gauge"][f"fc.{l}"][1] != prev["gauge"][f"fc.{l}"][1]) for l in range(12)]))
            R["attd"].append(L.att_delta(cur, prev)); R["attsh"].append(L.att_delta(cur, prev, qa))
            R["moving"].append((a != b).any(-1).any(0))
        hist.append(newT)
        if newT == T: break
        prev, T = cur, newT
    nw = len(hist) - 1
    same_light = np.array_equal(ow_snaps, cur["snaps"][:, :P - 1])
    out = {k: np.array(v) for k, v in R.items()}
    np.savez_compressed(os.path.join(OUT, f"text_{t:02d}.npz"), T=np.array(hist), ow_words=np.array(ow_words), S=S, P=P,
                        ow_bank=ow_bank, ow_att=ow_att, same_light=same_light, **out)
    right = [sum(int(x == y) for x, y in zip(h[S:], ow_words)) for h in hist]
    print(f"text {t:2d}: {nw:2d} waves; one-word words reached: {'YES' if hist[-1][S:] == ow_words else 'NO'}; "
          f"one-word light = soup's final light at every snapshot: {'YES' if same_light else 'NO'}; words right per wave: {right[1:]}  ({time.time() - t0:.1f} s)", flush=True)
