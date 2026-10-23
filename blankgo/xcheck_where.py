"""Checker extra (independent checking session): where.py's one-place-at-a-time numbers, which the build session marked
UNCHECKED, by the checker's own float64 model and its own scrambles.
    python3 xcheck_where.py

Step 3000, all 12 builds. For each place (wte input word table, wpe position rows, plates = every qkv/proj/fc/mproj
plate, head output word table): held-out loss on the 128 windows with that place's blank taken out (P alone there,
everything else W), and with that place's blank scrambled within each neuron (the checker's own 3 draws).

PREDICTIONS (the checker's, written before this script ran; they are the build session's numbers):
  H1. the 'out' losses within 1e-9 of where.py's (same definition, deterministic)
  H2. w64 L2 B: wte +0.207, wpe +0.103, plates +0.320, head +0.072; w128 L4 B: wte +0.189, wpe +0.060, plates +0.350,
      head +0.152; w128 L4 A: wte +0.007, wpe +0.006, plates +3.941, head +0.060
  H3. B and C: in every place, each of the checker's own scrambles costs at least as much as taking that place's blank
      out (the build session's claim, made with its own 3 draws)
"""
import os, json, time, numpy as np
import xcheck_lib as X

held = X.held_part(); WINS = X.loss_windows(held); out = {}
for d, L in ((64, 2), (64, 4), (128, 2), (128, 4)):
    sh = f"w{d}_L{L}"; names = [n for n, _, _ in X.part_list(d, L)]; drawn = X.drawn_names(d, L); B0 = X.redraw_blank(d, L)
    PLACES = {"wte": ["wte"], "wpe": ["wpe"], "plates": [n for n in drawn if n.endswith(".w")], "head": ["head"]}
    for rc in "ABC":
        nm = f"{sh}_{rc}"; t0 = time.time()
        log = json.load(open(os.path.join(X.HERE, f"build_{nm}.json")))
        WH = json.load(open(os.path.join(X.HERE, "evals", f"where_{nm}.json")))["snaps"]["3000"]
        Bt = X.shaved_blank(B0, drawn, log["peak"], log["wd"], 3000)
        z = np.load(os.path.join(X.HERE, "snaps", f"{nm}_step3000.npz"))
        P = {n: z["P." + n] for n in names}; W = {n: Bt[n] + P[n] for n in names}
        assert X.fingerprint(W, names) == log["snaps"]["3000"]["hash_W"]
        fin = X.float_loss(W, WINS, d, L); rec = dict(finished=fin)
        for place, ks in PLACES.items():
            Wo = dict(W); Wo.update({n: P[n] for n in ks}); lo = X.float_loss(Wo, WINS, d, L)
            scr = []
            for k in (1, 2, 3):
                Ws = dict(W); Ws.update({n: P[n] + X.scr_within_neuron(n, Bt[n], k) for n in ks}); scr.append(X.float_loss(Ws, WINS, d, L))
            rec[place] = dict(out_cost=lo - fin, out_claimed_cost=WH[place]["out"] - WH["finished"], out_diff=lo - WH[place]["out"],
                              scr_cost=[s - fin for s in scr], claimed_scr_cost=[WH[place][f"scr{k}"] - WH["finished"] for k in (1, 2, 3)],
                              own_scr_at_least_out=sum(s >= lo for s in scr))
        out[nm] = rec
        print(f"{nm}: " + "  ".join(f"{p} out +{r['out_cost']:.3f} (claimed +{r['out_claimed_cost']:.3f}, diff {r['out_diff']:.0e}) own scr "
                                    + "/".join(f"{v:+.3f}" for v in r["scr_cost"]) + f" [>= out {r['own_scr_at_least_out']}/3; claimed scr "
                                    + "/".join(f"{v:+.3f}" for v in r["claimed_scr_cost"]) + "]" for p, r in rec.items() if p != "finished")
              + f"  ({time.time() - t0:.0f}s)", flush=True)
        json.dump(out, open(os.path.join(X.HERE, "evals", "xcheck_where.json"), "w"), indent=1)
print("done")
