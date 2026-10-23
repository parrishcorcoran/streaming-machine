"""Checker extra (independent checking session): recipe A's one-word stage, by the checker's own float64 model.
    python3 xcheck_astage.py

The build session's finding: "From about step 10 to 1000, the model gives the same word at all 480 positions. In that
stage, taking the blank out costs -0.09 to +0.10 (sometimes it helps)." eval.py's own numbers, read before this script
was written, show costs of +0.148 (w64 L2 A step 10), +0.146 (w64 L4 A step 10), +0.263 (w128 L4 A step 10) and +0.152
(w128 L4 A step 30) at snapshots whose pairing control is 480/480. This script recomputes, for every A build at steps
10, 30, 100, 300, 1000 and 2000: the float held-out loss (128 windows) of finished and of the pushes alone, and how many
of the 480 positions of the 30 texts carry the single most common finished word (480 = one word everywhere).

PREDICTIONS (the checker's, written before this script ran): the losses equal eval.py's within 1e-9, so the costs above
stand; the one-word count is 480 at w64 L2 A steps 10-1000, w64 L4 A 10-1000, w128 L4 A 10-2000, w128 L2 A 100-300;
hence the stage's cost range is about -0.09 to +0.26, not -0.09 to +0.10.
"""
import os, json, numpy as np, torch
import xcheck_lib as X

held = X.held_part(); WINS = X.loss_windows(held); out = {}
for d, L in ((64, 2), (64, 4), (128, 2), (128, 4)):
    sh = f"w{d}_L{L}"; nm = f"{sh}_A"; names = [n for n, _, _ in X.part_list(d, L)]; drawn = X.drawn_names(d, L); B0 = X.redraw_blank(d, L)
    log = json.load(open(os.path.join(X.HERE, f"build_{nm}.json"))); EV = json.load(open(os.path.join(X.HERE, "evals", f"eval_{nm}.json")))
    tx = torch.from_numpy(np.array(EV["texts"]))
    for t in (10, 30, 100, 300, 1000, 2000):
        z = np.load(os.path.join(X.HERE, "snaps", f"{nm}_step{t}.npz"))
        P = {n: z["P." + n] for n in names}; W = {n: B0[n] + P[n] for n in names}
        assert X.fingerprint(W, names) == log["snaps"][str(t)]["hash_W"]
        Wp = {n: (P[n] if n in drawn else W[n]) for n in names}
        lf, lp = X.float_loss(W, WINS, d, L), X.float_loss(Wp, WINS, d, L)
        with torch.no_grad():
            f1 = X.float_forward({n: torch.from_numpy(a.astype(np.float64) / 2.0 ** 30) for n, a in W.items()}, tx, d, L)[:, 8:, :].argmax(-1).numpy()
        one = int(np.bincount(f1.ravel()).max())
        ev = EV["snaps"][str(t)]
        out[f"{nm}_{t}"] = dict(finished=lf, pushes=lp, cost=lp - lf, one_word=one,
                                diff_finished=lf - ev["finished"]["held_loss"], diff_pushes=lp - ev["pushes"]["held_loss"])
        print(f"{nm} step {t:4d}: cost {lp - lf:+.3f}  (finished {lf:.4f} diff {lf - ev['finished']['held_loss']:.0e}, pushes {lp:.4f} diff {lp - ev['pushes']['held_loss']:.0e})  most common finished word at {one}/480", flush=True)
json.dump(out, open(os.path.join(X.HERE, "evals", "xcheck_astage.json"), "w"), indent=1)
print("done")
