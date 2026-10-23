"""Checker 3 of 3 (independent checking session): case (i) the pushes alone (the blank taken out) and case (iii) the
finished machine, rerun by the checker's own whole-number machine and its own float64 model.
    python3 xcheck_remove.py

All 12 builds, steps 300 and 3000.
  MACHINE (xcheck_lib.WholeMachine; (i) = .w - .s on every drawn part, gains/offsets/biases finished; (iii) = .w):
    - the first 10 of the 30 held-out texts (8 words lead, then 16 read): the top word at each of the 160 positions and
      16 greedy words (each machine's own word fed back), against remove.py's lists for those texts, word for word
    - (i)'s top word as (iii)'s on those 160 positions, against the same count from remove.py's lists
    - held-out loss (log-sum-exp of the piles / 2^16) on the first 10 loss windows, and on the first 64 (remove.py's)
      against remove.py's loss
  FLOAT (xcheck_lib.float_forward, step 3000 only): held-out loss on the 128 windows of finished, pushes, and the blank
    scrambled within each neuron by the checker's own 3 draws (P + scr(B_t) on the drawn parts); top word of pushes as
    finished at the 30 texts' 480 positions. Against eval.py.

PREDICTIONS (the checker's, written before this script ran; they are the build session's numbers):
  R1. 160/160 top words and 160/160 greedy words equal to remove.py's, (i) and (iii), every build and step; (i)-as-(iii)
      counts equal.
  R2. 64-window machine losses within 1e-9 of remove.py's; at 3000 the cost of (i) over (iii): w64 L2 A +0.523, B +0.731,
      C +0.635 (machine, 64 windows).
  R3. float losses within 1e-9 of eval.py's finished and pushes; blank-out cost at 3000 (float, 128 windows):
      w64 L2 A +0.509 B +0.731 C +0.635; w64 L4 A +0.280 B +0.513 C +0.447; w128 L2 A +0.964 B +0.934 C +0.789;
      w128 L4 A +2.988 B +0.974 C +0.733. Pushes-as-finished words (float, /480): w64 L2 255 281 294; w64 L4 226 278 299;
      w128 L2 274 255 284; w128 L4 16 220 246.
  R4. with the checker's own scrambles, the scrambled blank costs more than taking it out in 11 of 12 builds (not
      w64 L2 A), each own draw inside or near the claimed range (+-0.16).
"""
import os, json, time, numpy as np
import xcheck_lib as X

np.seterr(all="ignore")
held = X.held_part(); WINS = X.loss_windows(held)
LEAD, NREAD, NT = 8, 16, 10
out = {}

def mloss(mc, wins):
    tot = 0.0
    for w in wins:
        _, p = mc.run(w[:-1], 128); z = p.astype(np.float64) / 2.0 ** 16
        mx = z.max(1); tot += float((mx + np.log(np.exp(z - mx[:, None]).sum(1)) - z[np.arange(128), w[1:]]).sum())
    return tot / (len(wins) * 128)

for d, L in ((64, 2), (64, 4), (128, 2), (128, 4)):
    sh = f"w{d}_L{L}"; names = [n for n, _, _ in X.part_list(d, L)]; drawn = X.drawn_names(d, L); B0 = X.redraw_blank(d, L)
    for rc in "ABC":
        nm = f"{sh}_{rc}"
        RM = json.load(open(os.path.join(X.HERE, "evals", f"remove_{nm}.json")))
        EV = json.load(open(os.path.join(X.HERE, "evals", f"eval_{nm}.json")))
        log = json.load(open(os.path.join(X.HERE, f"build_{nm}.json")))
        TEXTS = np.array(EV["texts"])[:NT]
        # the texts are the held-out part's own words (checked against the checker's own reading of their starts)
        st = np.linspace(1000, len(held) - 2000, 30).astype(np.int64) + 37
        assert np.array_equal(np.array(EV["texts"]), np.stack([held[s:s + 24] for s in st]))
        for t in (300, 3000):
            t0 = time.time(); rec = {}
            mpath = os.path.join(X.HERE, "machines", f"{nm}_step{t}.npz"); rm = RM["snaps"][str(t)]
            tops = {}
            for var in ("finished", "pushes"):
                mc = X.WholeMachine(mpath, var)
                top = np.array([mc.run(ids, NREAD)[0] for ids in TEXTS])
                gr = []
                for ids in TEXTS:
                    seq = list(ids[:LEAD])
                    for _ in range(NREAD): seq.append(int(mc.run(np.array(seq), 1)[0][0]))
                    gr.append(seq[LEAD:])
                gr = np.array(gr); tops[var] = (top, gr)
                ct, cg = np.array(rm[var]["top1"])[:NT], np.array(rm[var]["greedy"])[:NT]
                r = dict(top1_same=int((top == ct).sum()), greedy_same=int((gr == cg).sum()),
                         loss10=mloss(mc, WINS[:10]))
                if t == 3000:
                    r["loss64"] = mloss(mc, WINS[:64]); r["loss64_claimed"] = rm[var]["loss"]
                rec[var] = r
            ft, fg = tops["finished"]; pt, pg = tops["pushes"]
            rec["pushes_top1_as_finished_10"] = int((pt == ft).sum())
            rec["claimed_pushes_top1_as_finished_10"] = int((np.array(rm["pushes"]["top1"])[:NT] == np.array(rm["finished"]["top1"])[:NT]).sum())
            rec["pushes_greedy_as_finished_10"] = int((pg == fg).sum())
            rec["claimed_pushes_greedy_as_finished_10"] = int((np.array(rm["pushes"]["greedy"])[:NT] == np.array(rm["finished"]["greedy"])[:NT]).sum())
            if t == 3000:
                Bt = X.shaved_blank(B0, drawn, log["peak"], log["wd"], t)
                z = np.load(os.path.join(X.HERE, "snaps", f"{nm}_step{t}.npz"))
                P = {n: z["P." + n] for n in names}; W = {n: Bt[n] + P[n] for n in names}
                assert X.fingerprint(W, names) == log["snaps"][str(t)]["hash_W"]
                Wp = {n: (P[n] if n in drawn else W[n]) for n in names}
                fl = dict(finished=X.float_loss(W, WINS, d, L), pushes=X.float_loss(Wp, WINS, d, L))
                fl["scr"] = [X.float_loss({n: (P[n] + X.scr_within_neuron(n, Bt[n], k) if n in drawn else W[n]) for n in names}, WINS, d, L) for k in (1, 2, 3)]
                ev = EV["snaps"]["3000"]
                fl["finished_claimed"], fl["pushes_claimed"] = ev["finished"]["held_loss"], ev["pushes"]["held_loss"]
                fl["scr_claimed"] = [ev[f"scr{k}"]["held_loss"] for k in (1, 2, 3)]
                import torch
                with torch.no_grad():
                    Wf = {n: torch.from_numpy(a.astype(np.float64) / 2.0 ** 30) for n, a in W.items()}
                    Wpf = {n: torch.from_numpy(a.astype(np.float64) / 2.0 ** 30) for n, a in Wp.items()}
                    tx = torch.from_numpy(np.array(EV["texts"]))
                    f1 = X.float_forward(Wf, tx, d, L)[:, 8:, :].argmax(-1).numpy(); p1 = X.float_forward(Wpf, tx, d, L)[:, 8:, :].argmax(-1).numpy()
                fl["pushes_words_as_finished_480"] = int((f1 == p1).sum()); fl["claimed_words"] = ev["pushes"]["agree_finished_top1"]
                rec["float"] = fl
            rec["seconds"] = time.time() - t0
            out[f"{nm}_{t}"] = rec
            line = (f"{nm} {t:4d}: (iii) top1 {rec['finished']['top1_same']}/160 greedy {rec['finished']['greedy_same']}/160 | (i) top1 {rec['pushes']['top1_same']}/160 greedy {rec['pushes']['greedy_same']}/160"
                    f" | (i) as (iii): top1 {rec['pushes_top1_as_finished_10']} (claimed {rec['claimed_pushes_top1_as_finished_10']}) greedy {rec['pushes_greedy_as_finished_10']} (claimed {rec['claimed_pushes_greedy_as_finished_10']})"
                    f" | loss10 (iii) {rec['finished']['loss10']:.4f} (i) {rec['pushes']['loss10']:.4f} (+{rec['pushes']['loss10'] - rec['finished']['loss10']:.3f})")
            if t == 3000:
                f = rec["float"]
                line += (f"\n      machine 64 windows: (iii) {rec['finished']['loss64']:.6f} (claimed {rec['finished']['loss64_claimed']:.6f})  (i) {rec['pushes']['loss64']:.6f} (claimed {rec['pushes']['loss64_claimed']:.6f})  cost +{rec['pushes']['loss64'] - rec['finished']['loss64']:.3f}"
                         f"\n      float 128 windows: finished {f['finished']:.6f} (claimed {f['finished_claimed']:.6f})  pushes {f['pushes']:.6f} (claimed {f['pushes_claimed']:.6f})  blank-out cost +{f['pushes'] - f['finished']:.3f}"
                         f"  own scrambles cost " + " ".join(f"+{v - f['finished']:.3f}" for v in f["scr"]) + "  (claimed " + " ".join(f"+{v - f['finished_claimed']:.3f}" for v in f["scr_claimed"]) + ")"
                         f"  scr > out: {sum(v > f['pushes'] for v in f['scr'])}/3  words {f['pushes_words_as_finished_480']} (claimed {f['claimed_words']})")
            print(line + f"  [{rec['seconds']:.0f}s]", flush=True)
            json.dump(out, open(os.path.join(X.HERE, "evals", "xcheck_remove.json"), "w"), indent=1)
print("done")
