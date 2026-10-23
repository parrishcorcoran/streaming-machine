"""Checker 2 of 3 (independent checking session): s, k, gamma and the light ratio, by the checker's own whole-number
machine (xcheck_lib.WholeMachine) and its own scrambles.
    python3 xcheck_work.py

Builds w64 L2 C (shave on) and w128 L4 B (shave off), steps 300 and 3000. Plates fc.0, mproj.0, the last fc, head.
  0. IDENTITY: the checker's machine, finished and pushes (.w - .s), at the 30 word texts: every head pile the same
     whole number as machine.py's (sha256 of the int64 piles, words/words_<build>.json).
  1. IN PLACE (exact whole-number dots): s = (W . W0)/(W0 . W0), W = B_t + P; the push part (P . W0)/(W0 . W0);
     control (P . scr(W0))/(W0 . W0) with the checker's own 3 scrambles within each neuron. Against inplace.py's s.
  2. ON THE LIGHT (the checker's machine, finished plates, the 128 held-out loss windows, every position):
     b = x16 . s, d = x16 . (w - s), on their real size (x 2^(position power + neuron power));
     k = -2<b,d>/(|b|^2 + |d|^2), gamma = 1 + <b,d>/<b,b>; control: s scrambled within each neuron (own 3 draws).
     Against work.py's k and gamma.
  3. LIGHT RATIO: sqrt( sum (x . s)^2 / sum E[(x_order . s)^2] ), the chance level over every order of the light's lanes
     worked out by the checker's own derivation:
        E = sum x^2 sum s^2 / n + ((sum x)^2 - sum x^2)((sum s)^2 - sum s^2) / (n (n - 1))
     and, as a check on that derivation, the average over 20 drawn lane orders. Against dodge_exact.py.
  4. the machine's held-out loss on the 128 windows (log-sum-exp of the piles / 2^16), against work.py's machine_loss.

PREDICTIONS (the checker's, written before this script ran; they are the build session's numbers):
  X0. 30/30 identical piles, finished and pushes, all 4 (build, step) points.
  X1. s within 1e-12 of inplace.py: w64 L2 C 3000: fc.0 1.316369, mproj.0 1.369685, fc.1 0.737121, head 0.688964;
      w128 L4 B 3000: fc.0 1.424729, mproj.0 1.774294, fc.3 0.929535, head 0.814089. Own controls |push part| < 0.07.
  X2. k, gamma within 1e-9 of work.py: w64 L2 C 3000: fc.0 k -0.20470 gamma 1.92021; mproj.0 -0.14320 / 3.53434;
      fc.1 +0.20198 / 0.56130; head +0.00841 / 0.42630. w128 L4 B 3000: fc.0 -0.27167 / 2.07880; mproj.0 -0.08847 /
      6.52965; fc.3 +0.14306 / 0.67457; head +0.01658 / 0.41987. Own controls: |k'| < 0.06, and gamma outside the
      own control range wherever work.py's summary marks it A or B.
  X3. light ratio within 1e-9 of dodge_exact.py: w64 L2 C 3000: fc.0 1.04007, mproj.0 1.19592, fc.1 0.99011, head
      1.00142; w128 L4 B 3000: fc.0 1.04284, mproj.0 1.20378, fc.3 0.99745, head 1.00187. The 20-order average within
      the scatter of 20 draws (about 0.03).
  X4. machine loss within 1e-9: w64 L2 C 3000 4.757175, w128 L4 B 3000 4.584310.
"""
import os, json, time, hashlib, math, numpy as np
import xcheck_lib as X

np.seterr(all="ignore")
held = X.held_part(); WINS = X.loss_windows(held)
out = {}
for d, L, rc in ((64, 2, "C"), (128, 4, "B")):
    sh = f"w{d}_L{L}"; nm = f"{sh}_{rc}"; names = [n for n, _, _ in X.part_list(d, L)]; drawn = X.drawn_names(d, L)
    log = json.load(open(os.path.join(X.HERE, f"build_{nm}.json")))
    EV = json.load(open(os.path.join(X.HERE, "evals", f"eval_{nm}.json"))); TEXTS = np.array(EV["texts"])
    WJ = json.load(open(os.path.join(X.HERE, "words", f"words_{nm}.json")))
    WK = json.load(open(os.path.join(X.HERE, "evals", f"work_{nm}.json")))
    IP = json.load(open(os.path.join(X.HERE, "evals", f"inplace_{nm}.json")))
    DX = json.load(open(os.path.join(X.HERE, "evals", f"dodge_exact_{nm}.json")))
    B0 = X.redraw_blank(d, L)
    PLATES = ["fc.0", "mproj.0", f"fc.{L - 1}", "head"]
    for t in (300, 3000):
        t0 = time.time(); key = f"{nm}_{t}"; rec = {}
        mpath = os.path.join(X.HERE, "machines", f"{nm}_step{t}.npz")
        # 0. identity
        ident = {}
        for var in ("finished", "pushes"):
            mc = X.WholeMachine(mpath, var)
            shas = [hashlib.sha256(mc.run(ids, 16)[1].tobytes()).hexdigest() for ids in TEXTS]
            ident[var] = sum(a == b for a, b in zip(shas, WJ["snaps"][str(t)][var]["piles_sha256"]))
        rec["identity"] = ident
        # 1. in place
        Bt = X.shaved_blank(B0, drawn, log["peak"], log["wd"], t)
        z = np.load(os.path.join(X.HERE, "snaps", f"{nm}_step{t}.npz"))
        assert X.fingerprint({n: Bt[n] + z["P." + n] for n in names}, names) == log["snaps"][str(t)]["hash_W"]
        ipl = {}
        for pl in PLATES + ["wte", "wpe"]:
            part = pl + ".w" if pl not in ("head", "wte", "wpe") else pl
            P = z["P." + part]; W = Bt[part] + P; w0 = B0[part]
            w0w0 = X.exact_dot(w0, w0)
            s = X.exact_dot(W, w0) / w0w0; push = X.exact_dot(P, w0) / w0w0
            ctrl = [X.exact_dot(P, X.scr_within_neuron(part, w0, k)) / w0w0 for k in (1, 2, 3)]
            ipl[pl] = dict(s=s, s_claimed=IP["snaps"][str(t)]["in_place"][part]["s"], push=push, push_claimed=IP["snaps"][str(t)]["in_place"][part]["push_part"],
                           control=ctrl, claimed_control=IP["snaps"][str(t)]["in_place"][part]["control"],
                           push_beyond_all_controls=bool(abs(push) > max(abs(c) for c in ctrl)))
        rec["in_place"] = ipl
        # 2-4. on the light
        M = np.load(mpath)
        mc = X.WholeMachine(mpath, "finished", capture=True)
        Q = {}
        for pl in PLATES:
            w = M[pl + ".w"].astype(np.int64); s = M[pl + ".s"].astype(np.int64); n = w.shape[0]
            scr = [X.scr_within_neuron(pl, s, k).astype(np.float64) for k in (1, 2, 3)]
            orders = [np.random.default_rng(424 + 13 * j + n).permutation(n) for j in range(20)]
            Q[pl] = dict(s=s.astype(np.float64), p=(w - s).astype(np.float64), ce=2.0 ** M[pl + ".e"].astype(np.float64),
                         scr=scr, orders=orders, ss=(s.astype(np.float64) ** 2).sum(0), s1=s.astype(np.float64).sum(0), n=n)
        acc = {pl: dict(bb=0.0, dd=0.0, bd=0.0, E=0.0, sbb=[0.0] * 3, sbd=[0.0] * 3, mc=[0.0] * 20) for pl in PLATES}
        lsum, lcnt = 0.0, 0
        for win in WINS:
            _, piles = mc.run(win[:-1], 128)
            zf = piles.astype(np.float64) / 2.0 ** 16
            mx = zf.max(1); lse = mx + np.log(np.exp(zf - mx[:, None]).sum(1))
            lsum += float((lse - zf[np.arange(128), win[1:]]).sum()); lcnt += 128
            for pl in PLATES:
                q = Q[pl]; a = acc[pl]
                x16, ex = mc.cap[pl]; x = x16.astype(np.float64); rs = 2.0 ** ex.astype(np.float64)
                scale = rs[:, None] * q["ce"][None, :]
                bi = x @ q["s"]; di = x @ q["p"]
                assert np.array_equal(bi, np.round(bi)) and np.abs(bi).max() < 2 ** 53
                b = bi * scale; dd = di * scale
                a["bb"] += float((b * b).sum()); a["dd"] += float((dd * dd).sum()); a["bd"] += float((b * dd).sum())
                for k in range(3):
                    bs = (x @ q["scr"][k]) * scale
                    a["sbb"][k] += float((bs * bs).sum()); a["sbd"][k] += float((bs * dd).sum())
                n = q["n"]; sx2 = (x * x).sum(1); sx = x.sum(1)
                E = np.outer(sx2, q["ss"]) / n + np.outer(sx * sx - sx2, q["s1"] ** 2 - q["ss"]) / (n * (n - 1))
                a["E"] += float((E * scale * scale).sum())
                for j, o in enumerate(q["orders"]):
                    bo = (x[:, o] @ q["s"]) * scale; a["mc"][j] += float((bo * bo).sum())
        lt = {}
        for pl in PLATES:
            a = acc[pl]; wk = WK["snaps"][str(t)]["plates"][pl]
            k = -2 * a["bd"] / (a["bb"] + a["dd"]); g = 1 + a["bd"] / a["bb"]
            kc = [-2 * a["sbd"][i] / (a["sbb"][i] + a["dd"]) for i in range(3)]; gc = [1 + a["sbd"][i] / a["sbb"][i] for i in range(3)]
            ratio = math.sqrt(a["bb"] / a["E"]); mcr = math.sqrt(a["bb"] / np.mean(a["mc"]))
            spread = [math.sqrt(v / np.mean(a["mc"])) for v in a["mc"]]
            lt[pl] = dict(k=k, k_claimed=wk["k"], gamma=g, gamma_claimed=wk["gamma"], k_control=kc, gamma_control=gc,
                          k_control_claimed=wk["k_control"], gamma_control_claimed=wk["gamma_control"],
                          gamma_vs_own_control=("ABOVE" if g > max(gc) else "BELOW" if g < min(gc) else "IN"),
                          gamma_vs_claimed_control=("ABOVE" if g > max(wk["gamma_control"]) else "BELOW" if g < min(wk["gamma_control"]) else "IN"),
                          light_ratio=ratio, light_ratio_claimed=DX["snaps"][str(t)][pl]["blank"], light_ratio_20_orders=mcr,
                          order_draw_spread=[min(spread), max(spread)], drawn_ratio_work_py=wk["dodge_ratio"])
        rec["light"] = lt
        rec["machine_loss"] = lsum / lcnt; rec["machine_loss_claimed"] = WK["snaps"][str(t)]["machine_loss"]
        rec["seconds"] = time.time() - t0
        out[key] = rec
        print(f"{key}: identity finished {ident['finished']}/30 pushes {ident['pushes']}/30   machine loss {rec['machine_loss']:.6f} (claimed {rec['machine_loss_claimed']:.6f})  {rec['seconds']:.0f}s", flush=True)
        for pl, o in ipl.items():
            print(f"   in place {pl:8s} s {o['s']:.6f} (claimed {o['s_claimed']:.6f}, diff {o['s'] - o['s_claimed']:.1e})  push {o['push']:+.5f}  own controls {min(o['control']):+.5f}..{max(o['control']):+.5f}  beyond all {o['push_beyond_all_controls']}", flush=True)
        for pl, o in lt.items():
            print(f"   light {pl:8s} k {o['k']:+.5f} (claimed {o['k_claimed']:+.5f}, diff {o['k'] - o['k_claimed']:.1e})  gamma {o['gamma']:.5f} (claimed {o['gamma_claimed']:.5f})  "
                  f"own ctrl k {min(o['k_control']):+.4f}..{max(o['k_control']):+.4f} gamma {min(o['gamma_control']):.3f}..{max(o['gamma_control']):.3f} -> {o['gamma_vs_own_control']} (claimed ctrl -> {o['gamma_vs_claimed_control']})  "
                  f"| ratio {o['light_ratio']:.5f} (claimed {o['light_ratio_claimed']:.5f}, diff {o['light_ratio'] - o['light_ratio_claimed']:.1e})  20 orders {o['light_ratio_20_orders']:.5f} (draws {o['order_draw_spread'][0]:.3f}..{o['order_draw_spread'][1]:.3f})", flush=True)
        json.dump(out, open(os.path.join(X.HERE, "evals", "xcheck_work.json"), "w"), indent=1)
print("done")
