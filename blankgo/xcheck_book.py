"""Checker 1 of 3 (independent checking session): the bookkeeping W = shaved blank + pushes, by the checker's own code.
    python3 xcheck_book.py

For w64 L2 and w128 L4, recipes A, B and C, at steps 10, 300 and 3000:
  - the blank redrawn from the seed by xcheck_lib's own part list equals blank_<shape>.npz, every part, every digit
  - the shaved blank B_t replayed by xcheck_lib's own step sizes (C: 0.1 x step size, drawn parts only; A, B: none)
  - W = B_t + P (P from the snapshot). Its fingerprint must equal the hash of the RUNNING weight W that build.py logged
    at that step (build.py held W as its own running total, changed only by its own adds), and B_t's must equal hash_Bt.
    C: R = P + S_P must equal hash_R (every push added, never shaved), S_P equal hash_SP. A, B: R = P, S_P = 0.
  - blank left, sum|B_t| / sum|B_0| over the drawn parts
  - the float64 held-out loss of W by the checker's own float model on its own 128 windows, against the build log
  - machine files: at qkv.0, the last mproj, head and wte, .w = rint(W / 2^(e+30)) and .s = rint(B_t / 2^(e+30)) with
    one shared power per neuron, the smallest that holds both in 16 places (|q| <= 32767)

PREDICTIONS (the checker's, written before this script ran; they are the build session's claims):
  B1. every fingerprint matches (W, B_t, R, S_P) at all 18 (shape, recipe, step) points; the redrawn blank is identical
  B2. blank left: A and B exactly 1; C 0.999817 (10), 0.972013 (300), 0.848012 (3000), the same in both shapes to 6 digits
  B3. float loss within 1e-9 of the build log; at 3000: w64 L2 A 5.647, B 4.756, C 4.757; w128 L4 A 6.205, B 4.584, C 4.586
  B4. every machine-file check holds; the power is minimal for every neuron
"""
import os, json, time, numpy as np
import xcheck_lib as X

res = {}
held = X.held_part(); wins = X.loss_windows(held)
for d, L in ((64, 2), (128, 4)):
    sh = f"w{d}_L{L}"; names = [n for n, _, _ in X.part_list(d, L)]; drawn = X.drawn_names(d, L)
    B0 = X.redraw_blank(d, L); disk = np.load(os.path.join(X.HERE, f"blank_{sh}.npz"))
    same_blank = all(np.array_equal(B0[n], disk[n]) for n in names) and set(disk.keys()) == set(names)
    print(f"{sh}: redrawn blank identical to the file: {same_blank}", flush=True)
    abs0 = sum(int(np.abs(B0[n]).sum()) for n in drawn)
    for rc in "ABC":
        log = json.load(open(os.path.join(X.HERE, f"build_{sh}_{rc}.json")))
        peak, wd = {"A": 0.5, "B": 1e-3, "C": 1e-3}[rc], (0.1 if rc == "C" else 0.0)
        assert log["peak"] == peak and log["wd"] == wd
        Bt = {n: a.copy() for n, a in B0.items()}; done = 0
        for t in (10, 300, 3000):
            t0 = time.time()
            if wd:
                for s in range(done + 1, t + 1):
                    f = X.step_lr(s, peak) * wd
                    for n in drawn: Bt[n] = Bt[n] - np.rint(Bt[n] * f).astype(np.int64)
            done = t
            z = np.load(os.path.join(X.HERE, "snaps", f"{sh}_{rc}_step{t}.npz"))
            P = {n: z["P." + n] for n in names}
            SP = {n: (z["SP." + n] if rc == "C" else np.zeros_like(P[n])) for n in names}
            W = {n: Bt[n] + P[n] for n in names}
            R = {n: P[n] + SP[n] for n in names}
            rec = log["snaps"][str(t)]
            r = dict(W=X.fingerprint(W, names) == rec["hash_W"], Bt=X.fingerprint(Bt, names) == rec["hash_Bt"],
                     R=X.fingerprint(R, names) == rec["hash_R"], SP=X.fingerprint(SP, names) == rec["hash_SP"],
                     undrawn_unshaved=all(np.array_equal(Bt[n], B0[n]) for n in names if n not in drawn),
                     blank_left=sum(int(np.abs(Bt[n]).sum()) for n in drawn) / abs0, blank_left_log=rec["blank_left"])
            if t == 3000:
                r["float_loss"] = X.float_loss(W, wins, d, L); r["float_loss_log"] = rec["held_loss"]
                M = np.load(os.path.join(X.HERE, "machines", f"{sh}_{rc}_step{t}.npz")); mf = {}
                for key, part, ax in (("qkv.0", "qkv.0.w", 0), (f"mproj.{L - 1}", f"mproj.{L - 1}.w", 0), ("head", "head", 0), ("wte", "wte", 1)):
                    e = M[key + ".e"].astype(np.int64); shift = e + 30
                    shp = np.expand_dims(shift, ax)
                    qw = np.rint(W[part] / 2.0 ** shp).astype(np.int64); qs = np.rint(Bt[part] / 2.0 ** shp).astype(np.int64)
                    mx = np.maximum(np.abs(W[part]).max(ax), np.abs(Bt[part]).max(ax)).astype(np.float64)
                    minimal = bool(np.all(mx / 2.0 ** shift <= 32767) and np.all((mx / 2.0 ** (shift - 1) > 32767)))
                    mf[key] = dict(w=bool(np.array_equal(qw, M[key + ".w"].astype(np.int64))), s=bool(np.array_equal(qs, M[key + ".s"].astype(np.int64))),
                                   power_minimal=minimal)
                r["machine_file"] = mf
            r["seconds"] = time.time() - t0
            res[f"{sh}_{rc}_{t}"] = r
            print(f"  {sh} {rc} step {t:4d}: W {r['W']} Bt {r['Bt']} R {r['R']} SP {r['SP']} undrawn {r['undrawn_unshaved']}  blank left {r['blank_left']:.6f} (log {r['blank_left_log']:.6f})"
                  + (f"  float loss {r['float_loss']:.6f} (log {r['float_loss_log']:.6f}, diff {r['float_loss'] - r['float_loss_log']:.2e})  machine file {r['machine_file']}" if t == 3000 else ""), flush=True)
json.dump(res, open(os.path.join(X.HERE, "evals", "xcheck_book.json"), "w"), indent=1)
print("done")
