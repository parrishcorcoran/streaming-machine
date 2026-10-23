"""Where the blank goes, in the WORK: every pile of the whole-number machine split into the blank's part and the built
part, on the machine's own light, on held-out text.
    python3 work.py width=64 layers=2 recipe=A [windows=128]

HIS WORDS (2026-10-06): "If you run the blank model ALL that compute is the opposite sides canceling that we are trying
to remove." "It is zero sum because it must be. It doens have to be in the same spot."

SESSION'S RESTATEMENT (the session's words, not his): in the blank machine every pile is meetings whose ups and downs
cancel. The building only adds pushes; nothing takes the blank out (shave off). So the blank's part of every pile is
still there in the finished machine, and if the machine works, that part is either cancelled by an opposite part in
what was built (same pile, opposite sign), or kept small by the light (the light steering around the blank), or it is
carried along.

THE SPLIT (exact, digit for digit). The machine file keeps the blank (.s) on the finished plate's power of two, so at
every plate the light x16 (16 places, its power per position, captured by bmachine.py exactly as the product uses it)
meets   w = s + (w - s):   pile t = b + d,   b = x16 . s (the blank's part),   d = x16 . (w - s) (the built part),
whole numbers that add exactly. For the sums below every pile is put on its real size (x 2^(position power + neuron
power)); the sums are readings, not part of the machine. Plates: qkv, proj (out), fc (up), mproj (down) of every layer,
and the output word table (head, read at all 128 positions). Held-out text: the first `windows` of the 128 fixed
held-out windows of model.held_windows, 128 words each (the loss windows of every earlier script).

Readings per plate, every snapshot:
  1. CANCELLATION IN THE MEETINGS (his "opposite sides canceling"): share = 1 - sum|pile| / sum over piles of the sum of
     |meeting| (|x_i| |w_ij|), for the finished piles t (meetings x w), the blank's part b (x s) and the built part d
     (x (w - s)). Coin toss: the same meetings with random signs (each light value's sign flipped by a fair coin, 2
     draws), and the formula 1 - sqrt(2/pi) |meetings|_2 / |meetings|_1. Also the blank's share of the meeting size,
     sum|x||s| / (sum|x||s| + sum|x||w - s|).
  2. A COUNTER IN WHAT WAS BUILT:   k = -2<b,d> / (|b|^2 + |d|^2)  (k > 0: the built part cancels the blank's part of
     the pile on the real light; 1: exactly opposite and equal)  and  gamma = <t,b>/<b,b> = 1 + <d,b>/<b,b> (the share
     of the blank's pile still showing in the finished pile: 1 carried as it is, 0 cancelled, > 1 built along it).
     Also with the plate's bias counted in the built part. Control: the plate's blank scrambled within each neuron
     (snapio.scramble, draws 1-3; the same values meeting the wrong lines): k', gamma' with b' = x16 . scr(s).
  3. DODGED BY THE LIGHT: ratio = |x16 . s| / |x16[lanes scrambled] . s|, the light's lanes put in a fixed random order
     (5 draws, the same order at every position; same values). Under 1: the light steers around the blank.
  Order-free check, every snapshot: on the first window, b and d of every plate rebuilt with the lanes in a random
  order and int64 sums: the same whole numbers as the machine's float64 sums.
  The machine's own held-out loss (from its whole-number head piles: log-sum-exp in float64 as a reading) is kept for
  remove.py's comparison.

PREDICTIONS (the session's, written before any run of this script; inplace.py had already run: in place, s is about 1
in A, and in B and C below 1 in the tables (wpe 0.23-0.53, head 0.69-0.81) but ABOVE 1 in some first-layer plates
(fc.0, mproj.0 up to 1.4-1.8: pushes along the blank)):
  W1. Blank machine (step 0): cancelled share of the meetings equals the coin toss within +-0.01 in every plate: about
      0.84 (64 lines in), 0.89 (128), 0.92 (256), 0.945 (512) for bell-shaped light; lower for the light out of the bend
      (into mproj; lopsided). (Fixed by the blank's random signs: this reads the SIZE of the cancellation, not whether
      it happens.)
  W2. Finished machines, from step 100 on: the finished piles cancel LESS than the coin toss, by 0.02-0.15 in every
      plate; the blank's part b cancels as the coin toss (+-0.01).
  W3. gamma follows the in-place s where s departs from 1 (above 1 where s is above 1), and is LOWER than s by 0.05-0.3
      in most plates at step 3000 (the pushes take the blank out more along the light's strong directions than over all
      directions); so k > 0 (0.03-0.5) in most plates of every recipe at step 3000 and k < 0 at fc.0/mproj.0 of B and C.
      Controls: |k'| < 0.02, gamma' = 1 +- 0.05.
  W4. Dodge ratio 0.9-1.15 at every plate, inside or near the lane-scrambled draws' spread (the raw word rows met the
      blank fc.0 1.08-1.12 times harder than scrambled rows, so the first layer may lean in, not dodge).
  W5. Order-free check: every b and d identical, every plate, every snapshot.
"""
import os, sys, json, math, time, numpy as np
from model import HERE, V, tag, data, held_windows
from snapio import scramble, build_log
from bmachine import Machine, machine_file, ES

np.seterr(all="ignore")                               # Accelerate's matmul raises spurious divide warnings; every result checked finite
A = dict(a.split("=") for a in sys.argv[1:])
d, L, RECIPE = int(A.get("width", 64)), int(A.get("layers", 2)), A.get("recipe", "A")
NW = int(A.get("windows", 128))
name = f"{tag(d, L)}_{RECIPE}"
_, held = data(); WIN = held_windows(held)[:NW]
STEPS = sorted(int(s) for s in build_log(d, L, RECIPE)["snaps"])
if "steps" in A: STEPS = [int(x) for x in A["steps"].split(",")]
OUTF = os.path.join(HERE, "evals", f"work_{name}.json" if "steps" not in A and NW == 128 else f"work_{name}_trial.json")
PLATES = [f"{k}.{l}" for l in range(L) for k in ("qkv", "proj", "fc", "mproj")] + ["head"]
C2 = math.sqrt(2 / math.pi)

def mm(a, b):
    r = a @ b; assert np.isfinite(r).all(); return r

res = dict(shape=tag(d, L), recipe=RECIPE, windows=NW, positions=NW * 128, snaps={})
print(f"work {name}: {NW} windows x 128 words", flush=True)
for t in STEPS:
    t0 = time.time()
    M = machine_file(d, L, RECIPE, t)
    m = Machine(d, L, RECIPE, t, "finished", capture=True, M=M)
    PD = {}
    for pl in PLATES:
        w, s = M[pl + ".w"].astype(np.float64), M[pl + ".s"].astype(np.float64)
        bias = (M[pl + ".b"].astype(np.float64) / 2.0 ** ES) if pl != "head" else np.zeros(w.shape[1])
        fname = pl + ".w" if pl != "head" else "head"
        nin = w.shape[0]
        PD[pl] = dict(w=w, s=s, p=w - s, aw=np.abs(w), as_=np.abs(s), ap=np.abs(w - s),
                      w2=w * w, s2=s * s, p2=(w - s) ** 2, cs=2.0 ** M[pl + ".e"].astype(np.float64), bias=bias,
                      scr=[scramble(fname, M[pl + ".s"].astype(np.int64), k).astype(np.float64) for k in (1, 2, 3)],
                      perms=[np.random.default_rng(5000 + 17 * j + nin).permutation(nin) for j in range(5)],
                      nin=nin)
    acc = {pl: dict.fromkeys(["bb", "dd", "bd", "tt", "bdb", "dbdb", "abs_t", "abs_b", "abs_d", "m_w", "m_s", "m_p",
                              "l2_w", "l2_s", "l2_p", "coin_t", "coin_b", "coin_d"], 0.0) | dict(
        scr_bb=[0.0] * 3, scr_bd=[0.0] * 3, perm_bb=[0.0] * 5, n=0) for pl in PLATES}
    loss_sum, loss_n, order_free = 0.0, 0, dict(checked=0, identical=0)
    rng = np.random.default_rng(31 + t)
    for wi, win in enumerate(WIN):
        m.cap = {}
        _, lg = m.forward(win[:-1], 128)
        z = lg.astype(np.float64) / 2.0 ** ES
        mx = z.max(1); lse = mx + np.log(np.exp(z - mx[:, None]).sum(1))
        loss_sum += float((lse - z[np.arange(128), win[1:]]).sum()); loss_n += 128
        for pl in PLATES:
            q = PD[pl]; a = acc[pl]
            x16, ex = m.cap[pl][0]
            xf = x16.astype(np.float64); rs = 2.0 ** ex.astype(np.float64)
            sc = rs[:, None] * q["cs"][None, :]
            b = mm(xf, q["s"]) * sc; dd = mm(xf, q["p"]) * sc; tt = b + dd
            a["bb"] += float((b * b).sum()); a["dd"] += float((dd * dd).sum()); a["bd"] += float((b * dd).sum())
            a["tt"] += float((tt * tt).sum())
            db = dd + q["bias"][None, :]
            a["bdb"] += float((b * db).sum()); a["dbdb"] += float((db * db).sum())
            for k in range(3):
                bs = mm(xf, q["scr"][k]) * sc
                a["scr_bb"][k] += float((bs * bs).sum()); a["scr_bd"][k] += float((bs * dd).sum())
            for j in range(5):
                bp = mm(xf[:, q["perms"][j]], q["s"]) * sc
                a["perm_bb"][j] += float((bp * bp).sum())
            ax = np.abs(xf); x2 = xf * xf
            a["abs_t"] += float(np.abs(tt).sum()); a["abs_b"] += float(np.abs(b).sum()); a["abs_d"] += float(np.abs(dd).sum())
            a["m_w"] += float((mm(ax, q["aw"]) * sc).sum()); a["m_s"] += float((mm(ax, q["as_"]) * sc).sum()); a["m_p"] += float((mm(ax, q["ap"]) * sc).sum())
            a["l2_w"] += float((np.sqrt(mm(x2, q["w2"])) * sc).sum()); a["l2_s"] += float((np.sqrt(mm(x2, q["s2"])) * sc).sum())
            a["l2_p"] += float((np.sqrt(mm(x2, q["p2"])) * sc).sum())
            for _ in range(2):
                xe = xf * rng.choice([-1.0, 1.0], size=xf.shape)
                a["coin_t"] += float(np.abs(mm(xe, q["w"]) * sc).sum()) / 2
                a["coin_b"] += float(np.abs(mm(xe, q["s"]) * sc).sum()) / 2
                a["coin_d"] += float(np.abs(mm(xe, q["p"]) * sc).sum()) / 2
            a["n"] += b.size
            if wi == 0:                                   # order-free: lanes in a random order, int64 sums
                pr = rng.permutation(q["nin"])
                bi = x16[:, pr] @ M[pl + ".s"].astype(np.int64)[pr]
                di = x16[:, pr] @ (M[pl + ".w"].astype(np.int64) - M[pl + ".s"].astype(np.int64))[pr]
                same = np.array_equal(bi, mm(xf, q["s"]).astype(np.int64)) and np.array_equal(di, mm(xf, q["p"]).astype(np.int64))
                order_free["checked"] += 1; order_free["identical"] += int(same)
    out = {}
    for pl in PLATES:
        a = acc[pl]; o = {}
        sh = lambda num, den: (1 - num / den) if den > 0 else None
        o["cancel_finished"] = sh(a["abs_t"], a["m_w"]); o["coin_finished"] = sh(a["coin_t"], a["m_w"]); o["formula_finished"] = sh(C2 * a["l2_w"], a["m_w"])
        o["cancel_blank"] = sh(a["abs_b"], a["m_s"]); o["coin_blank"] = sh(a["coin_b"], a["m_s"]); o["formula_blank"] = sh(C2 * a["l2_s"], a["m_s"])
        o["cancel_built"] = sh(a["abs_d"], a["m_p"]); o["coin_built"] = sh(a["coin_d"], a["m_p"]); o["formula_built"] = sh(C2 * a["l2_p"], a["m_p"])
        o["blank_share_of_meeting_size"] = a["m_s"] / (a["m_s"] + a["m_p"]) if (a["m_s"] + a["m_p"]) > 0 else None
        o["rms_b"], o["rms_d"], o["rms_t"] = (math.sqrt(a[k] / a["n"]) for k in ("bb", "dd", "tt"))
        if a["dd"] > 0:
            o["k"] = -2 * a["bd"] / (a["bb"] + a["dd"]); o["gamma"] = 1 + a["bd"] / a["bb"]
            o["k_with_bias"] = -2 * a["bdb"] / (a["bb"] + a["dbdb"]); o["gamma_with_bias"] = 1 + a["bdb"] / a["bb"]
            o["k_control"] = [-2 * a["scr_bd"][k] / (a["scr_bb"][k] + a["dd"]) for k in range(3)]
            o["gamma_control"] = [1 + a["scr_bd"][k] / a["scr_bb"][k] for k in range(3)]
        o["dodge_ratio"] = math.sqrt(a["bb"] / np.mean(a["perm_bb"]))
        o["dodge_ratio_draws"] = [math.sqrt(a["bb"] / v) for v in a["perm_bb"]]
        o["dodge_draw_spread"] = [math.sqrt(v / np.mean(a["perm_bb"])) for v in a["perm_bb"]]
        out[pl] = o
    res["snaps"][t] = dict(plates=out, machine_loss=loss_sum / loss_n, order_free=order_free, seconds=time.time() - t0)
    print(f"  step {t:5d}  machine held-out loss {loss_sum / loss_n:.4f}  order-free {order_free['identical']}/{order_free['checked']}  ({time.time() - t0:.0f}s)", flush=True)
    for pl in PLATES:
        o = out[pl]
        line = (f"     {pl:8s} cancel: finished {o['cancel_finished']:.3f} (coin {o['coin_finished']:.3f})  blank {o['cancel_blank']:.3f} (coin {o['coin_blank']:.3f})"
                + (f"  built {o['cancel_built']:.3f} (coin {o['coin_built']:.3f})" if o["cancel_built"] is not None else "")
                + f"  | blank share of meetings {o['blank_share_of_meeting_size']:.3f}  |b|/|t| {o['rms_b'] / max(o['rms_t'], 1e-300):.3f}")
        if "k" in o:
            line += (f"  | k {o['k']:+.3f} (ctrl {min(o['k_control']):+.3f}..{max(o['k_control']):+.3f})  gamma {o['gamma']:.3f}"
                     f" (ctrl {min(o['gamma_control']):.3f}..{max(o['gamma_control']):.3f})  with bias k {o['k_with_bias']:+.3f} gamma {o['gamma_with_bias']:.3f}")
        line += f"  | dodge {o['dodge_ratio']:.3f} (draws {min(o['dodge_draw_spread']):.3f}..{max(o['dodge_draw_spread']):.3f})"
        print(line, flush=True)
    json.dump(res, open(OUTF, "w"), indent=1)
print("done")
