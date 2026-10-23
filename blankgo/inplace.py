"""Where the blank goes, read from the weights themselves (exact whole numbers at 2^-30): in place, the zero sum of the
write lines, and the word rows meeting the blank first plate.
    python3 inplace.py width=64 layers=2 recipe=A

HIS WORDS (2026-10-06): "It is zero sum because it must be. It doens have to be in the same spot." "because the embedding
table is the opposite as the ENTIRE all neuron rows, a tiny change in embedding is a bigger change on other side, so if
it's not balanced perfectly then the blank is stuck inside the embedding table."

SESSION'S RESTATEMENT (the session's words, not his): nothing in the building takes the blank out (shave off), so if the
blank's work is undone, something opposite to it must stand somewhere: in the same spot (the pushes on a weight running
against that weight's blank) or elsewhere (the word table built so its rows do not excite the blank plates).

Three readings, every snapshot (snapio.snapshots, checked against the build's hashes):
  1. IN PLACE, per plate and table. W0 = the blank as drawn, Bt = as it stands (C: shaved), P = the pushes as they stand.
       s = (W . W0)/(W0 . W0) = (Bt . W0)/(W0 . W0) + (P . W0)/(W0 . W0)        (A, B: the first part is exactly 1)
     s < 1 with the shave off: the pushes ran against the blank in its own spot. Control: (P . scr_k(W0))/(W0 . W0),
     W0 scrambled within each neuron (snapio.scramble, draws 1-3): the same values the pushes never met.
  2. ZERO SUM OF THE WRITE LINES. A write line is one row of values written across the stream's lanes: a row of an out
     plate (proj, mproj), an input word row (wte), a position row (wpe), an out plate's bias. Every reader of the stream
     is a water level, which takes out the lanes' mean, so a plain push (recipe A) on a write line sums to 0 over the
     lanes in exact arithmetic (bookkeeping, true by construction, not a finding); only the rounding of each push to the
     grain is left. Measured: each line's lane sum of P in grains (exact int64); the size a coin-toss sum of the same
     values would have, sqrt(sum of squares); per plate, per table, the whole model. Same for the blank W0 (its lane mean
     is never seen by any reader; bookkeeping). Contrast: the read plates' (qkv, fc, head) column sums of P, which no
     water level constrains.
  3. THE WORD ROWS MEETING THE BLANK FIRST PLATE (raw rows, no water level, no position row: a base he never gave,
     rule 13.2; asked for directly, and the real light is measured in work.py). Rows X in {blank rows Bt_wte, finished
     rows W_wte, pushes P_wte}, each weighted by its word's count in the held-out part, meeting the blank plate Q (as it
     stands) of qkv.0 and of fc.0:
       passage rho_X = sum_w c_w |X_w Q|^2 / (sum_w c_w |X_w|^2 x |Q|^2 / lanes)      (1 = as a random direction)
     control: the same rows with their lanes scrambled (one order per draw, 5 draws, same values).
       k_wte = -2 <Bt_wte Q, P_wte Q> / (|Bt_wte Q|^2 + |P_wte Q|^2)    (count-weighted; > 0: the word table's pushes
     carry an opposite of the blank rows' passage through the blank plate). Control: Bt_wte scrambled within each row.

PREDICTIONS (the session's, written before any run of this script):
  I1. In place, step 3000, shave off: s in 0.5-0.95 in every plate of A and B; wte 0.95-1.0 in A (its pushes are
      0.07-0.37 of the blank by size) and 0.8-1.0 in B; head 0.4-0.95. C: s(C) = s(B) - 0.12 to -0.18 (the shave 0.848
      plus pushes like B's). s falls with steps (from exactly 1 at step 0). Controls within +-0.03 of 0.
  I2. A: every write line's lane sum is rounding only: RMS 100-200 grains (d=64) and 140-260 grains (d=128) on the out
      plates' rows at step 3000 (sqrt(3000 x lanes / 12)), wte rows smaller (pushed only when their word comes up);
      against a coin-toss size about 1e7-1e8 grains, a ratio under 1e-5. B and C: lane sums 0.2-1.0 of the coin-toss
      size (the scaled push breaks the zero sum). The blank's own lane sums: about 1.0 of the coin-toss size. Read plates'
      column sums of P: 0.3-3 of the coin-toss size in every recipe.
  I3. Word rows: rho(blank rows) 0.95-1.05; rho(finished rows) 0.85-1.1 and inside the scrambled rows' range; k_wte
      within +-0.05 of 0 and inside its control's range.
"""
import os, sys, json, numpy as np
from model import HERE, layout, drawn, tag, blank, data
from snapio import snapshots, scramble

A = dict(a.split("=") for a in sys.argv[1:])
d, L, RECIPE = int(A.get("width", 64)), int(A.get("layers", 2)), A.get("recipe", "A")
name = f"{tag(d, L)}_{RECIPE}"
dr = drawn(d, L)
W0 = blank(d, L)
_, held = data()
cnt = np.bincount(held, minlength=8193).astype(np.float64)
np.seterr(all="ignore")                               # Accelerate's matmul raises spurious divide warnings; results checked finite
f64 = lambda a: a.astype(np.float64)
dot = lambda a, b: float(np.dot(f64(a).ravel(), f64(b).ravel()))

def lane_stats(lines):                                # lines: (n, lanes) int64; lane sums exact
    s = lines.sum(1); ref = np.sqrt((f64(lines) ** 2).sum(1))
    return dict(lines=int(len(s)), max_abs_sum_grains=int(np.abs(s).max()), rms_sum_grains=float(np.sqrt((f64(s) ** 2).mean())),
                coin_rms_grains=float(np.sqrt((ref ** 2).mean())), ratio=float(np.sqrt((f64(s) ** 2).mean()) / max(np.sqrt((ref ** 2).mean()), 1e-300)),
                exactly_zero=int((s == 0).sum()))

res = dict(shape=tag(d, L), recipe=RECIPE, snaps={})
print(f"inplace {name}", flush=True)
for t, Bt, P, W in snapshots(d, L, RECIPE):
    rec = {"in_place": {}, "lanesum": {}, "rows": {}}
    for n in dr:
        w0w0 = dot(W0[n], W0[n])
        r = dict(shave_part=dot(Bt[n], W0[n]) / w0w0, push_part=dot(P[n], W0[n]) / w0w0,
                 size_P_over_W0=float(np.sqrt(dot(P[n], P[n]) / w0w0)))
        r["s"] = r["shave_part"] + r["push_part"]
        r["control"] = [dot(P[n], scramble(n, W0[n], k)) / w0w0 for k in (1, 2, 3)]
        rec["in_place"][n] = r
    # write lines: rows of proj/mproj ([in][out]: a row is one input neuron writing across the d lanes), wte rows, wpe rows, biases
    groups = {"wte": P["wte"], "wpe": P["wpe"]}
    for l in range(L):
        groups[f"proj.{l}.w"] = P[f"proj.{l}.w"]; groups[f"mproj.{l}.w"] = P[f"mproj.{l}.w"]
        groups[f"proj.{l}.b"] = P[f"proj.{l}.b"][None, :]; groups[f"mproj.{l}.b"] = P[f"mproj.{l}.b"][None, :]
    for g, a in groups.items(): rec["lanesum"][g] = lane_stats(a)
    rec["lanesum"]["out_plates_all"] = lane_stats(np.concatenate([P[f"{k}.{l}.w"] for l in range(L) for k in ("proj", "mproj")]))
    rec["lanesum"]["whole_model"] = lane_stats(np.concatenate(list(groups.values())))
    rec["lanesum"]["blank_whole_model"] = lane_stats(np.concatenate([W0["wte"], W0["wpe"]] + [W0[f"{k}.{l}.w"] for l in range(L) for k in ("proj", "mproj")]))
    rec["lanesum"]["read_plates_columns"] = lane_stats(np.concatenate([P[f"{k}.{l}.w"].T for l in range(L) for k in ("qkv", "fc")] + [P["head"].T]))
    # word rows meeting the blank first plates
    cw = cnt[:, None]
    for pl in ("qkv.0.w", "fc.0.w"):
        Q = f64(Bt[pl]); qn = (Q ** 2).sum() / Q.shape[0]
        def rho(X):
            X = f64(X); Y = X @ Q; assert np.isfinite(Y).all(); return float((cw * Y ** 2).sum() / ((cw * X ** 2).sum() * qn))
        out = {}
        for lab, X in (("blank_rows", Bt["wte"]), ("finished_rows", W["wte"]), ("push_rows", P["wte"])):
            out[lab] = rho(X) if np.any(X) else None
            if np.any(X):
                out[lab + "_lane_scrambled"] = [rho(X[:, np.random.default_rng(900 + k).permutation(d)]) for k in range(5)]
        if np.any(P["wte"]):
            bq, pq = f64(Bt["wte"]) @ Q, f64(P["wte"]) @ Q
            kk = lambda a, b: float(-2 * (cw * a * b).sum() / ((cw * a * a).sum() + (cw * b * b).sum()))
            out["k_wte"] = kk(bq, pq)
            out["k_wte_control"] = [kk(f64(scramble("wte", Bt["wte"], k)) @ Q, pq) for k in (1, 2, 3)]
        rec["rows"][pl] = out
    res["snaps"][t] = rec
    ip = rec["in_place"]; ls = rec["lanesum"]
    print(f"  step {t:5d}  s (control range): " + " ".join(f"{n.replace('.w', '')} {ip[n]['s']:.3f} ({min(ip[n]['control']):+.3f}..{max(ip[n]['control']):+.3f})" for n in dr), flush=True)
    print(f"             lane sums: write lines RMS {ls['whole_model']['rms_sum_grains']:.1f} grains (coin {ls['whole_model']['coin_rms_grains']:.3g}, ratio {ls['whole_model']['ratio']:.2e}), "
          f"out plates RMS {ls['out_plates_all']['rms_sum_grains']:.1f} ratio {ls['out_plates_all']['ratio']:.2e} max {ls['out_plates_all']['max_abs_sum_grains']}, wte {ls['wte']['ratio']:.2e}, "
          f"blank's ratio {ls['blank_whole_model']['ratio']:.3f}, read columns {ls['read_plates_columns']['ratio']:.3f}", flush=True)
    for pl, o in rec["rows"].items():
        if o.get("finished_rows") is None: continue
        print(f"             rows x blank {pl}: rho blank {o['blank_rows']:.3f} finished {o['finished_rows']:.3f} "
              f"(lanes scrambled {min(o['finished_rows_lane_scrambled']):.3f}-{max(o['finished_rows_lane_scrambled']):.3f})"
              + (f"  k_wte {o['k_wte']:+.4f} (ctrl {min(o['k_wte_control']):+.4f}..{max(o['k_wte_control']):+.4f})" if "k_wte" in o else ""), flush=True)
    json.dump(res, open(os.path.join(HERE, "evals", f"inplace_{name}.json"), "w"), indent=1)
print("done")
