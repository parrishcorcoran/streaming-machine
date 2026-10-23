"""Collects inplace.py, work.py, remove.py and workcheck.py into one table of numbers (summary_work.log). Reads only
their JSON files; computes nothing new except the blank machine's words against the text (from words/ and evals/).
    python3 summarize_work.py
"""
import os, json, numpy as np
from model import HERE

BUILDS = [(d, L, r) for d, L in ((64, 2), (64, 4), (128, 2), (128, 4)) for r in "ABC"]
STEPS = [0, 10, 30, 100, 300, 1000, 2000, 3000]
J = lambda f: json.load(open(os.path.join(HERE, "evals", f))) if os.path.exists(os.path.join(HERE, "evals", f)) else None
lines = []
P = lambda s="": lines.append(s)

P("0. COMPACT, per build and step: head gamma [ctrl] | trunk plates gamma ABOVE/BELOW/IN control, median trunk gamma, median trunk k |"
  " exact dodge mean: qkv fc proj mproj head | in place s: wte wpe head | remove cost (nats): P alone, blank alone, iv_place, iv_light")
for d, L, r in BUILDS:
    w, de, ip, rm = J(f"work_w{d}_L{L}_{r}.json"), J(f"dodge_exact_w{d}_L{L}_{r}.json"), J(f"inplace_w{d}_L{L}_{r}.json"), J(f"remove_w{d}_L{L}_{r}.json")
    if not (w and de and ip): continue
    for t in STEPS[1:]:
        if str(t) not in w["snaps"] or str(t) not in de["snaps"]: continue
        pl = w["snaps"][str(t)]["plates"]; h = pl["head"]; trunk = {k: o for k, o in pl.items() if k != "head"}
        ab = sum(o["gamma"] > max(o["gamma_control"]) for o in trunk.values()); be = sum(o["gamma"] < min(o["gamma_control"]) for o in trunk.values())
        dx = de["snaps"][str(t)]; kd = {}
        for k, v in dx.items(): kd.setdefault(k.split(".")[0], []).append(v["blank"])
        i3 = ip["snaps"][str(t)]["in_place"]
        row = (f"  w{d}_L{L}_{r} {t:5d}: head {h['gamma']:.2f} [{min(h['gamma_control']):.2f}-{max(h['gamma_control']):.2f}] | {ab}/{be}/{len(trunk) - ab - be} "
               f"med {np.median([o['gamma'] for o in trunk.values()]):.2f} k {np.median([o['k'] for o in trunk.values()]):+.3f} | "
               + " ".join(f"{np.mean(kd[k]):.2f}" for k in ("qkv", "fc", "proj", "mproj", "head"))
               + f" | {i3['wte']['s']:.2f} {i3['wpe']['s']:.2f} {i3['head']['s']:.2f}")
        if rm and str(t) in rm["snaps"]:
            q = rm["snaps"][str(t)]; row += " | " + " ".join(f"{q[v]['loss_cost']:+.2f}" for v in ("pushes", "blank", "iv_place", "iv_light") if v in q)
        P(row)
P()
P("1. THE BLANK MACHINE (step 0, finished = blank), 128 held-out windows: cancelled share of the meetings, measured (coin toss)")
for d, L, r in BUILDS:
    if r != "A": continue
    w = J(f"work_w{d}_L{L}_{r}.json")
    if not w: continue
    pl = w["snaps"]["0"]["plates"]
    P(f"  w{d}_L{L}: " + "  ".join(f"{k} {o['cancel_finished']:.3f} ({o['coin_finished']:.3f}; formula {o['formula_finished']:.3f})" for k, o in pl.items()))
P("  the blank machine's words at the 30 texts' 480 positions (8 lead + 16): distinct words / equal to the true next word "
  "(450 positions with a next word) / equal to the step-3000 finished machine's word")
for d, L, r in BUILDS:
    ev = J(f"eval_w{d}_L{L}_{r}.json"); wj = json.load(open(os.path.join(HERE, "words", f"words_w{d}_L{L}_{r}.json")))
    texts = np.array(ev["texts"]); w0 = np.array(wj["snaps"]["0"]["finished"]["words"]); w3 = np.array(wj["snaps"]["3000"]["finished"]["words"])
    nxt = texts[:, 9:]; hit = int((w0[:, :15] == nxt).sum()); hit3 = int((w3[:, :15] == nxt).sum())
    P(f"  w{d}_L{L}_{r}: distinct {len(set(w0.ravel().tolist()))}  true next word {hit}/450 (step-3000 finished: {hit3}/450)  "
      f"same as step-3000 finished {int((w0 == w3).sum())}/480")

P(); P("2. IN PLACE: s = (W . W0)/(W0 . W0) per part; * = the push part (P . W0)/(W0 . W0) is larger in size than all 3 controls (P . scrambled W0)/(W0 . W0)")
for d, L, r in BUILDS:
    ip = J(f"inplace_w{d}_L{L}_{r}.json")
    if not ip: continue
    for t in STEPS:
        o = ip["snaps"][str(t)]["in_place"]
        P(f"  w{d}_L{L}_{r} {t:5d}: " + " ".join(f"{n.replace('.w', '')} {v['s']:.3f}" + ("*" if abs(v["push_part"]) > max(abs(c) for c in v["control"]) else "")
                                                 for n, v in o.items()))

P(); P("3. A COUNTER IN WHAT WAS BUILT, on the machine's own light (128 held-out windows).")
P("   per plate: gamma = share of the blank's pile still showing in the finished pile [control range, 3 scrambled blanks]; k")
P("   ABOVE = gamma above its control range (built along the blank), BELOW = under it (cancelled), IN = inside")
for d, L, r in BUILDS:
    w = J(f"work_w{d}_L{L}_{r}.json")
    if not w: continue
    for t in STEPS[1:]:
        if str(t) not in w["snaps"]: continue
        pl = w["snaps"][str(t)]["plates"]; ab = be = inn = 0; parts = []
        for k, o in pl.items():
            lo, hi = min(o["gamma_control"]), max(o["gamma_control"])
            tag_ = "A" if o["gamma"] > hi else ("B" if o["gamma"] < lo else "i")
            ab += tag_ == "A"; be += tag_ == "B"; inn += tag_ == "i"
            parts.append(f"{k} {o['gamma']:.2f}{tag_}/k{o['k']:+.2f}")
        P(f"  w{d}_L{L}_{r} {t:5d}: ABOVE {ab} BELOW {be} IN {inn} | " + " ".join(parts))

P(); P("4. DODGED BY THE LIGHT: |x . blank| / |x[lanes scrambled] . blank| per plate kind (mean over layers) "
       "[number of plates above / below all 5 lane-scrambled draws]")
for d, L, r in BUILDS:
    w = J(f"work_w{d}_L{L}_{r}.json")
    if not w: continue
    for t in STEPS:
        if str(t) not in w["snaps"]: continue
        pl = w["snaps"][str(t)]["plates"]; kinds = {}
        up = dn = 0
        for k, o in pl.items():
            kinds.setdefault(k.split(".")[0], []).append(o["dodge_ratio"])
            up += o["dodge_ratio"] > max(o["dodge_draw_spread"]); dn += o["dodge_ratio"] < min(o["dodge_draw_spread"])
        P(f"  w{d}_L{L}_{r} {t:5d}: " + "  ".join(f"{k} {np.mean(v):.3f}" for k, v in kinds.items()) + f"   [above {up}, below {dn} of {len(pl)}]")

P("   4b. the same with the chance level worked out exactly over every lane order (dodge_exact.py): blank per plate | built part, mean over plates")
for d, L, r in BUILDS:
    de = J(f"dodge_exact_w{d}_L{L}_{r}.json")
    if not de: continue
    for t in STEPS:
        if str(t) not in de["snaps"]: continue
        o = de["snaps"][str(t)]
        built = [v["built"] for v in o.values() if v["built"]]
        P(f"  w{d}_L{L}_{r} {t:5d}: " + " ".join(f"{k} {v['blank']:.3f}" for k, v in o.items()) + (f"  | built mean {np.mean(built):.2f}" if built else ""))

P(); P("5. CANCELLATION IN THE FINISHED MACHINE (mean over plates): finished piles (coin) | blank's part (coin) | built part (coin) "
       "| blank's share of the meeting size")
for d, L, r in BUILDS:
    w = J(f"work_w{d}_L{L}_{r}.json")
    if not w: continue
    for t in STEPS[1:]:
        if str(t) not in w["snaps"]: continue
        pl = w["snaps"][str(t)]["plates"].values(); m = lambda key: np.mean([o[key] for o in pl])
        P(f"  w{d}_L{L}_{r} {t:5d}: {m('cancel_finished'):.3f} ({m('coin_finished'):.3f}) | {m('cancel_blank'):.3f} ({m('coin_blank'):.3f}) | "
          f"{m('cancel_built'):.3f} ({m('coin_built'):.3f}) | {m('blank_share_of_meeting_size'):.3f}   order-free {w['snaps'][str(t)]['order_free']['identical']}/{w['snaps'][str(t)]['order_free']['checked']}")

P(); P("6. ZERO SUM OF THE WRITE LINES: lane sum of P, RMS in grains (coin-toss size; ratio) for the whole model and the out plates; "
       "the blank's own ratio; read plates' columns ratio")
for d, L, r in BUILDS:
    ip = J(f"inplace_w{d}_L{L}_{r}.json")
    if not ip: continue
    for t in STEPS[1:]:
        o = ip["snaps"][str(t)]["lanesum"]
        P(f"  w{d}_L{L}_{r} {t:5d}: whole {o['whole_model']['rms_sum_grains']:.4g} ({o['whole_model']['coin_rms_grains']:.3g}; {o['whole_model']['ratio']:.2e}) "
          f"max {o['whole_model']['max_abs_sum_grains']}  out plates {o['out_plates_all']['ratio']:.2e}  wte {o['wte']['ratio']:.2e}  "
          f"blank {o['blank_whole_model']['ratio']:.3f}  read columns {o['read_plates_columns']['ratio']:.3f}")
P("   word rows meeting the blank first plates (raw rows, rule 13.2): passage rho of finished rows [lane-scrambled range]; k_wte [control]")
for d, L, r in BUILDS:
    ip = J(f"inplace_w{d}_L{L}_{r}.json")
    if not ip: continue
    for t in (300, 1000, 3000):
        o = ip["snaps"][str(t)]["rows"]
        P(f"  w{d}_L{L}_{r} {t:5d}: " + "  ".join(
            f"{pl} rho {v['finished_rows']:.3f} [{min(v['finished_rows_lane_scrambled']):.3f}-{max(v['finished_rows_lane_scrambled']):.3f}] "
            f"(blank rows {v['blank_rows']:.3f}) k_wte {v['k_wte']:+.3f} [{min(v['k_wte_control']):+.3f}..{max(v['k_wte_control']):+.3f}]" for pl, v in o.items()))

P(); P("7. REMOVE IT (whole-number machine): held-out loss (first 64 held-out windows, 8,192 words) cost over finished; top-1 as finished /480; finished in top-5; "
       "greedy words as finished /480 (words before the first difference)")
for d, L, r in BUILDS:
    rm = J(f"remove_w{d}_L{L}_{r}.json")
    if not rm: continue
    for t in STEPS:
        s = rm["snaps"].get(str(t))
        if not s: continue
        f = s["finished"]
        row = f"  w{d}_L{L}_{r} {t:5d}: finished {f['loss']:.3f} (pairing ctrl top1 {f['top1_pairing_control']} greedy {f['greedy_pairing_control']})"
        for v in ("pushes", "blank", "iv_light", "iv_place"):
            if v in s:
                o = s[v]; row += f" | {v} {o['loss_cost']:+.3f} {o['top1_as_finished']} {o['finished_in_top5']} {o['greedy_as_finished']} ({o['greedy_words_before_first_difference_mean']:.1f})"
        P(row)
    if "3000" in rm["snaps"]: P(f"  w{d}_L{L}_{r} iv coefficients at 3000: light {{{', '.join(f'{k}: {v:.2f}' for k, v in rm['snaps']['3000']['coefs'].get('iv_light', {}).items())}}}  "
      f"place {{{', '.join(f'{k}: {v:.2f}' for k, v in rm['snaps']['3000']['coefs'].get('iv_place', {}).items())}}}")

P(); P("8. INDEPENDENT CHECKER (workcheck.py, float light, Hugging Face GPT-2): worst differences")
for d, L, r in BUILDS:
    c = J(f"workcheck_w{d}_L{L}_{r}.json")
    if c: P(f"  w{d}_L{L}_{r}: " + "  ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in c["worst"].items()))
open(os.path.join(HERE, "summary_work.log"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
