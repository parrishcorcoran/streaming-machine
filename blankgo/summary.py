"""Gather every build's numbers into tables (reads the json files only; no run, no prediction).
    python3 summary.py
"""
import os, json
HERE = os.path.dirname(os.path.abspath(__file__))
SH = ["w64_L2", "w64_L4", "w128_L2", "w128_L4"]; RC = "ABC"
def J(p):
    p = os.path.join(HERE, p); return json.load(open(p)) if os.path.exists(p) else None

print("HELD-OUT LOSS (nats a word; finished weights), per shape, recipe and step; bookkeeping failures; blank left")
for s in SH:
    for r in RC:
        b = J(f"build_{s}_{r}.json")
        if not b: continue
        st = sorted(int(t) for t in b["snaps"])
        print(f"  {s:8s} {r}  " + "  ".join(f"{t}:{b['snaps'][str(t)]['held_loss']:.3f}" for t in st)
              + f"   fails {b.get('bookkeeping_failures', '?')}  blank left {b['snaps'][str(st[-1])]['blank_left']:.6f}")

print("\nTAKING THE BLANK OUT (float64): held-out loss  finished | pushes alone | scrambled blank + pushes (3 draws)")
for s in SH:
    for r in RC:
        e = J(f"evals/eval_{s}_{r}.json")
        if not e: continue
        for t in sorted(int(x) for x in e["snaps"]):
            o = e["snaps"][str(t)]
            print(f"  {s:8s} {r} {t:5d}  {o['finished']['held_loss']:.3f} | {o['pushes']['held_loss']:.3f} (+{o['pushes']['held_loss'] - o['finished']['held_loss']:.3f}) | "
                  + " ".join(f"{o[k]['held_loss']:.3f}" for k in ('scr1', 'scr2', 'scr3'))
                  + f" (+{min(o[k]['held_loss'] for k in ('scr1','scr2','scr3')) - o['finished']['held_loss']:.3f}..+{max(o[k]['held_loss'] for k in ('scr1','scr2','scr3')) - o['finished']['held_loss']:.3f})"
                  + f"   top word as finished: pushes {o['pushes']['agree_finished_top1']}, scr {o['scr1']['agree_finished_top1']}/{o['scr2']['agree_finished_top1']}/{o['scr3']['agree_finished_top1']}; pairing control {o['control_pairing']}")

print("\nWHOLE-NUMBER MACHINE vs FLOAT (same plates): top word agree / 480, texts identical / 30, pairing control")
for s in SH:
    for r in RC:
        w = J(f"words/words_{s}_{r}.json")
        if not w: continue
        for t in sorted(int(x) for x in w["snaps"]):
            o = w["snaps"][str(t)]
            print(f"  {s:8s} {r} {t:5d}  " + "  ".join(
                f"{v}: {o[v]['agree_own']}/{o[v]['texts_identical']}/{o[v]['control']}"
                + (f" (as finished {o[v]['agree_finished']})" if v != 'finished' else "")
                + (f" maxmargin@diff {o[v]['max_float_margin_at_differ']:.4f}" if o[v]['agree_own'] < 480 else "") for v in o)
                + f"   lead moved at most {max(o[v].get('lead_moved_max', float('nan')) for v in o):.4f}")
        for tg in ("shuffle", "matmulint"):
            x = J(f"words/words_{s}_{r}-{tg}.json")
            if x:
                for t, o in x["snaps"].items():
                    for v, rec in o.items():
                        same = sum(a == b for a, b in zip(rec["piles_sha256"], w["snaps"][t][v]["piles_sha256"]))
                        print(f"  {s:8s} {r} {t:>5s}  {tg} {v}: piles identical in {same}/30 texts")

print("\nINDEPENDENT CHECKER (Hugging Face GPT-2, own rebuild): worst |loss diff|, words, replayed pushes")
for s in SH:
    for r in RC:
        c = J(f"evals/checker_{s}_{r}.json")
        if c: print(f"  {s:8s} {r}  {c['worst_loss_diff']:.1e}  words {min(c['words'])}-{max(c['words'])}/480  replay "
                    + "  ".join(f"step {t}: {x['differ']} of {x['values']:,} differ (max {x['max_grains']})" for t, x in c["replay"].items()))
for s in SH:
    for r in RC:
        c = J(f"machines/convert_{s}_{r}.json")
        if c: print(f"  convert {s} {r}: neurons whose power the blank sets, max over steps "
                    f"{max(v['blank_sets_power'] / v['neurons'] for v in c.values()) * 100:.1f}%")

print("\nSIZES at the last snapshot (context only, not a measure of the work): pushes RMS / blank RMS per part")
for s in SH:
    for r in RC:
        b = J(f"build_{s}_{r}.json")
        if not b: continue
        last = b["snaps"][str(max(int(t) for t in b["snaps"]))]
        print(f"  {s:8s} {r} step {last['step']}: " + "  ".join(f"{k} {v['pushes_rms'] / v['blank_rms']:.2f}" for k, v in last["sizes"].items()))
