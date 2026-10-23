"""Tables from zerosum_<size>.json and zerosum_spans70.json (reads only the json; no new counts).  2026-10-06.

    python3 zerosum_summary.py [sizes=14m,70m,160m]
"""
import os, sys, json
import numpy as np

A = dict(a.split("=") for a in sys.argv[1:])
SIZES = A.get("sizes", "14m,70m,160m").split(",")
HERE = os.path.dirname(os.path.abspath(__file__))
KINDS = [("embed", "input word table", "w"), ("proj", "attention-out", "w"), ("mproj", "MLP-out", "w"),
         ("qkv", "query/key/value", "r"), ("fc", "MLP-in", "r"), ("head", "output word table", "r")]
out = []
def say(s=""): print(s); out.append(s)
def rng_(v, f="{:.2f}"): return (f.format(min(v)) + "-" + f.format(max(v))) if len(v) > 1 else f.format(v[0])
def pick(P, k): return [(n, p) for n, p in P.items() if n.split(".")[0] == k]

for size in SIZES:
    fn = os.path.join(HERE, f"zerosum_{size}.json")
    if not os.path.exists(fn): continue
    J = json.load(open(fn)); P = J["plates"]
    say(f"==================== pythia-{size}   (c = {J['c30']}/2^30) ====================")
    say("ITEM 1. Size of the change D = W - W0 (pooled over plates of a kind, weighted by count; scr = finished weight against a")
    say("        blank weight from elsewhere in the plate).  10->12 = share with |D| <= 0.2 |W0|.")
    for k, lab, kind in KINDS:
        ps = pick(P, k); n = np.array([p["size"]["D"]["n"] for _, p in ps], float)
        def pool(var, f): return float((np.array([p["size"][var][f] for _, p in ps]) * n).sum() / n.sum())
        say(f"  {lab:18s} 10->12 {100 * pool('D', 'within20'):5.2f}% (scr {100 * pool('D_scramble', 'within20'):5.2f})  "
            f"|D|<|W0| {100 * pool('D', 'd_lt_blank'):5.1f}% (scr {100 * pool('D_scramble', 'd_lt_blank'):5.1f})  "
            f"|D|<|W| {100 * pool('D', 'd_lt_fin'):5.1f}% (scr {100 * pool('D_scramble', 'd_lt_fin'):5.1f})  "
            f"|W|<|W0| {100 * pool('D', 'fin_lt_blank'):5.1f}%  sign kept {100 * pool('D', 'sign_kept'):5.1f}% (scr {100 * pool('D_scramble', 'sign_kept'):5.1f})  "
            f"med|D|/|W0| {rng_([p['size']['D']['med_d_over_blank'] for _, p in ps])}  med|D|/|W| {rng_([p['size']['D']['med_d_over_fin'] for _, p in ps])}  "
            f"rms D/W0 {rng_([p['size']['D']['rms_d'] / p['size']['D']['rms_blank'] for _, p in ps])} W/W0 {rng_([p['size']['D']['rms_fin'] / p['size']['D']['rms_blank'] for _, p in ps])}")
        say(f"  {'':18s} D': 10->12 {100 * pool('Dp', 'within20'):5.2f}% (scr {100 * pool('Dp_scramble', 'within20'):5.2f})  |D'|<|W0| {100 * pool('Dp', 'd_lt_blank'):5.1f}%  "
            f"|D'|<|W| {100 * pool('Dp', 'd_lt_fin'):5.1f}%  med|D'|/|W| {rng_([p['size']['Dp']['med_d_over_fin'] for _, p in ps])}")
    tot = sum(p["size"]["D"]["n"] for p in P.values())
    w20 = sum(p["size"]["D"]["within20"] * p["size"]["D"]["n"] for p in P.values()) / tot
    w20s = sum(p["size"]["D_scramble"]["within20"] * p["size"]["D"]["n"] for p in P.values()) / tot
    say(f"  all plates: 10->12 {100 * w20:.2f}% (scr {100 * w20s:.2f}) of {tot:,} weights")
    say()
    say("ITEM 2. Ups against downs of the changes.  rms z = root mean square over lines (write) or neurons (read) of")
    say("        net / sqrt(sum of squares); a fair coin gives 1.  [range over the kind's plates]")
    for k, lab, kind in KINDS:
        ps = pick(P, k)
        def r(var, lab2, f="rms", where="units"): return rng_([p[var][lab2][where][f] for _, p in ps])
        unit = "write line over lanes" if kind == "w" else "read neuron over lines"
        say(f"  {lab:18s} {unit:23s} D {r('D', 'real')} (scr {r('D', 'scramble')}, fresh {r('D', 'fresh')}) | D' {r('Dp', 'real')} "
            f"(scr {r('Dp', 'scramble')}, fresh {r('Dp', 'fresh')}) | W {r('W', 'real')} | W0 {r('W0', 'real')}")
        say(f"  {'':18s} {'':23s} share beyond 3: D {rng_([100 * p['D']['real']['units']['over3'] for _, p in ps], '{:.2f}')}% D' "
            f"{rng_([100 * p['Dp']['real']['units']['over3'] for _, p in ps], '{:.2f}')}% (fresh {rng_([100 * p['Dp']['fresh']['units']['over3'] for _, p in ps], '{:.2f}')}%)"
            f" | share under 0.5: D' {rng_([100 * p['Dp']['real']['units']['under_half'] for _, p in ps], '{:.1f}')}% (fresh {rng_([100 * p['Dp']['fresh']['units']['under_half'] for _, p in ps], '{:.1f}')}%)")
        say(f"  {'':18s} cross (one lane over all {'write lines' if kind == 'w' else 'read neurons'}): D' {r('Dp', 'real', where='cross')} (scr {r('Dp', 'scramble', where='cross')}, fresh {r('Dp', 'fresh', where='cross')})")
        if k == "qkv":
            for part in "qkv":
                say(f"  {'':18s} {part} neurons: D' {rng_([p['Dp']['real']['qkv_parts'][part]['rms'] for _, p in ps])}  D {rng_([p['D']['real']['qkv_parts'][part]['rms'] for _, p in ps])}")
        pz = [p["D"]["real"]["plate"]["net_z"] for _, p in ps]; pzp = [p["Dp"]["real"]["plate"]["net_z"] for _, p in ps]
        fz = [p["Dp"]["fresh"]["plate"]["net_z"] for _, p in ps]
        cz = [p["D"]["real"]["plate"]["count_z"] for _, p in ps]; czp = [p["Dp"]["real"]["plate"]["count_z"] for _, p in ps]
        say(f"  {'':18s} per plate: net z D {' '.join(f'{v:+.1f}' for v in pz)} | D' {' '.join(f'{v:+.1f}' for v in pzp)} | fresh {' '.join(f'{v:+.1f}' for v in fz)}")
        say(f"  {'':18s}            count z (how many up minus how many down) D {' '.join(f'{v:+.1f}' for v in cz)} | D' {' '.join(f'{v:+.1f}' for v in czp)}")
        for form, fl in (("ss", "sign and size"), ("sd", "signed digits")):
            o3 = [p["D"]["real"]["places"][form]["over3"] for _, p in ps]; o3p = [p["Dp"]["real"]["places"][form]["over3"] for _, p in ps]
            f3 = [p["Dp"]["fresh"]["places"][form]["over3"] for _, p in ps]; lv = [p["Dp"]["real"]["places"][form]["live"] for _, p in ps]
            mz = [p["Dp"]["real"]["places"][form]["maxz"] for _, p in ps]
            say(f"  {'':18s} places, {fl:13s}: beyond 3 D {sum(o3)} D' {sum(o3p)} (fresh {sum(f3)}) of {sum(lv)}; largest |z| D' {rng_(mz, '{:.1f}')}; "
                f"cells ({'line' if kind == 'w' else 'neuron'}, place) rms z D' {rng_([p['Dp']['real']['places'][form]['cell_rms'] for _, p in ps])} "
                f"(scr {rng_([p['Dp']['scramble']['places'][form]['cell_rms'] for _, p in ps])}, fresh {rng_([p['Dp']['fresh']['places'][form]['cell_rms'] for _, p in ps])})")
    say("  write biases (start at 0; z over the lanes, attention-out / MLP-out): " + " ".join(f"L{l} {v['z']:+.2f}/{v['z_mlp']:+.2f}" for l, v in J["biases"].items()))
    say()

fn = os.path.join(HERE, "zerosum_spans70.json")
if os.path.exists(fn):
    J = json.load(open(fn)); P = J["plates"]
    say("==================== pythia-70m, ITEM 3: the change over each span between saved steps ====================")
    say("raw = W(t2) - W(t1); push = W(t2) - round(c2/c1 W(t1)) (the shave over the span taken out); shave = what the shave took.")
    say("rms z over write lines (over lanes) or read neurons (over their lines); [range over the kind's plates]")
    spans = ["0->1000", "1000->10000", "10000->50000", "50000->143000", "0->10000", "0->50000", "0->143000"]
    for k, lab, kind in KINDS:
        ps = pick(P, k)
        say(f"  {lab:18s} ({'write lines' if kind == 'w' else 'read neurons'})")
        for sp in spans:
            def g(lab2, f="units"): return rng_([p["spans"][sp][lab2][f]["rms"] for _, p in ps])
            say(f"    {sp:14s} raw {g('raw')} push {g('push')} (scr {rng_([p['spans'][sp]['push']['scramble']['rms'] for _, p in ps])} "
                f"fresh {rng_([p['spans'][sp]['push']['fresh']['rms'] for _, p in ps])}) shave {g('shave')} | size of push / plate before "
                f"{rng_([p['spans'][sp]['push']['rms'] / p['spans'][sp]['rms_before'] for _, p in ps])} | cross push {g('push', 'cross')}")
        say(f"    plate itself rms z at 0/1000/10000/50000/143000: " + " ".join(
            "/".join(f"{p['plate_rms_z'][t]:.2f}" for t in ("0", "1000", "10000", "50000", "143000")) for _, p in ps))
    say("  write biases z: " + "; ".join(f"L{l} " + " ".join(f"{k} {v:+.2f}" for k, v in d.items()) for l, d in J["biases"].items()))
    for t in ("0", "1000", "10000", "50000", "143000"):
        N = J["norms"]
        say(f"  gains spread/mean at {t:>6s}: ln1 {rng_([N[f'ln1.{l}'][t]['gain_spread'] for l in range(6)], '{:.3f}')} ln2 {rng_([N[f'ln2.{l}'][t]['gain_spread'] for l in range(6)], '{:.3f}')} "
            f"lnf {N['lnf'][t]['gain_spread']:.3f} | offsets rms ln1 {rng_([N[f'ln1.{l}'][t]['offset_rms'] for l in range(6)], '{:.3f}')} "
            f"ln2 {rng_([N[f'ln2.{l}'][t]['offset_rms'] for l in range(6)], '{:.3f}')} lnf {N['lnf'][t]['offset_rms']:.3f}")
open(os.path.join(HERE, "zerosum_summary.log"), "w").write("\n".join(out) + "\n")
