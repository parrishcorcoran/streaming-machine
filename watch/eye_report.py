"""Print watch/eye_results.json (from eye_waves.py) with every control beside the real number.
Shares are counts over counts; the number in brackets is the share in thousandths by whole-number division.
    python3 watch/eye_report.py
"""
import json
ROOT = "/Users/abundancemachine/streaming-machine/watch/"
R = json.load(open(ROOT + "eye_results.json")); C = json.load(open(ROOT + "eye_check.json"))
M = json.load(open(ROOT + "results_real.json"))
def sh(x, y): return f"{x}/{y} ({x * 1000 // y if y else 0})"
def th(x, y): return x * 1000 // y if y else 0
def h(v, lo): return "  ".join(f"{lo + i:+d}:{c}" for i, c in enumerate(v))
def g(k): return R[k]
def fold(gp): return [sum(gp[:13])] + gp[13:]                              # gap hist from -24: fold the tail at -12

print("CHECKS")
print(f"  (a) every count recomputed from traj_lines_shuffled.npz: {C['identical']} of {C['keys']} identical" + (f"; differ {C['differ'][:8]}" if C['differ'] else ""))
print(f"  top-down height moves here vs report M1 real: {'identical' if g('w.top.real.move') == M['w.real.m1.move'] else 'DIFFERENT'};"
      f"  sign flips here {g('w.top.real.flips')} vs report {M['w.real.m2.flips']}")
print(f"  (b) each line's 24 writes in its own order: {g('abelian.b_end_differs')} of {g('abelian.b_end_values')} final values differ")
print(f"  eye-level heights, all snapshots (-1 = dark .. 15):")
for v in ("real", "b"): print(f"    {v:4s} {h(g(f'w.eye.{v}.heights')[:17], -1)}")
print("  water (eye level): the position's power of two, rise per write (-4..8+):")
for v in ("real", "b"): print(f"    {v:4s} {h(g(f'w.eye.{v}.water.rise'), -4)}")

def e1(k, label):
    m = g(k + ".move"); n = sum(m); one = m[7] + m[9]; gp = g(k + ".gap"); ng = sum(gp)
    print(f"  {label:9s} photons of change / photons of light {sh(g(k + '.ph_change'), g(k + '.ph_light'))}   values unchanged {sh(g(k + '.unchanged'), g(k + '.values'))}")
    print(f"  {'':9s} height moves 0 {sh(m[8], n)}  1 {sh(one, n)}  2+ {sh(n - m[8] - one, n)}   (up {th(sum(m[9:]), n)}, down {th(sum(m[:8]), n)});"
          f"  direction flips {sh(g(k + '.flips'), g(k + '.lit_both'))}")
    print(f"  {'':9s} change's top place below the light's: 2+ places {sh(sum(gp[:23]), ng)}  4+ {sh(sum(gp[:21]), ng)}  1 {th(gp[23], ng)}  level or above {th(sum(gp[24:]), ng)}")

print("\nE1  THINNESS PER WRITE (light before vs after one write, each at its own water)")
for vw in ("eye", "top"):
    for v in ("real", "b"): e1(f"w.{vw}.{v}", f"{vw} {v}")
print("  only where the water did not move (same power of two before and after):")
for vw in ("eye", "top"):
    for v in ("real", "b"): e1(f"w.{vw}.{v}.still", f"{vw} {v}")
print("  change's top place minus the light's top place (lit light, change != 0; tails folded at -12 and +8):")
for vw in ("eye", "top"):
    for v in ("real", "b"): print(f"    {vw} {v:4s} {h(fold(g(f'w.{vw}.{v}.gap')), -12)}")
print("  height move hist (-8..8):")
for vw in ("eye", "top"):
    for v in ("real", "b"): print(f"    {vw} {v:4s} {h(g(f'w.{vw}.{v}.move'), -8)}")
print("  per write (thousandths): change/light photons | height stays | direction flips | change 2+ places below light | water moved (positions of 648);  each eye-real eye-b top-real")
for t in range(24):
    cols = []
    for f in (lambda k: th(g(k + ".ph_change"), g(k + ".ph_light")), lambda k: th(g(k + ".move")[8], sum(g(k + ".move"))),
              lambda k: th(g(k + ".flips"), g(k + ".lit_both")), lambda k: th(sum(g(k + ".gap")[:23]), sum(g(k + ".gap")))):
        cols.append(" ".join(f"{f(f'w.{v}.t{t:02d}'):4d}" for v in ("eye.real", "eye.b", "top.real")))
    cols.append(" ".join(f"{g(f'w.eye.{v}.t{t:02d}.water.moved'):4d}" for v in ("real", "b")))
    print(f"   layer {t // 2:2d} {'arrival' if t % 2 == 0 else 'edge   '} " + " | ".join(cols))

def e2(k, label):
    p = lambda n: g(f"{k}.{n}")
    print(f"  {label:9s} height: reverse {sh(p('h_rev'), p('h_rev') + p('h_cont'))} of back-to-back moves (continue {p('h_cont')});  skipping stays: reverse "
          f"{sh(p('h_rev_skip'), p('h_rev_skip') + p('h_cont_skip'))};  moves {sh(p('h_moves'), p('h_steps'))}")
    print(f"  {'':9s} value:  reverse {sh(p('v_rev'), p('v_rev') + p('v_cont'))} of back-to-back moves;  skipping stays: reverse {sh(p('v_rev_skip'), p('v_rev_skip') + p('v_cont_skip'))}")
    print(f"  {'':9s} direction: flips back right after a flip {sh(p('d_flip_flip'), p('d_flip_flip') + p('d_flip_keep'))};  flips after a keep "
          f"{sh(p('d_keep_flip'), p('d_keep_flip') + p('d_keep_keep'))}")
    print(f"  {'':9s} flips per line (0..12+): {h(p('d_flips_per_line'), 0)}")

print("\nE2  BOBBING: each line at each position, its 25 snapshots (24 moves)")
for vw in ("eye", "top"):
    for v in ("real", "b"): e2(f"w.{vw}.{v}.bob", f"{vw} {v}")

print("\nE3  THE SURFACE BETWEEN PULSES: same snapshot, position p-1 -> p (c = another random position in place of p-1;"
      " c2 = each line's positions in its own random order)")
for vw in ("eye", "top"):
    for v in ("real", "c", "c2"): e1(f"p.{vw}.{v}", f"{vw} {v}")
print("  change's top place minus the light's top place (-12..+8):")
for vw in ("eye", "top"):
    for v in ("real", "c", "c2"): print(f"    {vw} {v:4s} {h(fold(g(f'p.{vw}.{v}.gap')), -12)}")
print("  per snapshot, eye: change/light photons | height stays | direction flips;  each real c c2")
for t in range(25):
    k = lambda v: f"p.eye.{v}.t{t:02d}"
    if t % 4 and t != 24: continue
    print(f"   snapshot {t:2d}  " + " | ".join(" ".join(f"{f(v):4d}" for v in ("real", "c")) for f in
          (lambda v: th(g(k(v) + ".ph_change"), g(k(v) + ".ph_light")), lambda v: th(g(k(v) + ".move")[8], sum(g(k(v) + ".move"))),
           lambda v: th(g(k(v) + ".flips"), g(k(v) + ".lit_both")))))
print("  bobbing along the positions of a text (each line, each snapshot):")
for vw in ("eye", "top"):
    for v in ("real", "c2"): e2(f"p.{vw}.{v}.bob", f"{vw} {v}")

def e4(k, label):
    p = lambda n: g(f"{k}.{n}")
    print(f"  {label:9s} pairs of moving lines moving the same way {sh(p('agree'), p('agree') + p('disagree'))};  groups with 3/4+ of movers one way {sh(p('swell34'), p('groups'))}")
    print(f"  {'':9s} rising share of movers per group (sixteenths 0..16): {h(p('rise16'), 0)}")
def restore(k, label):
    r = g(k + ".restore"); rows = [r[3 * i:3 * i + 3] for i in range(17)]
    top = [sum(x) for x in zip(*rows[13:17])]; bot = [sum(x) for x in zip(*rows[1:5])]
    print(f"  {label:9s} top quarter (12..15): moves down {sh(top[0], sum(top))}, up {th(top[2], sum(top))};  bottom quarter (0..3): moves up {sh(bot[2], sum(bot))}, down {th(bot[0], sum(bot))}")
    print(f"  {'':9s} by height, down/stay/up in thousandths:  " + "  ".join(f"{i - 1}:{th(x[0], sum(x))}/{th(x[1], sum(x))}/{th(x[2], sum(x))}" for i, x in enumerate(rows) if sum(x)))

print("\nE4  SHARED MOTION AND RESTORING (per write; one group = one position's 768 lines)")
for vw in ("eye", "top"):
    for v in ("real", "b"): e4(f"w.{vw}.{v}", f"{vw} {v}")
print("  only where the water did not move:")
for vw in ("eye", "top"):
    for v in ("real", "b"): e4(f"w.{vw}.{v}.still", f"{vw} {v}")
print("  restoring (height before the write; top-down heights are counted above the position's own water, -1 = under it or dark):")
for vw in ("eye", "top"):
    for v in ("real", "b"): restore(f"w.{vw}.{v}", f"{vw} {v}")
print("  restoring, only where the water did not move:")
for v in ("real", "b"): restore(f"w.eye.{v}.still", f"eye {v}")
print("  between pulses (p-1 -> p, one group = one pair of positions at one snapshot):")
for v in ("real", "c", "c2"): e4(f"p.eye.{v}", f"eye {v}")
for v in ("real", "c", "c2"): restore(f"p.eye.{v}", f"eye {v}")
