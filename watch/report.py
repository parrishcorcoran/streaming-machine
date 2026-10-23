"""Print the counts from watch/results_real.json, with every control beside the real number, and check that the run
with the lines in a random order (results_lines_shuffled.json, and the trajectories themselves) gives the very same.
Shares are counts over counts; the number in brackets is the same share in thousandths, by whole-number division.
    python3 watch/report.py
"""
import json, numpy as np
ROOT = "/Users/abundancemachine/streaming-machine/watch/"
R = json.load(open(ROOT + "results_real.json")); Q = json.load(open(ROOT + "results_lines_shuffled.json"))
a = np.load(ROOT + "traj.npz"); b = np.load(ROOT + "traj_lines_shuffled.npz"); pl = b["lines"]
nd = sum(int(np.count_nonzero(b[f"R{i}"] != a[f"R{i}"][:, :, pl])) for i in range(30)); nv = sum(a[f"R{i}"].size for i in range(30))
diff = [k for k in R if R[k] != Q.get(k)]
def sh(x, y): return f"{x}/{y} ({x * 1000 // y if y else 0})"
def h(v, lo): return "  ".join(f"{lo + i:+d}:{c}" for i, c in enumerate(v))
V = ("real", "b1", "b2", "c")
print("ORDER-FREE (Abelian) CHECKS")
print(f"  machine run with lines, edge neurons, heads, places in a head all in a random order: {nd} of {nv} stream values differ")
print(f"  every count below, recomputed from that run (lines in its random order): {len(R) - len(diff)} of {len(R)} identical" + (f"; differ: {diff[:10]}" if diff else ""))
print(f"  each line's 24 writes added in a random order of its own: {R['abelian.c_end_differs']} of {R['abelian.c_end_values']} final values differ")
print("controls: b1 = another position's stream, same line; b2 = another line, same position; c = each line's writes in its own random order\n")

print("M1  WAVEFRONT: top-photon place after a write minus before (places; tails folded at -8 and +8)")
for v in V:
    m = R[f"w.{v}.m1.move"]; n = sum(m)
    print(f"  {v:5s} stays {sh(m[8], n)}  up {sh(sum(m[9:]), n)}  down {sh(sum(m[:8]), n)}")
    print(f"        {h(m, -8)}")
print("  context: top place of the stream minus top place of the write landing on it (write != 0; tails at -8, +24)")
for v in V: print(f"  {v:5s} {h(R[f'w.{v}.m2.gap'], -8)}")

print("\nM2  COIN PUSHER (photons = 1s of |value|, sign apart)")
for v in V:
    p = lambda n: R[f"w.{v}.m2.{n}"]
    print(f"  {v:5s} places changed {sh(p('changed'), p('own'))} of the write's own photons;  changed where the write has no photon (carry runs) "
          f"{sh(p('beyond'), p('changed'))} of changed;  write photons whose place did not end up changed {sh(p('cancelled'), p('own'))};  sign flips {sh(p('flips'), p('values'))}")
for v in V: print(f"  {v:5s} carry-run places per value (0..12+): {h(R[f'w.{v}.m2.beyond_hist'], 0)}")
for v in V: print(f"  {v:5s} top changed place minus write's top place: {h(R[f'w.{v}.m2.reach'], -8)}")
print("  water level (the next read keeps 16 places below the top place any line reaches at that position):")
for v in V:
    p = lambda n: R[f"w.{v}.m2.{n}"]
    print(f"  {v:5s} stream photons under water before {sh(p('phA_under'), p('phA'))}, after {sh(p('phB_under'), p('phB'))};  "
          f"write photons landing under water {sh(p('write_under'), p('own'))};  changed places under water {sh(p('changed_under'), p('changed'))}")
for v in V: print(f"  {v:5s} water rise per write (places, -4..8+): {h(R[f'w.{v}.m2.rise'], -4)}")

print("  direction: the write points the same way as the stream value it lands on (both lit), per write: real b1 b2")
for v in V: print(f"  {v:5s} same sign {sh(R[f'w.{v}.m2.same_sign'], R[f'w.{v}.m2.both_lit'])}")
for t in range(0, 24, 2):
    print("   " + "   ".join(f"layer {u // 2:2d} {'arrival' if u % 2 == 0 else 'edge   '} " + " ".join(f"{R[f'w.{v}.t{u:02d}.m2.same_sign'] * 1000 // R[f'w.{v}.t{u:02d}.m2.both_lit']:4d}" for v in V[:3]) for u in (t, t + 1)))
print("\nM3  THE LINE'S OWN WAVE MOVED: write = +-value x 2^k + tip, with 8|tip| < |write|   (prediction with no structure, log-uniform ratios: about 363/1000)")
for v in V:
    p = lambda n: R[f"w.{v}.m3.{n}"]
    tot = p("y_ph") - p("y_ph_held") + p("tip_ph_held")
    print(f"  {v:5s} holds {sh(p('holds'), p('valid'))} (more than one k: {p('multi')});  where it holds, tip photons {sh(p('tip_ph_held'), p('y_ph_held'))} of write photons;"
          f"  all writes, photons left (tip where it holds, whole write elsewhere) {sh(tot, p('y_ph'))}")
for v in V: print(f"  {v:5s} k where it holds (-16..16): {h(R[f'w.{v}.m3.k'], -16)}")

print("\nM4  SHARED MOTION (a search: fewest tip photons over every k, both signs, and no gain): tip photons / write photons")
for v in V:
    p = lambda n: R[f"w.{v}.m4.{n}"]
    print(f"  {v:5s} one gain per position (all 768 lines) {sh(p('row'), p('y_ph'))}, no gain beats it at {sh(p('row_nogain'), p('rows'))} positions;  "
          f"one gain per line, same for every position {sh(p('line'), p('y_ph'))};  one gain per value {sh(p('el'), p('y_ph'))}")

print("\nPER WRITE (thousandths). M1 stays | M3 holds | M3 photons left | M4 per position | M4 per line | M4 per value;  each real b1 b2")
for t in range(24):
    g = lambda v, m, n: R[f"w.{v}.t{t:02d}.{m}.{n}"]
    def k3(v):
        y = g(v, "m3", "y_ph"); return (y - g(v, "m3", "y_ph_held") + g(v, "m3", "tip_ph_held")) * 1000 // y
    flat = [[g(v, "m1", "move")[8] * 1000 // sum(g(v, "m1", "move")) for v in V[:3]],
            [g(v, "m3", "holds") * 1000 // g(v, "m3", "valid") for v in V[:3]], [k3(v) for v in V[:3]], [g(v, "m4", "row") * 1000 // g(v, "m4", "y_ph") for v in V[:3]],
            [g(v, "m4", "line") * 1000 // g(v, "m4", "y_ph") for v in V[:3]], [g(v, "m4", "el") * 1000 // g(v, "m4", "y_ph") for v in V[:3]]]
    print(f"  layer {t // 2:2d} {'arrival' if t % 2 == 0 else 'edge   '} " + " | ".join(" ".join(f"{x:4d}" for x in c) for c in flat))

print("\nM5  PULSE TO PULSE: the same snapshot's light at position p-1 against position p (618 pairs a snapshot, 25 snapshots)")
for v in V[:3]:
    m = R[f"p.{v}.m1.move"]; n = sum(m); p = lambda x: R[f"p.{v}.m1.{x}"]
    print(f"  {v:5s} M1 top place stays {sh(m[8], n)}  up {sh(sum(m[9:]), n)}  down {sh(sum(m[:8]), n)};  places changed {sh(p('changed'), p('y_ph'))} "
          f"of p's photons;  photons of the plain difference {sh(p('diff_ph'), p('y_ph'))}")
    print(f"        {h(m, -8)}")
for v in V[:3]: print(f"  {v:5s} same sign at p-1 and p {sh(R[f'p.{v}.m1.same_sign'], R[f'p.{v}.m1.both_lit'])}")
for v in V[:3]:
    p = lambda x: R[f"p.{v}.m3.{x}"]; tot = p("y_ph") - p("y_ph_held") + p("tip_ph_held")
    print(f"  {v:5s} M3 p = +-(p-1) x 2^k + tip, 8|tip| < |p|: holds {sh(p('holds'), p('valid'))} (more than one k: {p('multi')}); "
          f"tip photons where it holds {sh(p('tip_ph_held'), p('y_ph_held'))}; all, photons left {sh(tot, p('y_ph'))}")
    print(f"        k: {h(R[f'p.{v}.m3.k'], -16)}")
for v in V[:3]:
    p = lambda x: R[f"p.{v}.m4.{x}"]
    print(f"  {v:5s} M4 one gain per position {sh(p('row'), p('y_ph'))} (no gain beats it: {sh(p('row_nogain'), p('rows'))});  one gain per line {sh(p('line'), p('y_ph'))};  "
          f"one gain per value {sh(p('el'), p('y_ph'))}")
print("\nPER SNAPSHOT (thousandths). M1 stays | M3 holds | M3 photons left | plain difference photons | M4 per position | M4 per line;  each real b1 b2")
for t in range(25):
    g = lambda v, m, n: R[f"p.{v}.t{t:02d}.{m}.{n}"]
    def k3(v):
        y = g(v, "m3", "y_ph"); return (y - g(v, "m3", "y_ph_held") + g(v, "m3", "tip_ph_held")) * 1000 // y
    flat = [[g(v, "m1", "move")[8] * 1000 // sum(g(v, "m1", "move")) for v in V[:3]],
            [g(v, "m3", "holds") * 1000 // g(v, "m3", "valid") for v in V[:3]], [k3(v) for v in V[:3]],
            [g(v, "m1", "diff_ph") * 1000 // g(v, "m1", "y_ph") for v in V[:3]],
            [g(v, "m4", "row") * 1000 // g(v, "m4", "y_ph") for v in V[:3]], [g(v, "m4", "line") * 1000 // g(v, "m4", "y_ph") for v in V[:3]]]
    lab = "entry        " if t == 0 else f"after L{(t - 1) // 2:2d} {'arr ' if t % 2 == 1 else 'edge'}"
    print(f"  {lab} " + " | ".join(" ".join(f"{x:4d}" for x in c) for c in flat))
