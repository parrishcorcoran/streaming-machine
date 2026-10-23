"""The tables for halves.py: merges turns/halves_part_*.npz (each a range of the 30 refs texts) and prints steps 2-4.
Every number is a ratio of exact rationals (halves.py checks each split digit for digit), printed in floating point
for reading only. The predictions are in halves.py's docstring, written before any run.
    python3 turns/halves_report.py      -> turns/halves_report.log
"""
import glob, numpy as np
OUTD = "/Users/abundancemachine/streaming-machine/turns"
BK = ("top", "mproj", "bend", "fc_bias", "fc_round", "arrival")
parts = sorted(glob.glob(f"{OUTD}/halves_part_*.npz"))
L = [np.load(p) for p in parts]
NT = len(L[0]["done"]); done = np.zeros(NT, bool)
O = {k: L[0][k].copy() for k in L[0].files if k not in ("checks", "check_names", "done")}
chk = {}
for d in L:
    m = d["done"]; done |= m
    for k in O: O[k][m] = d[k][m]
    for n, v in zip(d["check_names"], d["checks"]): chk[str(n)] = chk.get(str(n), 0) + int(v)
idx = np.flatnonzero(done); nx = len(idx)
print(f"parts: {[p.split('/')[-1] for p in parts]};  texts done: {nx} of {NT}")
print("CHECKS, all parts:", chk)
print(f"  per-neuron (a)+(b) == 2 x part: {chk['neuron']}/{chk['neuron_n']};  part + earlier + stream + constants == lead, digit for"
      f" digit: {chk['rest']}/{chk['rest_n']};  copy == film_lead.rev_exact_from: {chk['copy']}/{chk['copy_n']};  patched plate =="
      f" machine.py's: {chk['plate']}/{len(parts)}")
print(f"  all layers at once exact: {chk['s3']}/{chk['s3_n']} (straight), {chk['cut']}/{chk['cut_n']} (cut);  write split exact:"
      f" {chk['write']}/{nx};  order-free sums: {chk['order']}/{chk['order_n']};  16-place (a)+(b) == 2 x part: {chk['s16']}/{chk['s16_n']}")
print(f"  the symmetric half: 2e < 0 at {int(O['eneg'][idx].sum())} of {nx * 12 * 5 * 3072:,} neurons x positions; |2e| > |H| + 1 at"
      f" {int(O['ebig'][idx].sum())}")

def med(x): return f"{np.nanmedian(x):8.3f} [{np.nanpercentile(x, 10):7.3f},{np.nanpercentile(x, 90):7.3f}]"
def medw(x): return f"{np.nanmedian(x):11.4g} [{np.nanpercentile(x, 10):10.4g},{np.nanpercentile(x, 90):10.4g}]"
s2 = O["s2"][idx]; T0 = O["target"][idx]
for w_, nm in ((0, "THE LEAD (answer - 2nd)"), (1, "CONTROL: A RANDOM WORD'S LEAD (answer - random word)")):
    print(f"\nSTEP 2, {nm}, exact, {nx} texts: the part through layer l's NEWEST bends, split into its two halves."
          f" Per layer, over texts: median [10th, 90th]")
    print("  layer | straight / lead            | symmetric / lead           | part / lead                | gross straight share"
          "       | |str| / (|str| + |sym|)    | opposite | SHUFFLED sym / lead (values changed) | older positions' bends: straight, sym / lead")
    for l in range(12):
        x = s2[:, l, w_]; ld = T0[:, w_]
        print(f"  {l:5d} | {med(x[:, 0] / ld)} | {med(x[:, 1] / ld)} | {med(x[:, 2] / ld)} | {med(x[:, 3] / (x[:, 3] + x[:, 4]))}"
              f" | {med(np.abs(x[:, 0]) / (np.abs(x[:, 0]) + np.abs(x[:, 1])))} | {int((np.sign(x[:, 0]) != np.sign(x[:, 1])).sum()):3d}/{nx}"
              f"   | {med(x[:, 5] / ld)}           | {np.nanmedian(x[:, 7] / ld):8.3f}, {np.nanmedian(x[:, 8] / ld):8.3f}")
    x = s2[:, :, w_]; ld = T0[:, w_][:, None]
    print(f"  all {x[..., 0].size} layer-texts: |sym| > |straight| at {int((np.abs(x[..., 1]) > np.abs(x[..., 0])).sum())};"
          f" opposite signs at {int((np.sign(x[..., 0]) != np.sign(x[..., 1])).sum())}; both halves larger than the part at"
          f" {int(((np.abs(x[..., 0]) > np.abs(x[..., 2])) & (np.abs(x[..., 1]) > np.abs(x[..., 2]))).sum())};"
          f" gross straight share > 1/2 at {int((x[..., 3] > x[..., 4]).sum())}")
    print(f"     median |straight| / lead {np.nanmedian(np.abs(x[..., 0]) / ld):.3f}, |sym| / lead {np.nanmedian(np.abs(x[..., 1]) / ld):.3f},"
          f" |part| / lead {np.nanmedian(np.abs(x[..., 2]) / ld):.3f};  shuffle: |sym - shuffled sym| / |sym| median"
          f" {np.nanmedian(np.abs(x[..., 1] - x[..., 5]) / np.abs(x[..., 1])):.3f}, |shuffled sym| / |sym| median"
          f" {np.nanmedian(np.abs(x[..., 5]) / np.abs(x[..., 1])):.3f}; straight shuffled: |str - shuf| / |str| median"
          f" {np.nanmedian(np.abs(x[..., 0] - x[..., 6]) / np.abs(x[..., 0])):.3f}")
s16 = O["s2_16"]; T = O["target"]; d16 = ~np.isnan(s16[:, 0, 0, 0]); n16 = int(d16.sum())
print(f"\nSTEP 2 in the 16-place style (reverse16's steps), {n16} texts, the lead (answer - 2nd): median [10th, 90th]")
for l in range(12):
    x = s16[d16, l, 0]; ld = T[d16, 0]
    print(f"  {l:5d} | straight / lead {med(x[:, 0] / ld)} | symmetric / lead {med(x[:, 1] / ld)} | |str|/(|str|+|sym|)"
          f" {med(np.abs(x[:, 0]) / (np.abs(x[:, 0]) + np.abs(x[:, 1])))} | opposite signs {int((np.sign(x[:, 0]) != np.sign(x[:, 1])).sum())}/{n16}")
g = np.abs(s16[idx][:, :, 0, :2] - s2[:, :, 0, :2]) / np.maximum(np.abs(s2[:, :, 0, :2]), 1)
print(f"  16-place against exact ({nx} texts): relative gap of the two halves' parts median {np.nanmedian(g):.1e}, max {np.nanmax(g):.1e}")
wr = O["write"][idx]
print(f"\nTHE WRITE OF EACH LAYER INTO THE STREAM: mproj's output at the NEWEST position split into (H/2) met with mproj and e met"
      f" with mproj, exact; sizes in stream units (2^-16) summed over 768 lines; {nx} texts, median [10th, 90th]")
print("  layer | sum|straight part|                  | sum|symmetric part|                 | sum|write| (no roundings, bias)     |"
      " straight share (sum|.|)   | straight share (energy)    | |write| / (|str| + |sym|)")
for l in range(12):
    x = wr[:, l, 4]
    print(f"  {l:5d} | {medw(x[:, 0])} | {medw(x[:, 1])} | {medw(x[:, 2])} | {med(x[:, 0] / (x[:, 0] + x[:, 1]))} |"
          f" {med(x[:, 3] ** 2 / (x[:, 3] ** 2 + x[:, 4] ** 2))} | {med(x[:, 2] / (x[:, 0] + x[:, 1]))}")
x = wr.reshape(-1, 6)
print(f"  all layers, all 5 positions: straight share (sum|.|) median {np.median(x[:, 0] / (x[:, 0] + x[:, 1])):.3f} [min"
      f" {np.min(x[:, 0] / (x[:, 0] + x[:, 1])):.3f}, max {np.max(x[:, 0] / (x[:, 0] + x[:, 1])):.3f}]; |write| / (|str|+|sym|) median"
      f" {np.median(x[:, 2] / (x[:, 0] + x[:, 1])):.3f}")
s3 = O["s3"][idx]
names = ["fixed machine's entry part", "symmetric halves, newest position", "symmetric halves, older positions"] + \
        [f"constants: {k}" for k in BK] + ["CUT: entry part (stream + arrivals only)", "CUT: whole bend outputs, newest",
                                           "CUT: whole bend outputs, older", "CUT: constants", "SHUFFLED symmetric halves (values changed)"]
for w_, nm in ((0, "the lead (answer - 2nd)"), (1, "CONTROL: a random word's lead (answer - random word)"), (2, "the answer's score")):
    ld = T0[:, w_]
    print(f"\nSTEP 3, ALL LAYERS AT ONCE, {nm}, exact, {nx} texts: each part / the target, median [10th, 90th];  |.| median")
    for k, n_ in enumerate(names):
        print(f"  {n_:50s} {medw(s3[:, w_, k] / ld)}   |.| {np.nanmedian(np.abs(s3[:, w_, k] / ld)):11.4g}")
    thr = (s3[:, w_, 0] - s3[:, w_, 9]) / ld
    print(f"  {'entry light through >= 1 fixed turn (fixed - cut)':50s} {medw(thr)}   |.| {np.nanmedian(np.abs(thr)):11.4g}")
    sym = (s3[:, w_, 1] + s3[:, w_, 2]) / ld
    fx = (s3[:, w_, 0] + s3[:, w_, 6]) / ld
    print(f"  {'symmetric halves, all':50s} {medw(sym)};  fixed machine and symmetric halves of opposite sign in"
          f" {int((np.sign(s3[:, w_, 0]) != np.sign(s3[:, w_, 1] + s3[:, w_, 2])).sum())}/{nx}")
    print(f"  {'fixed turns with their bias part (entry + fc bias)':50s} {medw(fx)}   |.| {np.nanmedian(np.abs(fx)):11.4g}")
    print(f"  fixed machine's entry part < 1/3 of the target in {int((s3[:, w_, 0] / ld < 1 / 3).sum())}/{nx} texts; between 0 and 1/3 in"
          f" {int(((s3[:, w_, 0] / ld >= 0) & (s3[:, w_, 0] / ld < 1 / 3)).sum())}/{nx}; |part| > 2 targets in"
          f" {int((np.abs(s3[:, w_, 0] / ld) > 2).sum())}/{nx}; log10 |part / target| median {np.nanmedian(np.log10(np.abs(s3[:, w_, 0] / ld))):.2f}")
    print(f"  shuffle: (shuffled sym - sym) / target median {np.nanmedian((s3[:, w_, 13] - s3[:, w_, 1] - s3[:, w_, 2]) / ld):.4g},"
          f" |.| median {np.nanmedian(np.abs(s3[:, w_, 13] - s3[:, w_, 1] - s3[:, w_, 2]) / np.abs(ld)):.4g} targets;"
          f" |shuffled - real| / |real| median {np.nanmedian(np.abs(s3[:, w_, 13] - s3[:, w_, 1] - s3[:, w_, 2]) / np.abs(s3[:, w_, 1] + s3[:, w_, 2])):.3f}")
    sp = O["s3_pos"][idx, w_] / ld[:, None]
    print("  fixed machine's entry part by token position (each token's own entry numbers) / target, median: "
          + ", ".join(f"pos {p}: {np.nanmedian(sp[:, p]):.4g}" for p in range(5)))
    sl = O["s3_lay"][idx, w_] / ld[:, None]
    print("  symmetric halves by layer (all positions) / target, median: " + ", ".join(f"{l}: {np.nanmedian(sl[:, l]):.4g}" for l in range(12)))
    print("  |.| median by layer: " + ", ".join(f"{l}: {np.nanmedian(np.abs(sl[:, l])):.4g}" for l in range(12)))
