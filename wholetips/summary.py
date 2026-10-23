"""Print wholetips/results.json as tables, and read the crude grid's words at the tips' own stored sizes (straight line
between the two grid points on either side, in bits a number; nothing chosen)."""
import json, numpy as np
R = json.load(open("/Users/abundancemachine/streaming-machine/wholetips/results.json"))
NV = 124_439_808; PARTS = ["word table", "position rows", "plates", "small pieces"]
KEYS = [("full_word_in_version_top5", 6000), ("version_word_in_full_top5", 6000), ("same_top", 6000), ("greedy_words", 480),
        ("greedy_texts", 30), ("record_next_word", 30)]
def pc(v, d): return f"{100 * v / d:6.2f}%"
print("version     bits/num    MB    | full word in its top5 | its word in full top5 | same top word | greedy /480 | texts /30 | record /30")
for n in ["full", "fp16", "tips_k1", "tips_k2", "tips_k3", "grid_b2", "grid_b3", "grid_b4", "grid_b5", "grid_b6"]:
    if n not in R: continue
    r = R[n]; w = r["whole"]
    print(f"{n:10s} {r['bits_a_number']:7.3f} {r['MB']:8.2f}  | {pc(w['full_word_in_version_top5'], 6000)} | {pc(w['version_word_in_full_top5'], 6000)} | "
          f"{pc(w['same_top'], 6000)} | {w['greedy_words']:4d} | {w['greedy_texts']:3d} | {w['record_next_word']:3d}"
          + (f"   plain {r['plain_bits_a_number']:.3f}" if "plain_bits_a_number" in r else ""))
print(f"\nternary 1.585 bits a number = {1.585 * NV / 8e6:.2f} MB;  record 80.711 MB = {80.711e6 * 8 / NV:.3f} bits a number over all stored numbers")
grid = sorted([R[n] for n in R if n.startswith("grid")], key=lambda r: r["bits_a_number"])
print("\nAT MATCHED STORED SIZE: the crude grid read at each tips version's bits a number")
for n in ["tips_k1", "tips_k2", "tips_k3"]:
    if n not in R or len(grid) < 2: continue
    t = R[n]; b = t["bits_a_number"]; xs = [g["bits_a_number"] for g in grid]
    if not (xs[0] <= b <= xs[-1]): print(f"  {n} at {b:.3f} bits: outside the grid's range {xs[0]:.3f}-{xs[-1]:.3f}"); continue
    i = int(np.searchsorted(xs, b)); i = max(1, min(i, len(xs) - 1)); lo, hi = grid[i - 1], grid[i]
    f = (b - lo["bits_a_number"]) / (hi["bits_a_number"] - lo["bits_a_number"])
    out = []
    for k, d in KEYS:
        g = lo["whole"][k] + f * (hi["whole"][k] - lo["whole"][k])
        out.append(f"{k} tips {t['whole'][k]} vs grid {g:.0f}")
    print(f"  {n} at {b:.3f} bits (grid between b{lo['spec']['b']} {lo['bits_a_number']:.3f} and b{hi['spec']['b']} {hi['bits_a_number']:.3f}): " + "; ".join(out))
print("\nBY PART (only that part cut, the rest uncut): full word in its top 5 / same top word / greedy, and bits a number of the part")
for n in ["tips_k1", "tips_k2", "tips_k3", "grid_b2", "grid_b3", "grid_b4", "grid_b5", "grid_b6"]:
    if n not in R: continue
    r = R[n]
    print(f"  {n:8s} " + " | ".join(f"{p}: {pc(r['by_part'][p]['full_word_in_version_top5'], 6000)} {pc(r['by_part'][p]['same_top'], 6000)} "
                                     f"{r['by_part'][p]['greedy_words']:3d} ({r['bits_by_part'][p]:.2f} b)" for p in PARTS))
print("\nCONTROL, errors shuffled within each row: full word in its top 5 / same top word / greedy (version itself in brackets)")
for n in ["tips_k1", "tips_k2", "tips_k3", "grid_b2", "grid_b3", "grid_b4", "grid_b5", "grid_b6"]:
    if n not in R: continue
    c = R[n]["control_errors_shuffled_in_row"]; w = R[n]["whole"]
    print(f"  {n:8s} {pc(c['full_word_in_version_top5'], 6000)} ({pc(w['full_word_in_version_top5'], 6000)})  {pc(c['same_top'], 6000)} "
          f"({pc(w['same_top'], 6000)})  {c['greedy_words']} ({w['greedy_words']})")
print("\nERROR LEFT, as a share of each part's own size (root of error power over the part's power)")
for n in ["tips_k1", "tips_k2", "tips_k3", "grid_b2", "grid_b3", "grid_b4", "grid_b5", "grid_b6", "fp16"]:
    if n not in R: continue
    print(f"  {n:8s} " + ", ".join(f"{p} {R[n]['error_share'][p]:.4f}" for p in PARTS))

print("\nBY PART AT MATCHED STORED SIZE: the grid's only-that-part words read at the tips' bits a number for that part")
for p in PARTS:
    gp = sorted([R[n] for n in R if n.startswith("grid")], key=lambda r: r["bits_by_part"][p]); xs = [g["bits_by_part"][p] for g in gp]
    for n in ["tips_k1", "tips_k2", "tips_k3"]:
        t = R[n]; b = t["bits_by_part"][p]; tw = t["by_part"][p]
        if not (xs[0] <= b <= xs[-1]):
            hi = gp[-1]; hw = hi["by_part"][p]
            print(f"  {p:13s} {n} at {b:.2f} b: {pc(tw['full_word_in_version_top5'], 6000)} {pc(tw['same_top'], 6000)} {tw['greedy_words']:3d} | "
                  f"grid's largest b{hi['spec']['b']} at {xs[-1]:.2f} b: {pc(hw['full_word_in_version_top5'], 6000)} {pc(hw['same_top'], 6000)} {hw['greedy_words']:3d} (outside the grid's range)")
            continue
        i = max(1, min(int(np.searchsorted(xs, b)), len(xs) - 1)); lo, hi = gp[i - 1], gp[i]
        f = (b - xs[i - 1]) / (xs[i] - xs[i - 1])
        g = {k: lo["by_part"][p][k] + f * (hi["by_part"][p][k] - lo["by_part"][p][k]) for k, _ in KEYS}
        print(f"  {p:13s} {n} at {b:.2f} b: {pc(tw['full_word_in_version_top5'], 6000)} {pc(tw['same_top'], 6000)} {tw['greedy_words']:3d} | "
              f"grid there (b{lo['spec']['b']}-b{hi['spec']['b']}): {pc(g['full_word_in_version_top5'], 6000)} {pc(g['same_top'], 6000)} {g['greedy_words']:5.0f}")
