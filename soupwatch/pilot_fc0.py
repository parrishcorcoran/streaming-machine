"""PART 2, ONE EXACT PILOT: layer 0's neurons-in bank (fc.0) over one whole soup run, poured as changes only.

The bank keeps its piles from the wave before. Each wave only d = (this wave's input to fc.0) - (last wave's input) is
poured, and the piles are kept by adding:  pile(w) = pile(w-1) + W d  (whole numbers, exact). Every wave the kept piles
are compared with the full recompute W x(w), np.array_equal. The same pour is done a second time with the 768 lines
in a random order (fixed seed): the kept piles must be identical (the adding does not care about order).

Then the gates: "a gate is recomputed only where its pile moved". Checked exactly: keep last wave's bend outputs,
recompute only the gates whose pile moved (with this wave's gauge), and compare with the true bend outputs.

Work (photon meetings, light photons x weight photons) both ways: the whole light of every moving position each
wave, and only d. Shuffle: d taken against a random other position's last input.

    python3 soupwatch/pilot_fc0.py [text=0]
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import soup_lib as L

opts = dict(a.split("=") for a in sys.argv[1:] if "=" in a); t = int(opts.get("text", 0)); N = 16
refs = json.load(open(L.HERE + "refs_many.json"))
prompt = (refs[t]["ids"] + refs[t]["ref"])[:5]; S = len(prompt); P = S + N; reads = list(range(S - 1, P - 1))
w, we, b = L.W["fc.0"]; rowph = L.ROWPH["fc.0"]
perm = np.random.default_rng(5).permutation(768); wp = w[perm]
kept = np.zeros((P, 3072), np.int64); kept_p = np.zeros((P, 3072), np.int64)
x_prev = np.zeros((P, 768), np.int64); G_prev = None; pile_prev = None
tot_full = tot_delta = tot_sh = tot_every = 0; all_equal = all_perm = True; gate_ok_all = True
print(f"text {t}: {refs[t]['prompt']!r} -> prompt = first 5 words, {N} future words; bank fc.0 (768 lines in, 3,072 piles out)")
print("wave | moving positions | kept piles == full recompute | same in random line order | gates: recompute only where pile moved == true | work: pour whole | pour d | shuffle d")
T = prompt + [prompt[-1]] * N; wv = 0
while True:
    wv += 1
    rec = L.light(T, reads); newT = prompt + rec["tops"]
    x = rec["ins"]["fc.0"]; d = x - x_prev
    mv = np.flatnonzero((d != 0).any(1)) if wv > 1 else np.arange(P)
    if len(mv):
        kept[mv] += L.product(d[mv], w)                                # pour only the change, keep by adding
        kept_p[mv] += L.product(d[mv][:, perm], wp)                     # the same, lines in a random order
    full = L.product(x, w)
    eq, eqp = np.array_equal(kept, full), np.array_equal(kept_p, full)
    all_equal &= eq; all_perm &= eqp
    ex, LS = rec["gauge"]["fc.0"]; G = rec["G"][0]
    if G_prev is None: gate_ok = True; gates = P * 3072
    else:
        moved = kept != pile_prev                                      # gates whose pile moved
        H = L.power(kept, (we[None, :] + ex[:, None] + L.ES) * 65536 - LS[:, None]) + b
        Gp = np.where(moved, L.edge(H), G_prev); gate_ok = np.array_equal(Gp, G); gates = int(moved.sum())
    gate_ok_all &= gate_ok
    g = np.random.default_rng(1000 * t + wv); q = (np.arange(P) + g.integers(1, P, size=P)) % P
    wf = int((L.bc(x[mv]) @ rowph).sum()); wd = int((L.bc(d[mv]) @ rowph).sum())
    ws = wf if wv == 1 else int((L.bc(x[mv] - x_prev[q][mv]) @ rowph).sum())
    tot_full += wf; tot_delta += wd; tot_sh += ws; tot_every += int((L.bc(x) @ rowph).sum())
    print(f"{wv:4d} | {len(mv):2d} (from position {mv[0] if len(mv) else '-'}) | {'EQUAL' if eq else 'DIFFERENT'} | {'EQUAL' if eqp else 'DIFFERENT'} | "
          f"{'EQUAL' if gate_ok else 'DIFFERENT'} ({gates:,} gates recomputed of {len(mv) * 3072:,} at moving positions) | {wf:,} | {wd:,} | {ws:,}")
    x_prev, G_prev, pile_prev = x, G, kept.copy()
    if newT == T: break
    T = newT
print(f"\nkept piles equal the full recompute at every wave: {'YES' if all_equal else 'NO'}; in random line order: {'YES' if all_perm else 'NO'}; "
      f"gates recomputed only where the pile moved give the true bend: {'YES' if gate_ok_all else 'NO'}")
print(f"work over the whole run, fc.0:  pour whole (moving positions) {tot_full:,}   pour d {tot_delta:,} = {tot_delta:,}/{tot_full:,} ({100 * tot_delta // tot_full}%)"
      f"   shuffle d {tot_sh:,} = {tot_sh:,}/{tot_full:,} ({100 * tot_sh // tot_full}%)   every position every wave {tot_every:,}")
ow = int((L.bc(x[:P - 1]) @ rowph).sum())
print(f"the one-word machine pours each of positions 0..{P - 2} once through fc.0: {ow:,};  pour d over the run = {tot_delta:,}/{ow:,} ({100 * tot_delta // ow}%) of it")
