"""Each line's trajectory: the stream R (positions x 768 lines, whole numbers at 2^16) as it enters, and after each of
the 24 writes (two a layer), for the full ids+ref of all 30 texts.

Then the SAME machine again with everything taken in a random order (seed 4): the 768 lines, the 3,072 edge neurons
of every layer (a new order each layer), the 12 heads, and the 64 places inside every head (one order shared by its
q and k, another for its v). The stream must come out as the very same numbers, only in the shuffled line order.

Saves watch/traj.npz (R0..R29, each 25 x positions x 768) and watch/traj_lines_shuffled.npz (same, lines in the
shuffled order, plus the order itself as 'lines').
    python3 watch/collect.py
"""
import os, sys, json, time, numpy as np
ROOT = "/Users/abundancemachine/streaming-machine/"
os.chdir(ROOT)
_src = open(ROOT + "base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
_src = _src.replace(", torch", "").replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer", "").replace(' + ["head"]', "")
_argv = sys.argv; sys.argv = ["x"]; exec(_src); sys.argv = _argv
refs = json.load(open(ROOT + "refs_many.json"))
texts = [r["ids"] + r["ref"] for r in refs]

def run(ids, wte, wpe):
    P = len(ids); R = wte[ids] + wpe[:P]; snaps = [R]
    for l in range(12):
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}"); snaps.append(R)
        R = R + plain_bank(edge(gauged_bank(R, f"fc.{l}")), f"mproj.{l}"); snaps.append(R)
    return np.stack(snaps)

wte, wpe = M["wte"], M["wpe"]
t0 = time.time(); real = [run(ids, wte, wpe) for ids in texts]
big = max(int(np.abs(x).max()) for x in real)
print(f"real order: {len(real)} texts, {sum(x.shape[1] for x in real)} positions, 25 snapshots each; "
      f"largest |value| {big} (top place {big.bit_length() - 1}); {time.time() - t0:.0f}s")
assert big < 2 ** 31
np.savez(ROOT + "watch/traj.npz", **{f"R{i}": x.astype(np.int32) for i, x in enumerate(real)})

rng = np.random.default_rng(4)
pl = rng.permutation(768); eta = rng.permutation(12)
sig = [rng.permutation(64) for _ in range(12)]; tau = [rng.permutation(64) for _ in range(12)]
pa = np.concatenate([eta[h] * 64 + tau[h] for h in range(12)])            # attention's 768 out, in the new order
pqk = np.concatenate([eta[h] * 64 + sig[h] for h in range(12)])
pq = np.concatenate([pqk, 768 + pqk, 1536 + pa])
for l in range(12):
    pn = rng.permutation(3072)
    w, e, b = W[f"qkv.{l}"];   W[f"qkv.{l}"] = (w[pl][:, pq].copy(), e[pq], b[pq])
    w, e, b = W[f"proj.{l}"];  W[f"proj.{l}"] = (w[pa][:, pl].copy(), e[pl], b[pl])
    w, e, b = W[f"fc.{l}"];    W[f"fc.{l}"] = (w[pl][:, pn].copy(), e[pn], b[pn])
    w, e, b = W[f"mproj.{l}"]; W[f"mproj.{l}"] = (w[pn][:, pl].copy(), e[pl], b[pl])
t0 = time.time(); shuf = [run(ids, wte[:, pl], wpe[:, pl]) for ids in texts]
diff = sum(int(np.count_nonzero(s != r[:, :, pl])) for s, r in zip(shuf, real))
tot = sum(r.size for r in real)
print(f"everything in a random order (lines, edge neurons, heads, places in a head): "
      f"{diff} of {tot} stream values differ from the real run's; {time.time() - t0:.0f}s")
np.savez(ROOT + "watch/traj_lines_shuffled.npz", lines=pl, **{f"R{i}": x.astype(np.int32) for i, x in enumerate(shuf)})
