"""PHASE C, STEP 4: THE SHUFFLE CONTROL. One plate's sheets given to the wrong neurons, same values.
Plate fc.6 (set before the run: the neurons-in plate of the middle layer). 64 of its 3,072 neurons, drawn at random
(seed 6), each given the sheets of the next one drawn (one cycle). Every value of the plate stays in the plate and every
line keeps exactly its lit places: only which neuron holds which column changes. The machine runs from the files on
the 30 refs texts (648 positions) up to that bank; every pile before it is checked too.
PREDICTION: every pile before fc.6 identical; at fc.6 the piles differ from base2.py's at exactly the 64 moved neurons
(at nearly every position) and are identical at the other 3,008; the bank's lit-pixel meetings are identical to the
real plate's (C_run_refs.py's count), so the count cannot see the pairing and the piles can.
    python3 recording/C_controls.py   -> recording/C_controls.log
"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = "/Users/abundancemachine/streaming-machine"
LOG = open(f"{HERE}/recording/C_controls.log", "w"); t0 = time.time()
def log(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
import C_machine as K
REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz"); NPOS = int(REF["npos"].sum())
BANK = "fc.6"; N = K.MAN["plates"][BANK]["neurons"]
moved = np.sort(np.random.default_rng(6).choice(N, 64, replace=False))
src = np.arange(N); src[moved] = np.roll(moved, -1)                 # neuron moved[i] gets the sheets of moved[i + 1]
bad = {}
def check(k, v):
    ref = REF[k.replace(" ", "_")]
    if not np.array_equal(ref, v): bad[k] = int((ref != v).any(1).sum())
C = K.Count()
pile = K.run(K.TEXTS, C=C, check=check, plate_opts={BANK: dict(neuron_src=src)}, stop_at=BANK, log=log)
ref = REF["fc_pile_6"]
before = [k for k in bad if k != "fc pile 6"]
diff = pile != ref; broke = np.flatnonzero(diff.any(0))
log(f"\nSHUFFLE CONTROL at {BANK}: {len(moved)} neurons given another neuron's sheets (same values)")
log(f"  piles before {BANK} that differ from base2.py's: {len(before)} {before}")
log(f"  neurons whose pile differs at some position: {len(broke)}; exactly the moved ones: {np.array_equal(broke, moved)}")
log(f"  at the moved neurons, positions differing: {int(diff[:, moved].sum()):,} of {NPOS * len(moved):,}; "
    f"at the other {N - len(moved)} neurons: {int(np.delete(diff, moved, axis=1).sum())}")
R = json.load(open(f"{HERE}/recording/C_logs/refs_counts.json"))["total"]
for w in ("rec", "plain"):
    k = f"bank|{BANK}|{w}"
    log(f"  {BANK} {'lit-pixel meetings, signed digits' if w == 'rec' else 'meetings, plain'}: moved plate {C[k]:,}, real plate "
        f"{R[k]:,}; identical: {C[k] == R[k]}")
log(f"({time.time() - t0:.0f}s, max memory {K.rss()} MB)")
