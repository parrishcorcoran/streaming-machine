"""AUDIT 5: THE TABLE READS THE GROUP BANKS MAKE A POSITION (plate only; the session's audit, its words).
Every up/hole read first looks up which ready pile the neuron's pattern names: one table read per (neuron, group,
up/hole place). Table reads cost no photons on either side (the bend's shares, the gains, the arrival's share table are
read free in plain and in the group machine alike), and plain reads its own plate's lit bits free too. This counts both,
to show their size beside the photons: the group banks' lookups a position, and plain's lit weight bits read a position
(each lit 1 of a weight is found once for every lit line it meets... counted here as the plate's 1s, once a position).
Rule from the builder's docstrings: place b of a plate is signed digits when its lit signed digits L_b < G x M_seg.
Written for this audit; no code from the builder's files.

PREDICTION (written before this file ran): GPT-2 about 100M lookups a position (12 layers x about 8.5M), against the
plates' 1s about 66M (85M weights x about 0.78... each weight about 7-8 1s: about 600M); Qwen about 480M lookups against
about 3.1-3.5 billion weight 1s. So the lookups are a few percent to a sixth of the plate bits plain finds; neither
side pays photons for them.

    python3 confirm_groups/audit_lookups.py     -> confirm_groups/audit_lookups.log
"""
import sys, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/confirm_groups")
from audit_common import HERE, CG, ones, member, naf_digits_lit

def count(w, segs, g=12):
    N = w.shape[0]; G = -(-N // g); look = 0
    for name, a, b in segs:
        Lb = naf_digits_lit(w[:, a:b]); U = G * (b - a); nup = sum(1 for q in range(16) if not Lb[q] < U)
        look += (b - a) * G * nup
    return look, int(ones(w).sum())

def main():
    f = open(f"{CG}/audit_lookups.log", "w")
    def log(s=""): print(s, flush=True); f.write(s + "\n"); f.flush()
    tl = tb = 0
    for l in range(12):
        for kind in ("qkv", "proj", "fc", "mproj"):
            w = np.asarray(member(f"{HERE}/base2_gpt2.npz", f"{kind}.{l}.w")).astype(np.int64)
            a, b = count(w, [(kind, 0, w.shape[1])]); tl += a; tb += b
    log(f"GPT-2 banks: ready-pile lookups {tl:,} a position; the plates' 1s {tb:,}")
    QSEG = {"qkv": [("q", 0, 2048), ("k", 2048, 3072), ("v", 3072, 4096)], "o": [("o", 0, 1024)],
            "gu": [("gate", 0, 3072), ("up", 3072, 6144)], "down": [("down", 0, 1024)]}
    tl = tb = 0
    for l in range(28):
        for slab, segs in QSEG.items():
            w = np.asarray(member(f"{HERE}/qwen/qwen_base2.npz", f"{slab}.{l}.w")).astype(np.int64)
            a, b = count(w, segs); tl += a; tb += b
    log(f"Qwen plates: ready-pile lookups {tl:,} a position; the plates' 1s {tb:,}")

if __name__ == "__main__":
    main()
