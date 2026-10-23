"""AUDIT 4: THE TOP READ'S UNCOUNTED COMPARES PRICED IN PHOTONS, AND ITS READY PILES' WRITING BOTH WAYS.
The counting-convention audit (the session's; its words, not his). Written for this audit: no import of, or code from,
base2_groups.py, qwen_groups.py, groups/, groups2/.

THE QUESTION. The top read (topg's (ii), g = 16) sets words aside after every place by comparing each word still in
against a threshold; its records keep these compares as a count (GPT-2 884,171 a position, Qwen 1,976,296), not as
photons. The plain read keeps its own compares (the race and the top 5) out too. In the race's own picture a compare is
read from the top digit down and pours nothing, so leaving compares out of BOTH sides is consistent. The strict price
tried here: each compare a take-away of the threshold, costing the threshold's signed digits; the plain read's compares
priced the same way, one per word (a scan holding the five best). The threshold's size is taken as the 5th-best word's
doubled pile at the top (the number every set-aside threshold sits near), from the direct product of the light at the
top and the word table (GPT-2: plain_refs' light; Qwen: capture's).
Beside it: the top tables' ready piles written at their signed digits (and at 1s, to set against the builder's record):
every half-table entry of every group of 16 (groups from numpy's generator keyed [1, 99, N, 16], a random line order cut
into runs of 16, as topg.py's docstring and its make_groups rule state); every entry is read when 50,257 or 151,936
words read 12-14 places of a 32,768-entry half table (an unread entry has chance about e^-18 or less).

PREDICTION (written before this file ran):
  - My top tables' writing at 1s = the builder's pwrite: 9,529,802 (GPT-2) and 12,836,909 (Qwen) a token, exactly if all
    entries are read and my grouping is theirs; at signed digits about 1/1.33 of that (about 7.2M and 9.7M).
  - The 5th-best doubled pile: about 2^25-2^27 (GPT-2) and 2^27-2^30 after the word's shift (Qwen): about 9-10 signed
    digits. Compares priced so: GPT-2 (ii) about 8-9M a position against plain's about 0.5M (+0.2 points of plain);
    Qwen about 19-20M against about 1.5M (+0.1 points).

    python3 confirm_groups/audit_top_side.py     -> confirm_groups/audit_top_side.json/.log
"""
import os, sys, json, time, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/confirm_groups")
from audit_common import HERE, CG, rss, ones, naf, member, product, lib

def table_writing(X, g, N):
    """every half-table entry of every group of g lines, once a position: (1s, signed digits) per position"""
    order = np.random.default_rng([1, 99, N, g]).permutation(N); runs = [order[s:s + g] for s in range(0, N, g)]
    glines = np.ascontiguousarray(np.concatenate(runs), np.int32); goff = np.concatenate([[0], np.cumsum([len(u) for u in runs])]).astype(np.int64)
    G = len(runs); gmax = max(len(u) for u in runs); readh = np.ones((G, 1 << (gmax - 1)), np.uint8)
    X = np.ascontiguousarray(X, np.int32); P = len(X)
    up = np.zeros(1, np.uint32); pat = np.zeros((G, 1, 16), np.uint16); UP = np.zeros((P, 1), np.int64); cnt = np.zeros((P, 4), np.int64)
    lib().bank(P, N, 0, G, gmax, X.ctypes.data, glines.ctypes.data, goff.ctypes.data, up.ctypes.data, pat.ctypes.data,
               readh.ctypes.data, UP.ctypes.data, cnt.ctypes.data)
    return cnt[:, 2], cnt[:, 3]

def fifth_naf(PILE, shift=None):
    V = 2 * PILE if shift is None else (2 * PILE) << shift[None, :]
    v5 = np.partition(V, V.shape[1] - 5, axis=1)[:, V.shape[1] - 5]
    return naf(v5), naf(V).mean(1)

def main():
    t0 = time.time(); log_f = open(f"{CG}/audit_top_side.log", "w")
    def log(s=""): print(s, flush=True); log_f.write(s + "\n"); log_f.flush()
    out = {}
    # GPT-2
    REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz"); X = REF["x16"].astype(np.int64); NP = len(X)
    T = np.load(f"{HERE}/groups2/machine/gpt2_top_refs.npz")
    w1, wn = table_writing(X, 16, 768)
    Wh = np.ascontiguousarray(member(f"{HERE}/base2_gpt2.npz", "head.w")); NW = Wh.shape[1]
    n5 = np.zeros(NP, np.int64); nm = []
    for p0 in range(0, NP, 48):
        a, b = fifth_naf(product(X[p0:p0 + 48], Wh)); n5[p0:p0 + 48] = a; nm += list(b)
    cmp_ii = T["ii_cmp"].astype(np.int64)
    out["gpt2"] = dict(npos=NP, pwrite_ones=int(w1.sum()), pwrite_naf=int(wn.sum()), theirs_pwrite=int(T["pwrite"].sum()),
                       cmp_ii=int(cmp_ii.sum()), cmp_plain=NP * NW, cmp_ii_photons=int((cmp_ii * n5).sum()), cmp_plain_photons=int((NW * n5).sum()),
                       fifth_naf_mean_x1000=int(1000 * n5.sum() // NP), all_words_naf_mean_x1000=int(1000 * sum(nm) // NP))
    log(f"GPT-2 top ({NP} positions): ready piles written: 1s {w1.sum() / NP:,.0f} (the builder's {T['pwrite'].sum() / NP:,.0f}; per position identical "
        f"{int((w1 == T['pwrite']).sum())}/{NP}), signed digits {wn.sum() / NP:,.0f}")
    log(f"  compares: (ii) {cmp_ii.sum() / NP:,.0f} a position, plain {NW:,}; a compare's price (the 5th-best doubled pile's signed digits) "
        f"mean {n5.sum() / NP:.2f} (all words' mean {sum(nm) / NP:.2f}); priced: (ii) {(cmp_ii * n5).sum() / NP:,.0f}, plain {(NW * n5).sum() / NP:,.0f} a position")
    del Wh
    # Qwen
    LIGHT = f"{HERE}/qwen06b/light"; Xq = np.load(f"{LIGHT}/top_light.npy").astype(np.int64); NQ = len(Xq)
    Tq = np.load(f"{HERE}/groups2/machine/qwen_top_machine.npz")
    w1, wn = table_writing(Xq, 16, 1024)
    QNPZ = f"{HERE}/qwen/qwen_base2.npz"; Wq = np.asarray(member(QNPZ, "head.w")); He = np.asarray(member(QNPZ, "head.e")).astype(np.int64)
    KW = He - He.min(); NWq = Wq.shape[1]
    n5 = np.zeros(NQ, np.int64); nm = []
    for p0 in range(0, NQ, 32):
        a, b = fifth_naf(product(Xq[p0:p0 + 32], Wq), KW); n5[p0:p0 + 32] = a; nm += list(b)
    cmp_ii = Tq["ii_cmp"].astype(np.int64)
    out["qwen"] = dict(npos=NQ, pwrite_ones=int(w1.sum()), pwrite_naf=int(wn.sum()), theirs_pwrite=int(Tq["pwrite"].sum()),
                       cmp_ii=int(cmp_ii.sum()), cmp_plain=NQ * NWq, cmp_ii_photons=int((cmp_ii * n5).sum()), cmp_plain_photons=int((NWq * n5).sum()),
                       fifth_naf_mean_x1000=int(1000 * n5.sum() // NQ), all_words_naf_mean_x1000=int(1000 * sum(nm) // NQ))
    log(f"Qwen top ({NQ} positions): ready piles written: 1s {w1.sum() / NQ:,.0f} (the builder's {Tq['pwrite'].sum() / NQ:,.0f}; per position identical "
        f"{int((w1 == Tq['pwrite']).sum())}/{NQ}), signed digits {wn.sum() / NQ:,.0f}")
    log(f"  compares: (ii) {cmp_ii.sum() / NQ:,.0f} a position, plain {NWq:,}; a compare's price (the 5th-best doubled pile at its shift, signed digits) "
        f"mean {n5.sum() / NQ:.2f} (all words' mean {sum(nm) / NQ:.2f}); priced: (ii) {(cmp_ii * n5).sum() / NQ:,.0f}, plain {(NWq * n5).sum() / NQ:,.0f} a position")
    json.dump(out, open(f"{CG}/audit_top_side.json", "w"), indent=0)
    log(f"({time.time() - t0:.0f}s, max memory {rss()} MB)")

if __name__ == "__main__":
    main()
