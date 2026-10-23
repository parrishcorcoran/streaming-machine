"""CHECKER: (1) the delivered files (changed 03:41, after the queue's runs ended 03:40) against the saved runs: the
per-position counts of the checker's runs of the delivered code at texts 5, 12, 19, 20, 28 set beside the saved full runs'
per-position records at the same texts (GPT-2 top read: gpt2_top_refs.npz; Qwen: qwen_machine_counts.npz and
qwen_top_machine.npz); (2) the reports' totals re-added from the saved runs by the checker (GPT-2: gpt2_refs.json and
gpt2_top_refs.npz, with rule 3 and rule 4 as applied; Qwen: qwen_machine_counts.npz and qwen_top_machine.npz).
"""
import json, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; MD = f"{HERE}/groups2/machine"; OUT = f"{HERE}/groups2/check_machines"
LOGF = open(f"{OUT}/saved_vs_delivered.log", "w")
def log(s=""): print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
# ---- GPT-2 top read, per position ----
J = json.load(open(f"{OUT}/gpt2_machine_check.json")); T = J["top"]
REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz"); npos = REF["npos"]; offs = np.concatenate([[0], np.cumsum(npos)])
rows = np.concatenate([np.arange(offs[t], offs[t + 1]) for t in (5, 12, 19, 20, 28)])
S_ = np.load(f"{MD}/gpt2_top_refs.npz"); assert np.array_equal(S_["part"], np.arange(648))
d = {k: int((S_[k][rows] != np.array(T[k])).sum()) for k in ("plain", "ii_total", "pwrite", "lwrite", "table", "ii_reads", "ii_bound", "ii_end", "ii_stand")}
log(f"GPT-2 top read: delivered code at texts 5, 12, 19, 20, 28 ({len(rows)} positions) against the saved run's records there: positions differing {d}")
# ---- GPT-2 totals re-added ----
G = json.load(open(f"{MD}/gpt2_refs.json")); C = G["counts"]; n = G["npos"]
K = {tuple(k.split("|", 3)): v for k, v in C.items()}
def S(pred): return sum(v for k, v in K.items() if pred(k))
GK = ["light written in signed digits", "half table (by the flip)", "ready piles written in signed digits", "reads, up/hole places",
      "reads, signed-digit places", "X (the light summed)", "X taken from each neuron"]
def grp(who, lay, part): return sum(C.get(f"{who}|{lay}|{part}|{g}", 0) for g in GK)
plain = {p: S(lambda k, p=p: k[0] == "plain" and k[2] == p) for p in ("qkv", "proj", "fc", "mproj", "arrival", "rest")}
mach = {p: S(lambda k, p=p: k[0] == "fr" and k[2] == p) for p in plain}; mach["fc"] += S(lambda k: k[0] == "fcw")
# rule 3 re-applied by the checker
st = {"qkv": C["side|L0|qkv|storage: the stored form (rows, split, the carry's work)"] + grp("fr", "L0", "qkv"),
      "proj": C["side|L0|proj|storage: the stored form (rows, split)"] + grp("fr", "L0", "proj"),
      "fc": C["side|L0|fc|storage: the stored form (rows, split)"] + grp("fr", "L0", "fc") + C["fcw|L0|fc|reads, up/hole places"] + C["fcw|L0|fc|reads, signed-digit places"]}
wh = {p: C[f"side|L0|{p}|storage: the whole light in the group form: group work"] for p in st}
log("rule 3 re-applied (a token): " + "; ".join(f"{p}.0 stored {st[p] / n:,.0f} whole {wh[p] / n:,.0f} -> {'stored' if st[p] < wh[p] else 'whole'}" for p in st))
for p in st:
    if not st[p] < wh[p]: mach[p] += wh[p] - st[p]
kn = {f"L{l}": (S(lambda k, l=l: k[0] == "fck" and k[1] == f"L{l}"), S(lambda k, l=l: k[0] == "fcw" and k[1] == f"L{l}")) for l in range(12)}
log(f"rule 4 re-applied: the knife lower at {sum(a < b for a, b in kn.values())} of 12 fc plates")
Tp = np.load(f"{MD}/gpt2_top_refs.npz"); plain["top"] = int(Tp["plain"].sum()); mach["top"] = int((Tp["ii_total"] + Tp["pwrite"] + Tp["lwrite"]).sum())
P0, M0 = sum(plain.values()), sum(mach.values()); pw = S(lambda k: k[0] == "fr" and k[3] == "ready piles written in signed digits") + int(Tp["pwrite"].sum())
log(f"GPT-2 whole machine re-added: plain {P0 / n:,.2f} a token, machine {M0 / n:,.2f} a token = {100 * M0 / P0:.4f}%; without the ready piles written "
    f"{(M0 - pw) / n:,.2f} = {100 * (M0 - pw) / P0:.4f}%; against base2_fringes2's 1,694,214,555: {100 * (M0 / n) / 1694214555:.2f}%")
log("   by part (a token): " + "; ".join(f"{p} {mach[p] / n:,.0f} ({100 * mach[p] / plain[p]:.2f}%)" for p in mach))
# ---- Qwen per position and totals ----
QM = np.load(f"{MD}/qwen_machine_counts.npz"); QMc = {k.replace("__", "|"): QM[k] for k in QM.files}
MT = json.load(open(f"{HERE}/qwen06b/light/meta.json")); mrows = MT["rows"]
Zc = np.load(f"{OUT}/qwen_machine_check_counts.npz"); myrows = Zc["rows"]; MYC = {k.replace("__", "|"): Zc[k] for k in Zc.files if k != "rows"}
nd = nk = 0
for j, t in enumerate((5, 12, 19, 20, 28)):
    a, b = mrows[t]; c_, d_ = myrows[j]
    for k, v in MYC.items():
        nk += 1; nd += not np.array_equal(QMc[k][a:b], v[c_:d_])
log(f"Qwen: delivered code's per-position count arrays at texts 5, 12, 19, 20, 28 against the saved run's at the same texts: {nd} of {nk} arrays differ")
QJ = json.load(open(f"{OUT}/qwen_machine_check.json"))["refs"]; QT = np.load(f"{MD}/qwen_top_machine.npz")
qrows = np.concatenate([np.arange(*mrows[t]) for t in (5, 12, 19, 20, 28)])
dq = {k: int((QT[k][qrows] != np.array(QJ[k])).sum()) for k in ("plain", "ii_total", "pwrite", "lwrite", "table", "ii_reads", "ii_bound", "ii_end")}
log(f"Qwen top read: delivered code against the saved run at those {len(qrows)} positions: positions differing {dq}")
NP = MT["npos"]; L = 28; KN = ["q", "k", "v", "o", "gate", "up", "down"]; SN = ["qkv", "o", "gu", "down"]
def Sq(k): return int(QMc[k].sum()) if k in QMc else 0
pp = sum(Sq(f"{k}.{l}|plain") for k in KN for l in range(L)); tp = int(QT["plain"].sum())
ap = sum(Sq(f"arr.{l}|{k}") for l in range(L) for k in ("plain_scores", "plain_values"))
pn = sum(Sq(f"{s}.{l}|g_{it}") for s in SN for l in range(L) for it in ("light", "table", "pwrite", "xsum", "xoff")) + \
     sum(Sq(f"{k}.{l}|g_reads") + Sq(f"{k}.{l}|g_sdreads") for k in KN for l in range(L))
tn = int((QT["ii_total"] + QT["pwrite"] + QT["lwrite"]).sum()); an = sum(Sq(f"arr.{l}|{k}") for l in range(L) for k in ("sd_scores", "sd_values", "write_q", "write_k", "write_v"))
QP0, QM0 = pp + tp + ap, pn + tn + an; qpw = sum(Sq(f"{s}.{l}|g_pwrite") for s in SN for l in range(L)) + int(QT["pwrite"].sum())
log(f"Qwen whole machine re-added: plain {QP0 / NP:,.2f} a token, machine {QM0 / NP:,.2f} = {100 * QM0 / QP0:.4f}%; plates {100 * pn / pp:.2f}%, top {100 * tn / tp:.2f}%, "
    f"arrival {an / NP:,.0f}; without the ready piles written {100 * (QM0 - qpw) / QP0:.4f}%; against qwen_fringes2's 5,602,023,037: {100 * (QM0 / NP) / 5602023037:.2f}%")
