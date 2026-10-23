"""CHECKER: the checker's own counts (count_gpt2_banks.json, topg_own_*.json, count_qwen.json, qwen_ref.json's arrival
counts) set beside the builder's counts of the same texts (gpt2_machine_check.json, qwen_machine_check.json and
qwen_machine_check_counts.npz: base2_groups / qwen_groups as delivered), part by part and item by item.
GPT-2 aggregated as base2_groups' phase_report aggregates (rule 3: proj.0 on the whole light; rule 4: the knife at no
fc plate); Qwen as qwen_groups' phase_report.
"""
import json, numpy as np
OUT = "/Users/abundancemachine/streaming-machine/groups2/check_machines"
LOGF = open(f"{OUT}/compare_counts.log", "w")
def log(s=""): print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
def cmp(name, a, b): return f"{name:44s} builder {a:>16,}  checker {b:>16,}  {'EQUAL' if a == b else f'DIFFER by {b - a:+,}'}"
# ================= GPT-2 =================
J = json.load(open(f"{OUT}/gpt2_machine_check.json")); C = J["runs"]["5-12"]["counts"]; n = J["runs"]["5-12"]["npos"]
K = {tuple(k.split("|", 3)): v for k, v in C.items()}
def S(pred): return sum(v for k, v in K.items() if pred(k))
GK = ["light written in signed digits", "half table (by the flip)", "ready piles written in signed digits", "reads, up/hole places",
      "reads, signed-digit places", "X (the light summed)", "X taken from each neuron"]
def grp(who, lay, part): return sum(C.get(f"{who}|{lay}|{part}|{g}", 0) for g in GK)
b_plain = {p: S(lambda k, p=p: k[0] == "plain" and k[2] == p) for p in ("qkv", "proj", "fc", "mproj", "arrival", "rest")}
b_mine = {p: S(lambda k, p=p: k[0] == "fr" and k[2] == p) for p in b_plain}
b_mine["fc"] += S(lambda k: k[0] == "fcw")
b_mine["proj"] += C["side|L0|proj|storage: the whole light in the group form: group work"] - C["side|L0|proj|storage: the stored form (rows, split)"] - grp("fr", "L0", "proj")
T = J["top"]; nt = 43
b_plain["top"] = sum(T["plain"][:nt]); b_mine["top"] = sum(T["ii_total"][:nt]) + sum(T["pwrite"][:nt]) + sum(T["lwrite"][:nt])
M_ = json.load(open(f"{OUT}/count_gpt2_banks.json")); m_mine, m_plain, items = M_["mine"], M_["plain"], M_["items"]
TO = json.load(open(f"{OUT}/topg_own_gpt2.json"))["recs"]; assert len(TO) == nt
m_plain["top"] = sum(r["plain"] for r in TO); m_mine["top"] = sum(r["ii_total"] + r["pwrite"] + r["lwrite"] for r in TO)
log(f"GPT-2, refs texts 5 and 12 ({n} positions): THE WORK BY PART (photons, the two texts together)")
for p in ("qkv", "proj", "fc", "mproj", "arrival", "rest", "top"):
    log("  " + cmp(f"{p}: plain", b_plain[p], m_plain[p])); log("  " + cmp(f"{p}: the group machine", b_mine[p], m_mine[p]))
BP, BM, MP, MM = sum(b_plain.values()), sum(b_mine.values()), sum(m_plain.values()), sum(m_mine.values())
log("  " + cmp("THE WHOLE MACHINE: plain", BP, MP)); log("  " + cmp("THE WHOLE MACHINE: the group machine", BM, MM))
log(f"  share of plain: builder {100 * BM / BP:.4f}%, checker {100 * MM / MP:.4f}%  (a token: plain {MP / n:,.0f}, machine {MM / n:,.0f})")
pw_b = S(lambda k: k[0] == "fr" and k[3] == "ready piles written in signed digits") + sum(T["pwrite"][:nt])
log(f"  without the ready piles written: builder {100 * (BM - pw_b) / BP:.4f}%")
# item level, the banks
log("  item by item (all 12 layers; proj: layers 1-11 plus layer 0's whole light; qkv: layer 0's carry included):")
lab = dict(light="light written in signed digits", table="half table (by the flip)", pwrite="ready piles written in signed digits",
           reads="reads, up/hole places", sdreads="reads, signed-digit places", xsum="X (the light summed)", xoff="X taken from each neuron")
for part in ("qkv", "fc", "mproj"):
    for it, L in lab.items():
        who = ("fcw",) if (part == "fc" and it in ("reads", "sdreads")) else ("fr",)
        bv = S(lambda k, L=L, part=part, who=who: k[0] in who and k[2] == part and k[3] == L)
        log("    " + cmp(f"{part} {it}", bv, items.get(f"{part}|{it}", 0)))
bv = S(lambda k: k[0] == "fr" and k[2] == "proj" and k[1] != "L0" and k[3] in GK) + C["side|L0|proj|storage: the whole light in the group form: group work"]
mv = sum(items.get(f"proj|{it}", 0) for it in lab)
log("    " + cmp("proj group work (L1-11 + L0 whole light)", bv, mv))
log("    " + cmp("qkv.0 stored rows", C.get("fr|L0|qkv|stored rows read and added", 0), items["qkv.0|stored rows"] + items["qkv.0|stored rows (position 0)"]))
log("    " + cmp("qkv.0 split", C.get("fr|L0|qkv|split (stored rounded rows taken off)", 0), items["qkv.0|split"]))
log("    " + cmp("fc.0 stored rows", C.get("fr|L0|fc|stored rows read and added", 0), items["fc.0|stored rows (ones of the sum, as counted)"]))
log(f"    (fc.0's stored rows counted as each row's 1s instead of the 1s of their sum: {items['fc.0|stored rows (each row' + chr(39) + 's ones)']:,} against "
    f"{items['fc.0|stored rows (ones of the sum, as counted)']:,}; {100 * (items['fc.0|stored rows (each row' + chr(39) + 's ones)'] - items['fc.0|stored rows (ones of the sum, as counted)']) / BP:+.4f} points of plain)")
# the top, position by position
F = ["plain", "ii_total", "table", "ii_reads", "ii_light", "ii_bound", "ii_end", "ii_stand", "pwrite", "lwrite"]
bad = {f: sum(int(T[f][q] != TO[q][f]) for q in range(nt)) for f in F}
sv = sum(int([v for v in T["ii_surv"][q] if v >= 0] != TO[q]["surv"]) for q in range(nt))
log(f"  the top read, position by position (43): fields differing {bad}; survivors after each place differing at {sv} positions")
log(f"    the checker's (ii) top 5 = its plain top 5 {sum(r['top5'] == r['ptop5'] for r in TO)}/{nt}, race winner {sum(r['win'] == r['pwin'] for r in TO)}/{nt}")
# ================= Qwen =================
log("")
QJ = json.load(open(f"{OUT}/qwen_machine_check.json")); Zc = np.load(f"{OUT}/qwen_machine_check_counts.npz"); rows = Zc["rows"]
QC = {k.replace("__", "|"): Zc[k] for k in Zc.files if k != "rows"}
MYQ = json.load(open(f"{OUT}/count_qwen.json"))["res"]; RJ = json.load(open(f"{OUT}/qwen_ref.json"))
QSEG = {"qkv": ["q", "k", "v"], "o": ["o"], "gu": ["gate", "up"], "down": ["down"]}
log("QWEN3-0.6B, refs texts 5 and 12 (32 positions): THE WORK BY PART")
tot_b = tot_m = 0; diffs = 0; nitems = 0; byslab = {}
for ti, t in enumerate((5, 12)):
    a, b = rows[ti]
    for l in range(28):
        for slab, kinds in QSEG.items():
            my = MYQ[f"r{t}|{slab}.{l}"]
            for it in ("light", "table", "pwrite", "xsum", "xoff"):
                bv = int(QC[f"{slab}.{l}|g_{it}"][a:b].sum()); mv = my[it]; nitems += 1; diffs += bv != mv
                byslab.setdefault(slab, [0, 0, 0, 0]); byslab[slab][0] += bv; byslab[slab][1] += mv
            for kd in kinds:
                for it in ("reads", "sdreads", "plain"):
                    key = f"{kd}.{l}|g_{it}" if it != "plain" else f"{kd}.{l}|plain"; bv = int(QC[key][a:b].sum()); mv = my[f"{it}|{kd}"]
                    nitems += 1; diffs += bv != mv
                    if it == "plain": byslab[slab][2] += bv; byslab[slab][3] += mv
                    else: byslab[slab][0] += bv; byslab[slab][1] += mv
log(f"  every item of every slab, layer and text ({nitems} numbers): differing {diffs}")
for slab, (bv, mv, bp, mp) in byslab.items():
    log("  " + cmp(f"{slab}: plain", bp, mp)); log("  " + cmp(f"{slab}: the group machine", bv, mv))
arr_b = sum(int(QC[f"arr.{l}|{k}"][rows[0][0]:rows[1][1]].sum()) for l in range(28) for k in ("sd_scores", "sd_values", "write_q", "write_k", "write_v"))
arr_bp = sum(int(QC[f"arr.{l}|{k}"][rows[0][0]:rows[1][1]].sum()) for l in range(28) for k in ("plain_scores", "plain_values"))
arr_m = sum(v for k, v in RJ["arr"].items() if k.split("|")[0] in ("r5", "r12") and k.split("|")[2] in ("sd_scores", "sd_values", "write_q", "write_k", "write_v"))
arr_mp = sum(v for k, v in RJ["arr"].items() if k.split("|")[0] in ("r5", "r12") and k.split("|")[2] in ("plain_scores", "plain_values"))
log("  " + cmp("arrival: plain", arr_bp, arr_mp)); log("  " + cmp("arrival: signed digits", arr_b, arr_m))
QT = QJ["refs"]; QO = json.load(open(f"{OUT}/topg_own_qwen.json"))["recs"]; nq = len(QO); assert nq == 32
badq = {f: sum(int(QT[f][q] != QO[q][f]) for q in range(nq)) for f in ["plain", "ii_total", "table", "ii_reads", "ii_light", "ii_bound", "ii_end", "pwrite", "lwrite"]}
svq = sum(int([v for v in QT["ii_surv"][q] if v >= 0] != QO[q]["surv"]) for q in range(nq))
log(f"  the top read, position by position ({nq}): fields differing {badq}; survivors differing at {svq} positions")
tb = sum(QT["ii_total"][:nq]) + sum(QT["pwrite"][:nq]) + sum(QT["lwrite"][:nq]); tm = sum(r["ii_total"] + r["pwrite"] + r["lwrite"] for r in QO)
log("  " + cmp("top: plain", sum(QT["plain"][:nq]), sum(r["plain"] for r in QO))); log("  " + cmp("top: the group machine", tb, tm))
QBP = sum(v[2] for v in byslab.values()) + arr_bp + sum(QT["plain"][:nq]); QBM = sum(v[0] for v in byslab.values()) + arr_b + tb
QMP = sum(v[3] for v in byslab.values()) + arr_mp + sum(r["plain"] for r in QO); QMM = sum(v[1] for v in byslab.values()) + arr_m + tm
log("  " + cmp("THE WHOLE MACHINE: plain", QBP, QMP)); log("  " + cmp("THE WHOLE MACHINE: the group machine", QBM, QMM))
log(f"  share of plain: builder {100 * QBM / QBP:.4f}%, checker {100 * QMM / QMP:.4f}%")
