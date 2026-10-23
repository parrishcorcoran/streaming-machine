"""AUDIT REPORT: THE COUNTING CONVENTIONS OF base2_groups.py (19.40%) AND qwen_groups.py (18.94%), ONE STEP AT A TIME.
The session's audit (its words, not his). Reads only the audit's own records (audit_gpt2_plain.json,
audit_bank_both_ways.json, audit_bank_both_ways_qwenall.json, audit_qwen_parts.json, audit_top_side.json) and, for the
delivered totals being audited, the builder's records (groups2/machine/gpt2_refs.json, gpt2_top_refs.npz,
qwen_top_machine.npz). Whole numbers throughout; shares are printed from whole-number quotients.

THE LADDER (each step adds one thing to BOTH sides, or changes the base for both):
  1  delivered, as the builder counts it
  2  everything left out on both sides counted on both sides (Qwen: water levels, head water, turns, bends, divides,
     the arrival's shares' total and divides, stream adds; GPT-2 counts these already)
  3  nothing stored: GPT-2's layer 0 met by the group form at every position (no stored word / position rows, no
     stored stream at position 0); Qwen has no storage
  4  every compare priced as a take-away of its threshold (the 5th-best doubled pile's signed digits), on both sides:
     the top read's set-aside compares against the plain read's one compare a word
  5  ONE DIGIT FORM ON BOTH SIDES: plain given the same rewrites at the same prices as the group machine (plate in
     signed digits free, light written in signed digits at its 1s, every meeting lit digit x lit digit)
Beside them: the writing priced at its 1s AND its signed digits; and the other single digit form, nothing rewritten
on either side (every add at its 1s; the top read's part of this one is an estimate, labelled).

PREDICTION (written before this file ran; from the four audits' logs): GPT-2 19.40 -> 19.40 -> 19.75 -> 20.00 ->
33.7%; Qwen 18.94 -> 19.39 -> 19.39 -> 19.51 -> 32.7%. All-binary about 23% (GPT-2) and 25% (Qwen).

    python3 confirm_groups/audit_report.py     -> confirm_groups/audit_report.log
"""
import json, numpy as np, sys
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/confirm_groups")
from audit_common import HERE, CG, ones, naf
def pct(a, b): v = (200000 * a + b) // (2 * b); return f"{v // 1000}.{v % 1000:03d}%"

def main():
    log_f = open(f"{CG}/audit_report.log", "w")
    def log(s=""): print(s, flush=True); log_f.write(s + "\n"); log_f.flush()
    A1 = json.load(open(f"{CG}/audit_gpt2_plain.json")); A2 = json.load(open(f"{CG}/audit_bank_both_ways.json"))
    A2q = json.load(open(f"{CG}/audit_bank_both_ways_qwenall.json")); A3 = json.load(open(f"{CG}/audit_qwen_parts.json"))
    A4 = json.load(open(f"{CG}/audit_top_side.json"))
    # ===================== GPT-2 =====================
    n = A1["npos"]; C1 = A1["counts"]; P1 = A1["plain_total"]; SD = A1["sd_total"]
    J = json.load(open(f"{HERE}/groups2/machine/gpt2_refs.json"))["counts"]; T = np.load(f"{HERE}/groups2/machine/gpt2_top_refs.npz")
    def S(pred): return sum(v for k, v in J.items() if pred(k.split("|", 3)))
    # the delivered total, as the builder's report forms it (fc: the whole form; proj.0: the whole light, rule 3)
    deliv = S(lambda k: k[0] == "fr") + S(lambda k: k[0] == "fcw")
    grp_proj0 = sum(J.get(f"fr|L0|proj|{it}", 0) for it in ("light written in signed digits", "half table (by the flip)", "ready piles written in signed digits",
                                                           "reads, up/hole places", "reads, signed-digit places", "X (the light summed)", "X taken from each neuron"))
    adj = J["side|L0|proj|storage: the whole light in the group form: group work"] - J["side|L0|proj|storage: the stored form (rows, split)"] - grp_proj0
    top = int(T["ii_total"].sum() + T["pwrite"].sum() + T["lwrite"].sum())
    M0 = deliv + adj + top
    fcdbb = S(lambda k: k[0] == "plain" and k[2] == "fc" and k[3].split(" (")[0] in ("divides", "biases", "bends"))
    banks = A2["totals"]["gpt2 all 48"]
    M_nostore = banks["F|delivered"] + fcdbb + C1["sd|arrival"] + C1["plain|rest"] + top
    g4 = A4["gpt2"]; cii, cpl = g4["cmp_ii_photons"], g4["cmp_plain_photons"]
    log("THE COUNTING-CONVENTION AUDIT: THE LADDER (a token; each step on both sides)")
    log("")
    log(f"GPT-2 (648 refs positions; plain {P1 / n:,.0f} rebuilt to the photon; the same pairs in signed digits {SD / n:,.0f} = {pct(SD, P1)})")
    rows = [("1 delivered (the builder's count)", M0, P1),
            ("2 + left out on both sides (GPT-2 counts them already)", M0, P1),
            ("3 + nothing stored: layer 0 met at every position", M_nostore, P1),
            ("4 + compares priced on both sides", M_nostore + cii, P1 + cpl),
            ("5 + one digit form: plain in signed digits", M_nostore + cii, SD + cpl)]
    for name, a, b in rows: log(f"    {name:58s} {a / n:15,.0f} / {b / n:15,.0f} = {pct(a, b)}")
    log(f"    (step 3's storage is worth {(M_nostore - M0) / n:,.0f} a token: layer 0's stored rows and position 0's stored stream; its building is not counted)")
    wb = banks["pwrite_naf"] + g4["pwrite_naf"]
    log(f"  beside step 5: the writing priced at 1s AND signed digits (+{wb / n:,.0f}): {pct(M_nostore + cii + wb, SD + cpl)}; at the digits placed only "
        f"(-{(banks['pwrite_ones'] - banks['pwrite_naf'] + g4['pwrite_ones'] - g4['pwrite_naf']) / n:,.0f}): "
        f"{pct(M_nostore + cii - (banks['pwrite_ones'] - banks['pwrite_naf']) - (g4['pwrite_ones'] - g4['pwrite_naf']), SD + cpl)}")
    # the other single digit form: nothing rewritten anywhere (the top read estimated from its tables' own 1s / signed-digit ratio)
    REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz"); xt = REF["x16"].astype(np.int64); lr_num, lr_den = int(ones(xt).sum()), int(naf(xt).sum())
    tr, tt = int(T["ii_reads"].sum()), int(T["table"].sum())
    top_bin = int(T["ii_total"].sum()) - tr - tt + tr * g4["pwrite_ones"] // g4["pwrite_naf"] + tt * lr_num // lr_den
    M_bin = banks["F|all_ones"] + fcdbb + C1["plain|arrival"] + C1["plain|rest"] + top_bin
    log(f"  the other single form, nothing rewritten on either side (every add at its 1s; top read ESTIMATED {top_bin / n:,.0f}): "
        f"{M_bin / n:,.0f} + compares / plain + compares = {pct(M_bin + cii, P1 + cpl)}")
    log(f"  the 48 banks alone (layer 0 whole light): delivered {pct(banks['F|delivered'], banks['plain'])} of their plain at 1s; {pct(banks['F|delivered'], banks['plain_sd'])} "
        f"of their plain in signed digits; nothing rewritten {pct(banks['F|all_ones'], banks['plain'])} of their plain at 1s")
    log("")
    # ===================== Qwen =====================
    nq = A3["npos"]; P1q = A3["plain_total"]; SDq = A3["sd_total"]; Z = sum(A3["left_out"].values())
    Tq = np.load(f"{HERE}/groups2/machine/qwen_top_machine.npz"); topq = int(Tq["ii_total"].sum() + Tq["pwrite"].sum() + Tq["lwrite"].sum())
    pq = A2q["totals"]["qwen all"]; arr_sd = A3["sd"]["arrival"]
    M0q = pq["F|delivered"] + topq + arr_sd
    q4 = A4["qwen"]; ciq, cplq = q4["cmp_ii_photons"], q4["cmp_plain_photons"]
    log(f"QWEN3-0.6B (613 positions; plain {P1q / nq:,.0f} rebuilt to the photon; the same pairs in signed digits {SDq / nq:,.0f} = {pct(SDq, P1q)})")
    rows = [("1 delivered (the builder's count; plates rebuilt here)", M0q, P1q),
            ("2 + left out on both sides (water, turns, bends, divides...)", M0q + Z, P1q + Z),
            ("3 + nothing stored (none in this machine)", M0q + Z, P1q + Z),
            ("4 + compares priced on both sides", M0q + Z + ciq, P1q + Z + cplq),
            ("5 + one digit form: plain in signed digits", M0q + Z + ciq, SDq + Z + cplq)]
    for name, a, b in rows: log(f"    {name:58s} {a / nq:15,.0f} / {b / nq:15,.0f} = {pct(a, b)}")
    log(f"    (step 2's parts: {Z / nq:,.0f} a token: " + ", ".join(f"{k} {v / nq:,.0f}" for k, v in A3["left_out"].items()) + ")")
    wbq = pq["pwrite_naf"] + q4["pwrite_naf"]
    log(f"  beside step 5: the writing priced at 1s AND signed digits (+{wbq / nq:,.0f}): {pct(M0q + Z + ciq + wbq, SDq + Z + cplq)}; at the digits placed only: "
        f"{pct(M0q + Z + ciq - (pq['pwrite_ones'] - pq['pwrite_naf']) - (q4['pwrite_ones'] - q4['pwrite_naf']), SDq + Z + cplq)}")
    xq = np.load(f"{HERE}/qwen06b/light/top_light.npy").astype(np.int64); lrq_num, lrq_den = int(ones(xq).sum()), int(naf(xq).sum())
    trq, ttq = int(Tq["ii_reads"].sum()), int(Tq["table"].sum())
    topq_bin = int(Tq["ii_total"].sum()) - trq - ttq + trq * q4["pwrite_ones"] // q4["pwrite_naf"] + ttq * lrq_num // lrq_den
    arr_plain = A3["plain"]["arrival"]
    M_binq = pq["F|all_ones"] + topq_bin + arr_plain
    log(f"  the other single form, nothing rewritten on either side (top read ESTIMATED {topq_bin / nq:,.0f}): {pct(M_binq + Z + ciq, P1q + Z + cplq)}")
    log(f"  the 196 plates alone: delivered {pct(pq['F|delivered'], pq['plain'])} of their plain at 1s; {pct(pq['F|delivered'], pq['plain_sd'])} in signed digits; "
        f"nothing rewritten {pct(pq['F|all_ones'], pq['plain'])}")
    log(f"  check: the plates' delivered total {pq['F|delivered'] / nq:,.0f} a token and the whole {M0q / nq:,.0f} (the builder's report: 2,828.9M and 3,105,280,101)")

if __name__ == "__main__":
    main()
