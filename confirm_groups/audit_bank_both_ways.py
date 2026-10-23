"""AUDIT 2: THE BANKS RECOUNTED BOTH WAYS -- THE READY PILES' WRITING AT THEIR 1s AGAINST THE READS AT SIGNED DIGITS.
The counting-convention audit (the session's; its words, not his). Written for this audit: no import of, or code from,
base2_groups.py, qwen_groups.py, groups/, groups2/. The group plate is laid by the rules the builder's docstrings state
(groups2/gbank.py, uh_rebuild.py, uh_perplace.py: g = 12; numpy's generator keyed [1, slab, layer, 12], a random line
order cut into runs of 12; place b in signed digits when the plate's lit signed digits there L_b < G x M; binary
digits at the up/hole places, offset C = 2^(top up/hole place); half table by the flip; ready piles by the lowest-bit
rule) and run by confirm_groups/audit_k.c. The light: GPT-2's rebuilt from base2.py's own run (audit_gpt2_plain.py's
steps, every pile checked against plain_refs); Qwen's from capture's record (qwen06b/light/).

THE QUESTION. base2_groups.py and qwen_groups.py cost every up/hole read at the ready pile's SIGNED DIGITS and every
ready pile the plate reads WRITTEN once at its 1s (and the light written at its 1s). Is that fair, or does it flatter?
Each bank is counted under five conventions, per position, all exact piles:
  delivered    light 1s + table (signed digits of each light value added) + ready piles written (1s) + reads (signed
               digits of the ready pile) + signed-digit places (signed digits of the light value) + X + X taken off
  write_at_sd  the same with the ready piles' writing priced at the signed digits placed instead of the 1s
  write_both   the writing priced at its 1s AND its signed digits (read the 1s, place the digits)
  no_write     the confirm steps' count: reads at signed digits, nothing written (17.30% / 17.59% machines)
  all_ones     NOTHING rewritten: every add at the 1s of the binary value added (table, reads, signed places, X, X off)
and beside them plain (1s x 1s per pair) and the same pairs in signed digits (light written once at its 1s).
Two conventions are each consistent on both sides: all_ones against plain, and delivered against plain in signed
digits. The delivered figure against plain at 1s mixes the two.

RUNS: GPT-2 every bank of all 12 layers (layer 0 on the WHOLE light at every position, the form with no storage, with
positions >= 1 and position 0 kept apart); Qwen3-0.6B the four slabs of layer 14 (layers=all for all 28).

CHECKS: every pile = the reference (GPT-2: plain_refs; Qwen: the direct product, audit_k.c's prod_f); my delivered
items = the builder's records item by item at every GPT-2 bank of layers 1-11 (groups2/machine/gpt2_refs.json) and every
Qwen plate run (qwen_machine_counts.npz); layer 0's whole-light group work on positions >= 1 = the builder's side
counts ("storage: the whole light in the group form").

PREDICTION (written before this file ran):
  - Every pile exact. My delivered items equal the builder's to the photon at every bank compared (if my reading of the
    grouping rule is right; if not, within 0.1% item by item and the piles still exact). Layer 0 whole light on
    positions >= 1: qkv.0 12,187,796, proj.0 5,002,115, fc.0 14,699,339 a token (the builder's side counts).
  - A ready pile of 12 lines holds about 1.5 x as many 1s as signed digits. Entries are read about 16 times a position
    at qkv and fc and about 5 times at proj and mproj. So: write_at_sd lowers a bank 2-4%; write_both raises it 4-9%;
    all_ones costs about 1.35-1.5 x delivered.
  - GPT-2 banks (all 48): delivered about 22.5% of plain (the builder's qkv/proj/fc/mproj give 22.6% with fc's divides);
    all_ones about 28-32% of plain; delivered about 37-39% of the same pairs in signed digits. Qwen layer 14: delivered
    about 23-25% of plain, all_ones about 30-34%, delivered about 39-42% of signed-digit plain.
  - Verdict expected: pricing the writing at 1s does NOT flatter (1s >= signed digits for every number, so it is the
    dearer of the two single prices); what flatters is the BASE: reads at signed digits against plain at 1s. Within one
    digit form the group form keeps about 30% (binary) or 38% (signed digits) of its plain, not 22%.

    python3 confirm_groups/audit_bank_both_ways.py [qwen_layers=14|all] [gpt2=1|0]
    -> confirm_groups/audit_bank_both_ways.json/.log
"""
import os, sys, json, time, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/confirm_groups")
from audit_common import HERE, CG, rss, ones, naf, member, product, rshift, to16, power, Plate, forms
import audit_gpt2_plain as A1

SLABS = ["qkv", "proj", "fc", "mproj", "o", "gu", "down"]
QSEG = {"qkv": [("q", 0, 2048), ("k", 2048, 3072), ("v", 3072, 4096)], "o": [("o", 0, 1024)],
        "gu": [("gate", 0, 3072), ("up", 3072, 6144)], "down": [("down", 0, 1024)]}
QNPZ = f"{HERE}/qwen/qwen_base2.npz"
ITEMS = ("light_ones", "table_naf", "pwrite_ones", "xsum_naf", "xoff_naf", "reads_naf", "sdreads_naf")
THEIRS = dict(light_ones="light written in signed digits", table_naf="half table (by the flip)", pwrite_ones="ready piles written in signed digits",
              xsum_naf="X (the light summed)", xoff_naf="X taken from each neuron", reads_naf="reads, up/hole places", sdreads_naf="reads, signed-digit places")
FORMS = ("delivered", "write_at_sd", "write_both", "no_write", "all_ones")
def pct(a, b): v = (20000 * a + b) // (2 * b); return f"{v // 100}.{v % 100:02d}%"

def summed(c, rows=None):
    f = forms(c); out = {k: int(v.sum() if rows is None else v[rows].sum()) for k, v in c.items()}
    out.update({"F|" + k: int(v.sum() if rows is None else v[rows].sum()) for k, v in f.items()}); return out

def run_gpt2(log, REC):
    REF = np.load(f"{HERE}/recording/C_ref/plain_refs.npz"); npos = REF["npos"]
    REFS = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [r["ids"] + r["ref"] for r in REFS]
    off = np.concatenate([[0], np.cumsum(npos)]); NP = int(off[-1])
    ids = np.concatenate([np.asarray(t, np.int64) for t in TEXTS]); pos = np.concatenate([np.arange(n) for n in npos])
    r1, f0 = np.flatnonzero(pos > 0), np.flatnonzero(pos == 0)
    J = json.load(open(f"{HERE}/groups2/machine/gpt2_refs.json"))["counts"]
    WTE, WPE = A1.member(A1.NPZ, "wte"), A1.member(A1.NPZ, "wpe")
    R0 = np.asarray(WTE[ids]).astype(np.int64) + np.asarray(WPE[pos]).astype(np.int64)
    dummy = {"plain|arrival": 0, "sd|arrival": 0}; ES = A1.ES; EXP2 = A1.EXP2
    BAD = {}; MIS = []; t0 = time.time()
    for l in range(12):
        Rin = R0 if l == 0 else REF[f"stream_{l - 1}"].astype(np.int64)
        lights = {}
        w, we, b = A1.wplate(f"qkv.{l}"); x16, ex, LS = A1.gauge_log(Rin); lights["qkv"] = x16
        QKV = power(REF[f"qkv_pile_{l}"].astype(np.int64), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None], EXP2) + b
        o = np.concatenate([A1.arrival(QKV[off[t]:off[t + 1]], dummy, l) for t in range(30)]); lights["proj"] = to16(o, -ES)[0]
        x16, ex, LS = A1.gauge_log(REF[f"stream_a_{l}"].astype(np.int64)); lights["fc"] = x16
        w, we, b = A1.wplate(f"fc.{l}")
        H = power(REF[f"fc_pile_{l}"].astype(np.int64), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None], EXP2) + b
        lights["mproj"] = to16(rshift(H * A1.share_of(H), 16), -ES)[0]
        for kind in ("qkv", "proj", "fc", "mproj"):
            w, we, b = A1.wplate(f"{kind}.{l}")
            pl = Plate(np.asarray(w).astype(np.int64), [(kind, 0, w.shape[1])], 12, [1, SLABS.index(kind), l, 12])
            pile, c = pl.run(lights[kind])
            BAD[f"GPT-2 piles vs plain_refs"] = BAD.get("GPT-2 piles vs plain_refs", 0) + int((pile != REF[f"{kind}_pile_{l}"]).any(1).sum())
            if l == 0:
                REC[f"gpt2|L0|{kind}|r1"] = summed(c, r1); REC[f"gpt2|L0|{kind}|f0"] = summed(c, f0)
            else:
                s = summed(c); REC[f"gpt2|L{l}|{kind}"] = s
                for it in ITEMS:
                    who = "fcw" if (kind == "fc" and it in ("reads_naf", "sdreads_naf")) else "fr"
                    theirs = J.get(f"{who}|L{l}|{kind}|{THEIRS[it]}", 0)
                    if theirs != s[it]: MIS.append((f"L{l} {kind} {it}", s[it], theirs))
                if J.get(f"plain|L{l}|{kind}|meetings, plain", 0) != s["plain"]: MIS.append((f"L{l} {kind} plain", s["plain"], J.get(f"plain|L{l}|{kind}|meetings, plain")))
            del pl
        log(f"  GPT-2 layer {l:2d} done ({time.time() - t0:.0f}s, {rss()} MB)")
    return NP, BAD, MIS, J

def run_qwen(log, REC, layers):
    from audit_common import member as mem
    M_ = json.load(open(f"{HERE}/qwen06b/light/meta.json")); NP = M_["npos"]
    Z = np.load(f"{HERE}/groups2/machine/qwen_machine_counts.npz"); Q = {k.replace("__", "|"): Z[k] for k in Z.files}
    BAD = {}; MIS = []; t0 = time.time()
    for l in layers:
        for slab in ("qkv", "o", "gu", "down"):
            x = np.asarray(np.load(f"{HERE}/qwen06b/light/{slab}_light.npy", mmap_mode="r")[l]).astype(np.int64)
            W16 = mem(QNPZ, f"{slab}.{l}.w")
            pl = Plate(np.asarray(W16).astype(np.int64), QSEG[slab], 12, [1, SLABS.index(slab), l, 12])
            pile, c = pl.run(x)
            BAD["Qwen piles vs the direct product"] = BAD.get("Qwen piles vs the direct product", 0) + int((pile != product(x, np.asarray(W16))).any(1).sum())
            s = summed(c); REC[f"qwen|L{l}|{slab}"] = s
            mine = dict(light_ones=s["light_ones"], table_naf=s["table_naf"], pwrite_ones=s["pwrite_ones"], xsum_naf=s["xsum_naf"], xoff_naf=s["xoff_naf"])
            th = {"light_ones": "g_light", "table_naf": "g_table", "pwrite_ones": "g_pwrite", "xsum_naf": "g_xsum", "xoff_naf": "g_xoff"}
            for it, k in th.items():
                v = int(Q[f"{slab}.{l}|{k}"].sum())
                if v != mine[it]: MIS.append((f"Qwen L{l} {slab} {it}", mine[it], v))
            rd = int(sum(Q[f"{kd}.{l}|g_reads"].sum() for kd, _, _ in QSEG[slab])); sd = int(sum(Q[f"{kd}.{l}|g_sdreads"].sum() for kd, _, _ in QSEG[slab]))
            pq = int(sum(Q[f"{kd}.{l}|plain"].sum() for kd, _, _ in QSEG[slab]))
            for name, a, b in ((("reads_naf", s["reads_naf"], rd)), ("sdreads_naf", s["sdreads_naf"], sd), ("plain", s["plain"], pq)):
                if a != b: MIS.append((f"Qwen L{l} {slab} {name}", a, b))
            del pl
        log(f"  Qwen layer {l:2d} done ({time.time() - t0:.0f}s, {rss()} MB)")
    return NP, BAD, MIS

def main():
    ARG = dict(a.split("=") for a in sys.argv[1:]); t0 = time.time()
    QL = list(range(28)) if ARG.get("qwen_layers", "14") == "all" else [int(v) for v in ARG.get("qwen_layers", "14").split(",")]
    TAG = "audit_bank_both_ways" + ("" if ARG.get("gpt2", "1") == "1" and ARG.get("qwen_layers", "14") == "14" else "_qwen" + ARG.get("qwen_layers", "14").replace(",", "-"))
    log_f = open(f"{CG}/{TAG}.log", "w")
    def log(s=""): print(s, flush=True); log_f.write(s + "\n"); log_f.flush()
    log(f"AUDIT 2: the banks recounted both ways; GPT-2 all 48 banks; Qwen layers {QL}")
    REC = {}
    if ARG.get("gpt2", "1") == "1":
        NPg, BADg, MISg, J = run_gpt2(log, REC)
    NPq, BADq, MISq = run_qwen(log, REC, QL)
    log("")
    log("EXACT: " + ", ".join(f"{k} {v}" for k, v in {**(BADg if ARG.get('gpt2', '1') == '1' else {}), **BADq}.items()))
    MIS = (MISg if ARG.get("gpt2", "1") == "1" else []) + MISq
    log(f"MY DELIVERED ITEMS AGAINST THE BUILDER'S RECORDS: {len(MIS)} items differ" + ("" if not MIS else ": " + "; ".join(f"{a}: mine {b:,} theirs {c:,}" for a, b, c in MIS[:20])))
    log("")
    def table(title, keys, n, plain_key="plain"):
        tot = {}
        for k in keys:
            for kk, v in REC[k].items(): tot[kk] = tot.get(kk, 0) + v
        p, s = tot["plain"], tot["plain_sd"]
        log(f"  {title}: plain {p / n:,.0f} a token; same pairs in signed digits {s / n:,.0f} ({pct(s, p)} of plain)")
        log(f"      ready piles: 1s {tot['pwrite_ones'] / n:,.0f}, signed digits {tot['pwrite_naf'] / n:,.0f}; reads: signed digits {tot['reads_naf'] / n:,.0f}, "
            f"1s {tot['reads_ones'] / n:,.0f} (1s / signed digits: reads {tot['reads_ones'] / max(1, tot['reads_naf']):.3f}, writing {tot['pwrite_ones'] / max(1, tot['pwrite_naf']):.3f})")
        for f in FORMS:
            v = tot["F|" + f]
            log(f"      {f:12s} {v / n:14,.0f}  {pct(v, p):>7s} of plain  {pct(v, s):>7s} of signed-digit plain  {pct(v, tot['F|delivered']):>8s} of delivered")
        return tot
    log("BOTH WAYS, BY BANK (a token)")
    out = {}
    if ARG.get("gpt2", "1") == "1":
        for kind in ("qkv", "proj", "fc", "mproj"):
            out[f"gpt2 {kind} L1-11"] = table(f"GPT-2 {kind}, layers 1-11", [f"gpt2|L{l}|{kind}" for l in range(1, 12)], NPg)
        for kind in ("qkv", "proj", "fc", "mproj"):
            out[f"gpt2 {kind} L0 r1"] = table(f"GPT-2 {kind}.0, whole light, positions >= 1", [f"gpt2|L0|{kind}|r1"], NPg)
            out[f"gpt2 {kind} L0 f0"] = table(f"GPT-2 {kind}.0, whole light, position 0", [f"gpt2|L0|{kind}|f0"], NPg)
        out["gpt2 all 48"] = table("GPT-2 ALL 48 BANKS (layer 0 whole light, every position)", [k for k in REC if k.startswith("gpt2|")], NPg)
        log(f"  the builder's layer-0 side counts, whole light, positions >= 1: qkv.0 {J.get('side|L0|qkv|storage: the whole light in the group form: group work', 0) / NPg:,.0f}, "
            f"proj.0 {J.get('side|L0|proj|storage: the whole light in the group form: group work', 0) / NPg:,.0f}, fc.0 (group items + whole-form reads) "
            f"{J.get('side|L0|fc|storage: the whole light in the group form: group work', 0) / NPg:,.0f}")
    for slab in ("qkv", "o", "gu", "down"):
        out[f"qwen {slab}"] = table(f"Qwen {slab}, layers {QL[0]}-{QL[-1]}" if len(QL) > 1 else f"Qwen {slab}.{QL[0]}", [f"qwen|L{l}|{slab}" for l in QL], NPq)
    out["qwen all"] = table("QWEN ALL FOUR SLABS", [k for k in REC if k.startswith("qwen|")], NPq)
    json.dump(dict(rec=REC, totals=out, mismatches=MIS, qwen_layers=QL), open(f"{CG}/{TAG}.json", "w"), indent=0)
    log(f"({time.time() - t0:.0f}s, max memory {rss()} MB)")

if __name__ == "__main__":
    main()
