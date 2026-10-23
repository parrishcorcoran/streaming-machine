"""CHECKER: qwen_groups.py AS DELIVERED (its product hook, through qwen_fringes2's forward), on the checker's draw:
refs texts [5, 12, 19, 20, 28] and greedy texts [5, 12, 19, 20, 28]; every pile, arrival output, stream and the light at
the top compared by sha256 with qwen_base2.py's own steps run fresh (qwen_ref.json); then the top read (gtop.run_top,
as qwen_groups' phase_top runs it) on the machine's light: score rows' sha256, top 5 and race winner against
qwen_base2's fresh top read; on the greedy texts, the race winner at the 16 read places against the next greedy word
(words_ref.json) and against qwen_base2's fresh race winner. Writes only into check_machines/.
    python3 qwen_machine_check.py -> qwen_machine_check.log / .json / _counts.npz
"""
import os, sys, json, time, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/groups2/check_machines"
sys.path.insert(0, HERE); sys.path.insert(0, f"{HERE}/groups2")
argv = sys.argv; sys.argv = ["qwen_groups.py"]
import qwen_groups as QG
sys.argv = argv
QF = QG.QF; import gtop
LOGF = open(f"{OUT}/qwen_machine_check.log", "w")
def log(s=""): print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
REF = json.load(open(f"{OUT}/qwen_ref.json")); SEL = REF["sel"]
MT = json.load(open(f"{HERE}/qwen06b/light/meta.json")); WR = json.load(open(f"{HERE}/qwen_fringes/words_ref.json"))
t0 = time.time(); out = {}
TG = gtop.top_module("qwen", QG.G_TOP, False)
for kind, texts, pre in (("refs", [MT["texts"][t] for t in SEL], "r"), ("words", [WR["texts"][t] for t in SEL], "w")):
    res = QF.forward(texts, log=lambda s: None)
    nd = 0; nk = {}; miss = 0
    for k, v in res["dig"].items():
        part, *mid, ti = k.split("|"); rk = "|".join([part] + mid + [f"{pre}{SEL[int(ti)]}"])
        if rk not in REF["dig"]: miss += 1; continue
        nk[part] = nk.get(part, 0) + 1; nd += v != REF["dig"][rk]
    log(f"{kind} texts {SEL}: digests compared with qwen_base2's fresh run {nk}; differing {nd}; not found {miss}")
    log(f"   the machine's own checks: " + ", ".join(f"{k} {v}" for k, v in QF.ST["chk"].items()))
    if kind == "refs":
        # capture's record too (qwen06b/light/meta.json), for the piles: a second look at the reference itself
        cd = sum(REF["dig"][f"pile|{n}|r{t}"] != MT["dig"][n][t] for n in MT["dig"] for t in SEL)
        log(f"   qwen_base2's fresh piles against capture's record: {cd} of {len(MT['dig']) * len(SEL)} differ")
        np.savez(f"{OUT}/qwen_machine_check_counts.npz", **{k.replace("|", "__"): v for k, v in res["cnt"].items()}, rows=np.array(res["rows"]))
    X, EX, LS = res["x16"].astype(np.int64), res["ex"], res["LS"]
    if kind == "words":
        offs = np.concatenate([[0], np.cumsum([len(t) for t in WR["texts"]])]); RR = np.array(WR["read_rows"]); NX = np.array(WR["next"])
        myoff = np.concatenate([[0], np.cumsum([len(t) for t in texts])])
        selr, nxt, mine = [], [], []
        for j, t in enumerate(SEL):
            m = (RR >= offs[t]) & (RR < offs[t + 1]); loc = RR[m] - offs[t]
            selr += list(myoff[j] + loc); nxt += list(NX[m]); mine += [REF["words"][str(t)]["win"][i] for i in loc]
        selr = np.array(selr); X, EX, LS = X[selr], EX[selr], LS[selr]
        ref_sha = [REF["words"][str(SEL[int(np.searchsorted(myoff, q, 'right') - 1)])]["sha"][q - myoff[np.searchsorted(myoff, q, 'right') - 1]] for q in selr]
        ref5 = None; refw = np.array(mine)
    else:
        ref_sha = sum((REF["refs"][str(t)]["sha"] for t in SEL), []); ref5 = np.concatenate([np.array(REF["refs"][str(t)]["top5"]) for t in SEL])
        refw = np.concatenate([np.array(REF["refs"][str(t)]["win"]) for t in SEL])
    rec, bad, pwrite, lwrite, ppmask, lab = gtop.run_top(TG, X, EX, LS, False, lambda s: None, batch=8)
    r = rec["pp"]; n = len(X)
    log(f"   top read ({n} positions): checks " + ", ".join(f"{k} {v}" for k, v in sorted(bad["pp"].items())))
    msg = f"   score rows' sha256 = qwen_base2's fresh {sum(a == b for a, b in zip(rec['sha'], ref_sha))}/{n}; (ii) race winner = qwen_base2's fresh {int((r['ii_win'] == refw).sum())}/{n}"
    if ref5 is not None: msg += f"; (ii) top 5 = qwen_base2's fresh (set and order) {int((r['ii_top5'] == ref5).all(1).sum())}/{n}"
    if kind == "words": msg += f"; race winner = the next greedy word {int((r['ii_win'] == np.array(nxt)).sum())}/{n}; qwen_base2's fresh winner = the next greedy word {int((refw == np.array(nxt)).sum())}/{n}"
    log(msg)
    pl = rec["plain_meet"] + rec["plain_div"]
    log(f"   plain read {pl.sum() / n:,.0f} a position; (ii) {r['ii_total'].sum() / n:,.0f}; + ready piles written {pwrite.sum() / n:,.0f} + light {lwrite.sum() / n:,.0f}"
        f" = {100 * (r['ii_total'] + pwrite + lwrite).sum() / pl.sum():.2f}% of the plain read")
    out[kind] = dict(n=n, plain=pl.tolist(), ii_total=r["ii_total"].tolist(), pwrite=pwrite.tolist(), lwrite=lwrite.tolist(), table=r["table"].tolist(),
                     ii_reads=r["ii_reads"].tolist(), ii_light=r["ii_light"].tolist(), ii_bound=r["ii_bound"].tolist(), ii_end=r["ii_end"].tolist(),
                     ii_surv=r["ii_surv"].tolist(), bad=bad, ppmask=int(ppmask))
    log(f"   ({time.time() - t0:.0f}s, {QF.rss()} MB)")
json.dump(out, open(f"{OUT}/qwen_machine_check.json", "w"))
log(f"done ({time.time() - t0:.0f}s, max memory {QF.rss()} MB)")
