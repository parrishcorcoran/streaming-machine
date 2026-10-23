"""Checker: the delivered machine on 10 texts against the checker's own reference run of qwen_base2.py (ref10.py), and
both against capture.py's record; the machine's counts and read against the builder's saved arrays at the same rows."""
import json, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/qwen_fringes/indep_check"; QFO = f"{HERE}/qwen_fringes"
RF = json.load(open(f"{OUT}/ref10.json")); MJ = json.load(open(f"{OUT}/machine10.json")); meta = json.load(open(f"{HERE}/qwen06b/light/meta.json"))
T10 = RF["T10"]
# 1. digests: machine vs my reference
parts = {}
for k, v in RF["dig"].items():
    p = k.split("|")[0]; parts.setdefault(p, [0, 0]); parts[p][0] += 1; parts[p][1] += (MJ["dig"].get(k) != v)
print("machine vs checker's qwen_base2 reference (10 texts):", {p: f"{d} of {n} differ" for p, (n, d) in parts.items()})
print("  keys only in machine:", len(set(MJ["dig"]) - set(RF["dig"])), " only in ref:", len(set(RF["dig"]) - set(MJ["dig"])))
# 2. my reference's piles vs capture's record
pd = sum(RF["dig"][f"pile|{n}|{t}"] != meta["dig"][n][t] for n in meta["dig"] for t in T10); print(f"checker's reference piles vs capture's record: {pd} of {len(meta['dig']) * len(T10)} differ")
# 3. the read
rows = meta["rows"]; MT = np.load(f"{OUT}/machine10_top.npz"); MTJ = json.load(open(f"{OUT}/machine10_top.json"))
r_sha, r_t5, r_win, c_sha, c_t5, c_win = [], [], [], [], [], []
for t in T10:
    a, b = rows[t]; r = RF["read"][str(t)]; assert b - a == len(r["sha"])
    r_sha += r["sha"]; r_t5 += r["top5"]; r_win += r["win"]; c_sha += meta["sha"][a:b]; c_t5 += meta["top5"][a:b]; c_win += meta["win"][a:b]
n = len(r_sha)
print(f"the read, {n} positions: machine top 5 (set and order) = checker's ref {int((MT['top5'] == np.array(r_t5)).all(1).sum())}/{n}; race winner {int((MT['win'] == np.array(r_win)).sum())}/{n}; "
      f"machine's check-path score rows sha = checker's {sum(a == b for a, b in zip(MTJ['sha'], r_sha))}/{n}")
print(f"  checker's ref vs capture: sha {sum(a == b for a, b in zip(r_sha, c_sha))}/{n}, top5 {sum(a == b for a, b in zip(r_t5, c_t5))}/{n}, win {sum(a == b for a, b in zip(r_win, c_win))}/{n}")
print("  machine's band read checks:", MTJ["bad"])
# 4. machine counts on these texts vs the builder's saved counts
C10 = np.load(f"{OUT}/machine10_counts.npz"); CB = np.load(f"{QFO}/machine_counts.npz")
gr = np.concatenate([np.arange(*rows[t]) for t in T10])
dk = [k for k in C10.files if not np.array_equal(C10[k], CB[k][gr])]
print(f"machine counts (10 texts) vs builder's machine_counts.npz at the same rows: {len(dk)} of {len(C10.files)} arrays differ {dk[:5]}; keys {len(C10.files)} vs {len(CB.files)}")
TB = np.load(f"{QFO}/top_machine.npz")
dt = [k for k in MT.files if k in TB.files and not np.array_equal(MT[k], TB[k][gr])]
print(f"band read arrays (10 texts) vs builder's top_machine.npz: {len(dt)} of {len([k for k in MT.files if k in TB.files])} differ {dt}")
MB = json.load(open(f"{QFO}/machine.json"))
print("zero gaps per plate equal to builder's:", MJ["zg"] == MB["zg"])
