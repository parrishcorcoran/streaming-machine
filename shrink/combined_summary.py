"""COMBINED summary: every tried point of the family, its words (nearest frame = what is stored) and, where measured,
the level (mean of the nearest frame and 4 shifted-grid frames), with the total re-counted under the family's final
entry-table coder (combined_rc, LIM=10; the developed values are the same, only their coding changed). Sizes only
(cached, each coded and decoded once); no words are re-judged here."""
import sys; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from combined_lib import *
R = json.load(open(OUTJ))
frames = {}
for r in R:
    if r.get("control"): frames.setdefault(r["version"].split(" [phase")[0], []).append(r["words"])
rows = []
for r in R:
    if r.get("control") or r["version"].startswith(("FILE", "READ")): continue
    old_t = r["tables_MB"]; new_t = table_bytes(r["c"], r["cap"], r.get("half")) / 1e6
    tot = r["plates_MB"] + new_t + r["rest_MB"]
    f = frames.get(r["version"], [])
    lvl = round((r["words"] + sum(f)) / (1 + len(f)), 1) if f else None
    rows.append((r["version"], round(tot, 3), r["words"], f, lvl, round(r["plates_MB"], 3), round(new_t, 3), r["rest_MB"]))
for v, tot, w, f, lvl, p, t, rs in rows:
    print(f"{v[:72]:72s} {tot:8.3f} MB  words {w:2d}  frames {str(f):16s} level {lvl}   (plates {p}, tables {t}, rest {rs})")
json.dump([dict(version=v, total_MB=tot, words=w, frames=f, level=lvl, plates_MB=p, tables_MB=t, rest_MB=rs)
           for v, tot, w, f, lvl, p, t, rs in rows], open(SP + "combined_summary.json", "w"), indent=1)
