"""FLAT_DESIGN/2/PREDICT: carries the sample (measure.json) to the whole GPT-2 machine. Arithmetic only, labelled as an
estimate: each bank kind's confirmed cost (docs/09, base2_groups.py, a token on the 648 refs positions) times the
sample's ratio for that kind; the top read, arrival and rest unchanged.
    python3 flat_design/2/predict.py  -> flat_design/2/predict.log
"""
import json
OUT = "/Users/abundancemachine/streaming-machine/flat_design/2"
R = json.load(open(f"{OUT}/measure.json")); LOG = open(f"{OUT}/predict.log", "w")
def log(s): print(s); LOG.write(s + "\n")
PLAIN = 3_666_596_259
CONF = dict(qkv=132_698_796, proj=62_809_289, fc=190_757_633, mproj=231_896_719)          # the confirmed machine
REST = dict(arrival=8_020_119, top=80_689_701, rest=4_558_811)
PLAIN_KIND = dict(qkv=679_699_985, proj=248_461_379, fc=902_064_996, mproj=914_738_348, top=904_017_313)
def tot(c, extra=0): return c["build"] + c["write"] + c["read"] + c["xsum"] + c["off"] + extra

log("PART A: the excess (sum |tip|), share of its own meetings / share of plain photon meetings (shuffled light in brackets)")
wsum = {f: [0, 0, 0] for f in ("uh", "sm", "sd")}
for name, A in R["A"].items():
    kind = name.split(".")[0]
    row = []
    for f in ("uh", "sm", "sd"):
        a, s = A[f], A[f + "_shuffle"]
        row.append(f"{f} {100 * a['tips_abs'] / a['meetings']:.2f}% / {100 * a['tips_abs'] / A['plain']:.2f}% "
                   f"[{100 * s['tips_abs'] / s['meetings']:.2f} / {100 * s['tips_abs'] / A['plain_shuffle']:.2f}]")
        wsum[f][0] += PLAIN_KIND[kind] * a["tips_abs"] / A["plain"] / 2; wsum[f][1] += PLAIN_KIND[kind] / 2
        wsum[f][2] += PLAIN_KIND[kind] * a["tips_sd"] / A["plain"] / 2
    log(f"  {name:8s} " + "   ".join(row))
for f in ("uh", "sm", "sd"):
    log(f"  the four banks, weighted by each kind's plain: {f}: tips {100 * wsum[f][0] / wsum[f][1]:.2f}% of plain; "
        f"the tips' own signed digits (adding them as numbers) {100 * wsum[f][2] / wsum[f][1]:.2f}% of plain")

log("\nPART B: on texts 15-29 (325 positions), % of plain photon meetings")
ratio = {}; acc = {}
for name, B in R["B"].items():
    kind = name.split(".")[0]; p = B["plain"]; gm = tot(B["gm"]); fl = tot(B["flat"], B["stored_add"])
    run = gm - B["gm"]["read"] + B["gm"]["read_run"]
    a = acc.setdefault(kind, dict(gm=0, fl=0, run=0, p=0, reads=0, reads_e=0, gm_n=0, fl_n=0, adds_plain=0, sh_gm=0, sh_fl=0))
    a["gm"] += gm; a["fl"] += fl; a["run"] += run; a["p"] += p; a["reads"] += B["gm"]["read"]; a["reads_e"] += B["flat"]["read"]
    a["gm_n"] += B["gm"]["build_n"] + B["gm"]["write_n"] + B["gm"]["read_n"] + B["gm"]["xsum_n"] + B["gm"]["off_n"]
    a["fl_n"] += B["flat"]["build_n"] + B["flat"]["write_n"] + B["flat"]["read_n"] + B["flat"]["xsum_n"] + B["flat"]["off_n"] + B["M"] * B["T"]
    a["adds_plain"] += B["adds_plain"]; a["sh_gm"] += tot(B["gm_shuffle"]); a["sh_fl"] += tot(B["flat_shuffle"], B["stored_add_shuffle"])
    log(f"  {name:8s} gm {100 * gm / p:6.2f}%  flat held both sides {100 * fl / p:6.2f}% ({100 * fl / gm:6.2f}% of gm)  "
        f"sign run read once {100 * run / p:6.2f}%  | energy left {100 * B['energy_dep'] / B['energy_light']:5.1f}%  "
        f"digits left {100 * B['sd_dep'] / B['sd_light']:5.1f}%  reads left {100 * B['flat']['read'] / B['gm']['read']:6.2f}%  "
        f"| shuffle flat/gm {100 * tot(B['flat_shuffle'], B['stored_add_shuffle']) / tot(B['gm_shuffle']):6.2f}%  "
        f"| reads a neuron-group with the run {B['run_places_mean']:.2f}  photons a read {B['gm']['read'] / B['gm']['read_n']:.2f}")
log("\n  by kind (layers 3 and 9 together): flat/gm, adds (plain count) gm and flat as % of the direct machine's adds")
for kind, a in acc.items():
    ratio[kind] = a["fl"] / a["gm"]
    log(f"  {kind:6s} flat/gm {100 * ratio[kind]:6.2f}%  (shuffle {100 * a['sh_fl'] / a['sh_gm']:6.2f}%)  reads left {100 * a['reads_e'] / a['reads']:6.2f}%  "
        f"run/gm {100 * a['run'] / a['gm']:6.2f}%  adds: gm {100 * a['gm_n'] / a['adds_plain']:5.2f}%, flat {100 * a['fl_n'] / a['adds_plain']:5.2f}% of plain adds")

whole_gm = sum(CONF.values()) + sum(REST.values())
whole_fl = sum(CONF[k] * ratio[k] for k in CONF) + sum(REST.values())
log(f"\nWHOLE MACHINE (estimate): confirmed group machine {whole_gm:,} = {100 * whole_gm / PLAIN:.2f}% of plain; "
    f"with the flat held on both sides {whole_fl:,.0f} = {100 * whole_fl / PLAIN:.2f}%")
# the plate's sign run read once, carried the same way (it overlaps the confirmed per-place choice; an upper bound on its gain)
whole_run = sum(CONF[k] * acc[k]["run"] / acc[k]["gm"] for k in CONF) + sum(REST.values())
log(f"  the plate's sign run read once on the up/hole form (overlaps the confirmed per-place choice): at most {100 * whole_run / PLAIN:.2f}%")
