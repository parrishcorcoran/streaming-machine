"""print develop_results.json as a table (words, photons, stored bytes per table)"""
import json, sys
rows = json.load(open("/Users/abundancemachine/streaming-machine/shrink/develop_results.json"))
flt = sys.argv[1] if len(sys.argv) > 1 else ""
for r in rows:
    if flt not in r["name"]: continue
    out = [f"{r['name']:<48s}", f"words {r.get('words', '?'):>2}"]
    for t in "ET":
        s = r.get("store_" + t)
        if f"photons_{t}" in r: out.append(f"{t}: photons {r[f'photons_{t}']:.2f}")
        if s: out.append(f"rc {s.get('rc_bytes', 0)/1e6:7.3f} MB lzma {s.get('lzma_bytes', 0)/1e6:7.3f} MB  {s['bits_per_value']:.3f} b/v {s['bits_per_word']:8.1f} b/word ok={s['roundtrip_ok']}")
    print("  ".join(out))
