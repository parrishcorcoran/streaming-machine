"""CHECKER: the delivered base2_groups.py on refs texts 0, 1, 2 (counted, shuffle control off), its counts set beside the
saved run groups2/machine/gpt2_refs_t0-1-2.json (made by the queue before the files changed at 03:41)."""
import sys, json, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, HERE); sys.path.insert(0, f"{HERE}/groups2")
argv = sys.argv; sys.argv = ["base2_groups.py"]; import base2_groups as BG; sys.argv = argv
C = BG.FM.Count(); BAD = {}
BG.forward([BG.FM.TEXTS[t] for t in (0, 1, 2)], C, None, lambda k, v: BAD.__setitem__(k, BAD.get(k, 0) + int(v)), log=lambda s: None)
S = json.load(open(f"{HERE}/groups2/machine/gpt2_refs_t0-1-2.json"))["counts"]
keys = set(S) | set(C); d = [k for k in keys if S.get(k) != C.get(k)]
print(f"delivered code, texts 0-2: {len(keys)} count keys, differing from the saved run {len(d)} {d[:5]}; check path {sum(BAD.values())}")
