"""Checker: the arrival's place-pair weights (qwen_fringes.TW) doubled. If the machine's attention output changes, the
machine is using its signed-digit products (QK/AV's S), not qwen_base2's int product. Text 12."""
import sys, json, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine")
import qwen_fringes as QF
OUT = "/Users/abundancemachine/streaming-machine/qwen_fringes/indep_check"
_tw = QF.TW; QF.TW = lambda na, nb: 2 * _tw(na, nb)
meta = json.load(open("/Users/abundancemachine/streaming-machine/qwen06b/light/meta.json")); RF = json.load(open(f"{OUT}/ref10.json"))
res = QF.forward([meta["texts"][12]], "fr", light_cmp=False, log=lambda s: None)
print("checks:", QF.ST["chk"])
print("arrival digests differing from the reference with TW doubled:", sum(res["dig"][f"arrival|{l}|0"] != RF["dig"][f"arrival|{l}|12"] for l in range(28)), "of 28")
