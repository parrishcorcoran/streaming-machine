"""Checker: the 16 greedy words. qwen_base2.py's own steps (int64 products, as ref10.py), text by text, on each prompt
followed by the 15 recorded greedy words: the race winner at each of the 16 read places must be the next recorded word
(then the 16 are qwen_base2's own greedy words). Decoded (tokenizer from the local cache, offline, nothing downloaded)
against check30.log's printed words and against fp32's 16. The machine's race winners (qwen_fringes/top_words.npz)
against this reference at all 480 places."""
import os, sys, json, ast, time, numpy as np
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
HERE = "/Users/abundancemachine/streaming-machine"; QD = f"{HERE}/qwen"; NPZ = f"{QD}/qwen_base2.npz"; Q = f"{HERE}/qwen_fringes"
MM = np.load(NPZ)
class OnePlate(dict):
    def __getitem__(self, name):
        if name not in self.keys():
            self.clear(); dict.__setitem__(self, name, (np.asfortranarray(MM[name + ".w"].astype(np.int64)), MM[name + ".e"]))
        return dict.__getitem__(self, name)
src = open(f"{QD}/qwen_base2.py").read(); src = src[:src.index("from transformers import AutoTokenizer")]
for a, b in [('M = np.load("qwen_base2.npz")', 'M = MM'), ('W = {k[:-2]: (M[k].astype(np.int64), M[k[:-2] + ".e"]) for k in M.files if k.endswith(".w")}', 'W = OnePlate()')]:
    assert src.count(a) == 1; src = src.replace(a, b)
G = dict(MM=MM, OnePlate=OnePlate); sys.argv = ["x"]; exec(src, G)
gauged_slab, plain_slab, attention, bend, race, W, L = (G[k] for k in ("gauged_slab", "plain_slab", "attention", "bend", "race", "W", "L"))
EMB = np.load(NPZ, mmap_mode="r")  # npz members are read whole; use rows via a memmap below
import zipfile
info = zipfile.ZipFile(NPZ).getinfo("embed.npy")
with open(NPZ, "rb") as f:
    f.seek(info.header_offset); h = f.read(30); f.seek(info.header_offset + 30 + int.from_bytes(h[26:28], "little") + int.from_bytes(h[28:30], "little"))
    ver = np.lib.format.read_magic(f); shape, fo, dt = np.lib.format.read_array_header_1_0(f) if ver == (1, 0) else np.lib.format.read_array_header_2_0(f); off = f.tell()
E = np.memmap(NPZ, dtype=dt, mode="r", offset=off, shape=shape)
HW = MM["head.w"]; HE = MM["head.e"]; NW = HW.shape[1]
WR = json.load(open(f"{Q}/words_ref.json")); RQ = json.load(open(f"{QD}/refs_qwen.json")); t0 = time.time()
wins = []
for i, r in enumerate(RQ):
    ids = WR["texts"][i]; R = np.asarray(E[ids]).copy(); n0 = len(r["ids"])
    for l in range(L):
        R = R + plain_slab(attention(gauged_slab(R, f"qkv.{l}"), l), f"o.{l}")
        GU = gauged_slab(R, f"gu.{l}"); R = R + plain_slab(bend(GU[:, :3072], GU[:, 3072:]), f"down.{l}")
    Rr = R[n0 - 1:n0 + 15]; S = np.empty((len(Rr), NW), np.int64)
    for c in range(0, NW, 16384):
        W.clear(); dict.__setitem__(W, "head", (np.asfortranarray(HW[:, c:c + 16384].astype(np.int64)), HE[c:c + 16384]))
        S[:, c:c + 16384] = gauged_slab(Rr, "head")
    W.clear(); wins += [race(s) for s in S]
print(f"reference read at the 480 places: race winner = the recorded next word {sum(a == b for a, b in zip(wins, WR['next']))}/480 ({time.time() - t0:.0f}s)")
words = [wins[16 * i:16 * (i + 1)] for i in range(30)]
print("recorded 16 words = the reference's greedy words:", sum(words[i] == WR["words"][i] for i in range(30)), "/30;  = fp32's:", sum(words[i] == RQ[i]["ref"] for i in range(30)), "/30")
TW = np.load(f"{Q}/top_words.npz"); print("machine's race winners (top_words.npz) = this reference:", int((TW["win"] == np.array(wins)).sum()), "/480")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
c30 = {}
for line in open(f"{QD}/check30.log"):
    for tag in ("IDENTICAL", "DIFFERENT"):
        if f" {tag}  " in line:
            pr, rest = line.strip().split(f" {tag}  ", 1); c30[ast.literal_eval(pr.strip())] = ast.literal_eval(rest.split("   model: ")[0].strip())
print("decoded = check30.log's words:", sum(tok.decode(words[i]) == c30[RQ[i]["prompt"]] for i in range(30)), "/30")
for i in range(30):
    if words[i] != RQ[i]["ref"]: print("  differs from fp32:", repr(RQ[i]["prompt"]), repr(tok.decode(words[i])))
