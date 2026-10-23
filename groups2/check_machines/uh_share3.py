"""CHECKER: the up/hole share recounted at 3 plates drawn at random (numpy default_rng(20261008): GPT-2 mproj.0, Qwen o.3
and down.5), g = 12, the confirm step's light (groups/light_gpt2: base2's light on the 648 refs positions;
qwen06b/light: qwen_base2's light on the 613 positions), by the checker's own counter (mycount.py), every pile checked
(GPT-2 against recording/C_ref/plain_refs.npz; Qwen against capture's sha256 per text). Both forms: up/hole (every place
up or hole, 0xFFFF) set beside uhr_<model>_g12.json, and per place (the plate's own rule) beside uhp_<model>_g12.json.
share = (half table + reads + signed-digit reads + X + X's take-away) / plain, as uh_compare.py's MACHINE.
"""
import json, hashlib, time, numpy as np, sys
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/groups2/check_machines"; sys.path.insert(0, OUT)
from mycount import slab_work
SLABS = ["qkv", "proj", "fc", "mproj", "o", "gu", "down"]; G = 12
LOGF = open(f"{OUT}/uh_share3.log", "w")
def log(s=""): print(s, flush=True); LOGF.write(s + "\n"); LOGF.flush()
def dg(a): return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.int64)).tobytes()).hexdigest()
t0 = time.time()
for model, slab, l in (("gpt2", "mproj", 0), ("qwen", "o", 3), ("qwen", "down", 5)):
    name = f"{slab}.{l}"
    if model == "gpt2":
        x = np.asarray(np.load(f"{HERE}/groups/light_gpt2/{slab}.npy", mmap_mode="r")[l]).astype(np.int64)
        w = np.asarray(np.load(f"{HERE}/base2_gpt2.npz")[f"{name}.w"]).astype(np.int64)
        ref = np.load(f"{HERE}/recording/C_ref/plain_refs.npz")[f"{slab}_pile_{l}"].astype(np.int64)
        def same(p): return int((p != ref).any(1).sum())
    else:
        x = np.asarray(np.load(f"{HERE}/qwen06b/light/{slab}_light.npy", mmap_mode="r")[l]).astype(np.int64)
        w = np.asarray(np.load(f"{HERE}/qwen/qwen_base2.npz")[f"{name}.w"]).astype(np.int64)
        MT = json.load(open(f"{HERE}/qwen06b/light/meta.json")); rows = MT["rows"]; dig = MT["dig"][name]
        def same(p): return sum(dg(p[a:b]) != dig[t] for t, (a, b) in enumerate(rows))
    segs = [(slab, 0, w.shape[1])]
    for form, tag in (("up/hole", "uhr"), ("per place", "uhp")):
        R = json.load(open(f"{HERE}/groups2/out/{tag}_{model}_g12.json"))["runs"]["main"]; rp, rs = R[name], R[f"{name}.slab"]
        pile, c, ch = slab_work(x, w, segs, G, [1, SLABS.index(slab), l, G], masks={slab: 0xFFFF} if form == "up/hole" else None)
        mine = dict(plain=int(c[f"plain|{slab}"].sum()), read=int(c[f"reads|{slab}"].sum()), sdread=int(c[f"sdreads|{slab}"].sum()),
                    half=int(c["table"].sum()), xsum=int(c["xsum"].sum()), xoff=int(c["xoff"].sum()))
        theirs = dict(plain=rp["plain"], read=rp["read"], sdread=rp["sdread"], half=rs["half"], xsum=rs["xsum"], xoff=rp["offset"] + rp["make"])
        sh_m = (mine["half"] + mine["read"] + mine["sdread"] + mine["xsum"] + mine["xoff"]) / mine["plain"]
        sh_t = (theirs["half"] + theirs["read"] + theirs["sdread"] + theirs["xsum"] + theirs["xoff"]) / theirs["plain"]
        diff = {k: mine[k] - theirs[k] for k in mine if mine[k] != theirs[k]}
        log(f"{model} {name} ({len(x)} positions), {form}: piles differing {same(pile)}; signed places {[b for b in range(16) if not (ch[slab] >> b) & 1]}; "
            f"record's share {100 * sh_t:.4f}%, checker's {100 * sh_m:.4f}%; numbers differing {diff if diff else 'none'}  ({time.time() - t0:.0f}s)")
        log(f"    checker: {mine}")
