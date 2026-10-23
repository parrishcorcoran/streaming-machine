"""Independent check of the thin-page report. Own alone run (the machine's own arrival called on one position at a
time, so each token's arrival covers only itself), own photon count (byte table), own work count."""
import os, sys, json, time, resource, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; os.chdir(HERE)
src = open("base2.py").read(); src = src[:src.index("tok = GPT2Tokenizer")]
src = src.replace("import sys, math, numpy as np, torch", "import sys, math, numpy as np").replace("from transformers import GPT2LMHeadModel, GPT2Tokenizer\n", "")
sys.argv = ["x"]; exec(src)
WTE, WPE = M["wte"], M["wpe"]
PC8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)
def phot(A):                                            # photons (1s) of |value|, per value
    a = np.abs(np.asarray(A, dtype=np.int64))
    return PC8[a.reshape(-1).view(np.uint8)].reshape(a.shape + (8,)).sum(-1)
BANKS = [f"{b}.{l}" for l in range(12) for b in ("qkv", "proj", "fc", "mproj")]
NAMES = BANKS + ["top"]
def plate(n): return W["head" if n == "top" else n][0]
ROWPH = {}
for n in NAMES:
    w = M[("head" if n == "top" else n) + ".w"]; r = np.zeros(w.shape[0], np.int64)
    for i in range(0, w.shape[0], 64): r[i:i+64] = phot(w[i:i+64]).sum(1)
    ROWPH[n] = r                                        # plate photons on each line (over every neuron out)
def arrival_alone(Q):                                   # the machine's arrival, handed one position at a time: covers only itself
    return np.vstack([attention(Q[p:p+1]) for p in range(Q.shape[0])])
def run(R0, alone):
    R = R0.copy(); x = {}; lv = {}
    for l in range(12):
        x[f"qkv.{l}"], lv[f"qkv.{l}"], _ = gauge_log(R)
        Q = gauged_bank(R, f"qkv.{l}")
        A = arrival_alone(Q) if alone else attention(Q)
        x[f"proj.{l}"], lv[f"proj.{l}"] = to16(A, -ES)
        R = R + plain_bank(A, f"proj.{l}")
        x[f"fc.{l}"], lv[f"fc.{l}"], _ = gauge_log(R)
        H = edge(gauged_bank(R, f"fc.{l}"))
        x[f"mproj.{l}"], lv[f"mproj.{l}"] = to16(H, -ES)
        R = R + plain_bank(H, f"mproj.{l}")
    x["top"], lv["top"], LS = gauge_log(R)
    w, we, b = W["head"]
    words = [race(power(product(x["top"][p:p+1], w), (we[None, :] + lv["top"][p] + ES) * 65536 - LS[p])[0] + b) for p in range(len(R))]
    return x, lv, R, words

refs = json.load(open("refs_many.json")); texts = [r["ids"] + r["ref"] for r in refs]
t0 = time.time()
XC, XO, LC, LO = [], [], [], []
ident_fail = 0; ident_n = 0; pos0_fail = 0; words_ok = words_n = 0; int64_ok = 0; int64_n = 0; last_ok = 0
for t, ids in enumerate(texts):
    R0 = WTE[ids] + WPE[:len(ids)]
    xc, lc, Rc, wc = run(R0, False)
    xo, lo, Ro, wo = run(R0, True)
    # the machine's own forward on the same text, last word
    last_ok += int(forward(ids) == wc[-1]) if t < 5 else 0
    for n in NAMES:
        if not (np.array_equal(xc[n][0], xo[n][0]) and lc[n][0] == lo[n][0]): pos0_fail += 1
        w = plate(n); thin = xc[n] - xo[n]
        pc, po, pt = product(xc[n], w), product(xo[n], w), product(thin, w)
        ident_n += 1; ident_fail += int(not np.array_equal(pc, po + pt))
        if t < 2 and n != "top":                        # plain int64, no float anywhere
            wi = M[n + ".w"].astype(np.int64)
            int64_n += 1; int64_ok += int(np.array_equal(xc[n] @ wi, xo[n] @ wi + thin @ wi) and np.array_equal(xc[n] @ wi, pc))
    nid = len(refs[t]["ids"])
    for k, rw in enumerate(refs[t]["ref"]):
        words_n += 1; words_ok += int(wc[nid - 1 + k] == rw)
    XC.append({n: xc[n].astype(np.int32) for n in NAMES}); XO.append({n: xo[n].astype(np.int32) for n in NAMES})
    LC.append(lc); LO.append(lo)
    print(f"text {t}: {len(ids)} positions  {time.time()-t0:.0f}s  rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30:.2f} GiB", flush=True)
print(f"\nidentity pile_ctx == pile_own + thin@plate: {ident_n - ident_fail}/{ident_n}")
print(f"int64 (no float) identity, texts 0-1, 48 banks each: {int64_ok}/{int64_n}")
print(f"position 0 alone == in context (x16 and water level) : {len(texts)*len(NAMES) - pos0_fail}/{len(texts)*len(NAMES)}")
print(f"words vs refs (teacher forced): {words_ok}/{words_n};  machine forward() last word == my run, texts 0-4: {last_ok}/5")

# ---- counts, positions >= 1
def tally(pairs):   # pairs: list of (n, ctx rows, other rows) -> dict per bank of sums
    pass
S = {n: dict(c=0, o=0, t=0, z=0, L=0, sc=0, st=0, wc=0, wt=0, so=0, mv=0, tok=0) for n in NAMES}
POS = {}
for t in range(len(texts)):
    for n in NAMES:
        xc, xo = XC[t][n][1:].astype(np.int64), XO[t][n][1:].astype(np.int64); th = xc - xo
        pc, po, pt = phot(xc), phot(xo), phot(th)
        s = S[n]; s["c"] += pc.sum(); s["o"] += po.sum(); s["t"] += pt.sum(); s["z"] += (th == 0).sum(); s["L"] += th.size
        s["sc"] += np.abs(xc).sum(); s["st"] += np.abs(th).sum(); s["so"] += np.abs(xo).sum()
        s["wc"] += (pc @ ROWPH[n]).sum(); s["wt"] += (pt @ ROWPH[n]).sum(); s["tok"] += len(xc)
        s["mv"] += (LC[t][n][1:] != LO[t][n][1:]).sum()
        for p in range(1, len(XC[t][n])):
            d = POS.setdefault(p, dict(c=0, t=0, wc=0, wt=0, tc=0, tt=0))
            a = phot(XC[t][n][p].astype(np.int64)); b = phot(XC[t][n][p].astype(np.int64) - XO[t][n][p])
            if n == "top": d["tc"] += a.sum(); d["tt"] += b.sum()
            else: d["c"] += a.sum(); d["t"] += b.sum()
            d["wc"] += a @ ROWPH[n]; d["wt"] += b @ ROWPH[n]
print("\nPER BANK (positions 1+): ctx photons/token, own/token, thin/token, thin %, own/ctx %, lines exactly 0 %, size %, work %, water moved %")
for n in NAMES:
    s = S[n]; k = s["tok"]
    print(f"  {n:9s} {s['c']/k:8.0f} {s['o']/k:8.0f} {s['t']/k:8.0f}  {100*s['t']/s['c']:6.1f}  {100*s['o']/s['c']:6.1f}  {100*s['z']/s['L']:5.2f}  {100*s['st']/s['sc']:6.1f}  {100*s['wt']/s['wc']:6.1f}  {100*s['mv']/k:5.1f}")
print("\nPER LAYER work share (thin with stored own piles / plain) and photon share")
for l in range(12):
    bs = [f"{b}.{l}" for b in ("qkv", "proj", "fc", "mproj")]
    print(f"  layer {l:2d}: work {100*sum(S[b]['wt'] for b in bs)/sum(S[b]['wc'] for b in bs):6.1f}%  photons {100*sum(S[b]['t'] for b in bs)/sum(S[b]['c'] for b in bs):6.1f}%")
def agg(ns, key_t, key_c): return 100 * sum(S[n][key_t] for n in ns) / sum(S[n][key_c] for n in ns)
tok = S["top"]["tok"]
print(f"\nOVERALL banks of 12 layers: photons {agg(BANKS,'t','c'):.1f}%  size {agg(BANKS,'st','sc'):.1f}%  work {agg(BANKS,'wt','wc'):.1f}%  zero lines {agg(BANKS,'z','L'):.2f}%  plain work/token {sum(S[n]['wc'] for n in BANKS)/tok:.3e}")
print(f"OVERALL with top read     : photons {agg(NAMES,'t','c'):.1f}%  size {agg(NAMES,'st','sc'):.1f}%  work {agg(NAMES,'wt','wc'):.1f}%  plain work/token {sum(S[n]['wc'] for n in NAMES)/tok:.3e}")
print(f"top read alone            : photons {agg(['top'],'t','c'):.1f}%  work {agg(['top'],'wt','wc'):.1f}%")
print(f"water moved overall (banks): {100*sum(S[n]['mv'] for n in BANKS)/(48*tok):.1f}% of bank-positions")
print("\nBY POSITION: banks photons %, top %, work % (with top)")
for p in sorted(POS):
    d = POS[p]; print(f"  pos {p:2d}: banks {100*d['t']/d['c']:6.1f}  top {100*d['tt']/d['tc']:6.1f}  work {100*d['wt']/d['wc']:6.1f}")

# ---- control: a different word's own column at the same position
rng = np.random.default_rng(12345)
C = {n: dict(c=0, t=0, st=0, sc=0, wc=0, wt=0) for n in NAMES}; nctl = 0; miss = 0
for t in range(len(texts)):
    for p in range(1, len(texts[t])):
        cands = [u for u in range(len(texts)) if u != t and len(texts[u]) > p and texts[u][p] != texts[t][p]]
        if not cands: miss += 1; continue
        u = cands[rng.integers(len(cands))]; nctl += 1
        for n in NAMES:
            xc = XC[t][n][p].astype(np.int64); th = xc - XO[u][n][p]
            a, b = phot(xc), phot(th); c = C[n]
            c["c"] += a.sum(); c["t"] += b.sum(); c["sc"] += np.abs(xc).sum(); c["st"] += np.abs(th).sum()
            c["wc"] += a @ ROWPH[n]; c["wt"] += b @ ROWPH[n]
def cag(ns, kt, kc): return 100 * sum(C[n][kt] for n in ns) / sum(C[n][kc] for n in ns)
print(f"\nCONTROL ({nctl} tokens, {miss} without a partner): banks photons {cag(BANKS,'t','c'):.1f}%  work {cag(BANKS,'wt','wc'):.1f}%  size {cag(BANKS,'st','sc'):.1f}%")
print(f"   with top: photons {cag(NAMES,'t','c'):.1f}%  work {cag(NAMES,'wt','wc'):.1f}%  size {cag(NAMES,'st','sc'):.1f}%   top alone photons {cag(['top'],'t','c'):.1f}%")
print("   per layer control photons: " + " ".join(f"{cag([f'{b}.{l}' for b in ('qkv','proj','fc','mproj')],'t','c'):.1f}" for l in range(12)))
print("   fc.6 / qkv.11 control photons: %.1f / %.1f" % (cag(['fc.6'],'t','c'), cag(['qkv.11'],'t','c')))

# ---- extra: this token's in-context light against the previous token's in-context light
E = dict(c=0, t=0, sc=0, st=0, wc=0, wt=0, wct=0, wtt=0)
for t in range(len(texts)):
    for n in NAMES:
        xc = XC[t][n].astype(np.int64); a = phot(xc[1:]); b = phot(xc[1:] - xc[:-1])
        if n != "top": E["c"] += a.sum(); E["t"] += b.sum(); E["sc"] += np.abs(xc[1:]).sum(); E["st"] += np.abs(xc[1:] - xc[:-1]).sum(); E["wc"] += (a @ ROWPH[n]).sum(); E["wt"] += (b @ ROWPH[n]).sum()
        E["wct"] += (a @ ROWPH[n]).sum(); E["wtt"] += (b @ ROWPH[n]).sum()
print(f"\nPREVIOUS PAGE (banks): photons {100*E['t']/E['c']:.1f}%  size {100*E['st']/E['sc']:.1f}%  work {100*E['wt']/E['wc']:.1f}%  work with top {100*E['wtt']/E['wct']:.1f}%")

# ---- order-free, fc.6
ok = 0
for t in range(len(texts)):
    perm = rng.permutation(768); w = plate("fc.6")
    xc, xo = XC[t]["fc.6"].astype(np.int64), XO[t]["fc.6"].astype(np.int64)
    ok += int(np.array_equal(product(xc, w), product(xo[:, perm], w[perm]) + product((xc - xo)[:, perm], w[perm])) and phot(xc - xo).sum() == phot((xc - xo)[:, perm]).sum())
print(f"ORDER-FREE fc.6: {ok}/{len(texts)}")

# ---- position: same word alone at its own position vs another position
del XO
cases = [(t, p) for t in range(len(texts)) for p in range(len(texts[t]))]
pick = rng.choice(len(cases), 200, replace=False); wid, p1, p2 = [], [], []
for i in pick:
    t, p = cases[i]; q = int(rng.integers(23)); q = q + (q >= p)
    wid.append(texts[t][p]); p1.append(p); p2.append(q)
wid, p1, p2 = map(np.array, (wid, p1, p2))
x1, _, _, _ = run(WTE[wid] + WPE[p1], True); x2, _, _, _ = run(WTE[wid] + WPE[p2], True)
inv0 = (p1 != 0) & (p2 != 0)
for n in ["qkv.0", "fc.0", "qkv.6", "fc.6", "qkv.11"]:
    d = x1[n] - x2[n]
    print(f"POSITION {n:7s}: diff photons {100*phot(d).sum()/phot(x1[n]).sum():5.1f}% of own   (no pos 0: {100*phot(d[inv0]).sum()/phot(x1[n][inv0]).sum():5.1f}%)  size {100*np.abs(d).sum()/np.abs(x1[n]).sum():5.1f}%  zero lines {100*(d==0).mean():4.1f}%")
print(f"\ndone {time.time()-t0:.0f}s  peak rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30:.2f} GiB")
