"""THE FINISHED PATTERN. Compare fringes after the meeting, not the light going in.
At a plate, each neuron's finished pattern is its 16x16 table of tips: each place of the light (up or hole) met against
each place of the plate over all lines, what is left after every up has met a down. The pile is made of these alone:
sum 2^(a+b) T_ab = 4 pile + 2 sum(light) + 2 sum(plate) + N, checked against the machine's pile.
Three pairs of finished patterns, the same token and plate:
  ctx vs own   : the token in context vs the token alone
  p vs p-1     : the token vs the token before it, both in context
  control      : the token vs a token at the same position in another text, both in context (pairing broken)
Measured: share of tips exactly equal; mean |change of tip|; share of neurons whose pile keeps its side (bright/dark);
and at the top read, how many of the 5 words agree.
PREDICTIONS (before running): tips: ctx vs own and p vs p-1 near the control at most place pairs, above it only at the
sign corner; piles keeping their side: 60-75% (control 50-60%)."""
import sys, os, json, numpy as np
HERE = "/Users/abundancemachine/streaming-machine"; os.chdir(HERE)
src = open("thinpage/thin_page.py").read(); exec(src[src.index("ARGS = sys.argv"):src.index("def ones(A)")])
TEXTS = [r["ids"] + r["ref"] for r in json.load(open("refs_many.json"))]; WTE, WPE = M["wte"], M["wpe"]
TIPB = ["qkv.6", "proj.6", "fc.6", "mproj.6"]
PL = {}
for k in TIPB:
    w = M[k + ".w"].astype(np.int64); assert np.abs(w).max() < 32768
    PL[k] = (np.stack([((((w + 32768) >> b) & 1) * 2 - 1) for b in range(16)]).astype(np.float32), w.sum(0))  # (16, lines, N)
def tips(x, k):                                              # x (P, lines) whole numbers -> T (P, 16, 16, N), exact
    assert np.abs(x).max() < 32768
    X = np.stack([((((x.astype(np.int64) + 32768) >> a) & 1) * 2 - 1) for a in range(16)], 1).astype(np.float32)
    Wp, _ = PL[k]; P, L = x.shape
    T = np.einsum("pal,bln->pabn", X, Wp, optimize=True)
    return np.rint(T).astype(np.int32)
def check(T, x, pile, k):
    s = sum((T[:, a, b, :].astype(np.int64) << (a + b)) for a in range(16) for b in range(16))
    L = x.shape[1]; rhs = 4 * pile.astype(np.int64) + 2 * x.astype(np.int64).sum(1)[:, None] + 2 * PL[k][1][None, :] + L
    return np.array_equal(s, rhs)
rng = np.random.default_rng(11)
runs = []
for t, ids in enumerate(TEXTS):                              # every text in context and alone
    R0 = WTE[ids] + WPE[:len(ids)]
    cc, topc = run(R0, attention); oo, topo = run(R0, attention_alone)
    runs.append(({k: (np.asarray(cc[k][0]).astype(np.int64), np.asarray(cc[k][2]).astype(np.int64)) for k in NAMES if k != "head"},
                 {k: (np.asarray(oo[k][0]).astype(np.int64), np.asarray(oo[k][2]).astype(np.int64)) for k in NAMES if k != "head"},
                 np.argsort(-topc, 1)[:, :5], np.argsort(-topo, 1)[:, :5]))
    print(f"  ran text {t}", flush=True) if t % 10 == 0 else None
ST = {}
def add(key, a, b): ST.setdefault(key, [0, 0]); ST[key][0] += a; ST[key][1] += b
for t, (C, O, t5c, t5o) in enumerate(runs):
    P = len(TEXTS[t]); others = [u for u in range(len(TEXTS)) if u != t and len(TEXTS[u]) >= P]
    u = others[rng.integers(len(others))] if others else None
    for k in NAMES[:-1]:
        pc, po = C[k][1], O[k][1]
        add(("side", k, "own"), int((np.sign(pc[1:]) == np.sign(po[1:])).sum()), pc[1:].size)
        add(("side", k, "prev"), int((np.sign(pc[1:]) == np.sign(pc[:-1])).sum()), pc[1:].size)
        if u is not None:
            pu = runs[u][0][k][1][:P]; add(("side", k, "ctl"), int((np.sign(pc[1:]) == np.sign(pu[1:])).sum()), pc[1:].size)
    for k in TIPB:
        xc, pc = C[k]; xo, po = O[k]
        Tc, To = tips(xc, k), tips(xo, k)
        assert check(Tc, xc, pc, k) and check(To, xo, po, k)
        pairs = {"own": (Tc[1:], To[1:]), "prev": (Tc[1:], Tc[:-1])}
        if u is not None:
            xu, pu = runs[u][0][k]; Tu = tips(xu[:P], k); assert check(Tu, xu[:P], pu[:P], k); pairs["ctl"] = (Tc[1:], Tu[1:])
        for name, (A, B) in pairs.items():
            eq = (A == B); d = np.abs(A.astype(np.int64) - B)
            add(("eq", k, name), int(eq.sum()), eq.size); add(("dt", k, name), int(d.sum()), d.size)
            ST.setdefault(("eqab", k, name), np.zeros((16, 16), np.int64)); ST[("eqab", k, name)] += eq.sum((0, 3))
            ST.setdefault(("nab", k, name), 0); ST[("nab", k, name)] += eq.shape[0] * eq.shape[3]
    for name, other in (("own", t5o[1:]), ("prev", t5c[:-1])):
        add(("top5", name), sum(len(set(a) & set(b)) for a, b in zip(t5c[1:], other)), 5 * (len(t5c) - 1))
    if u is not None:
        t5u = runs[u][2][:P]; add(("top5", "ctl"), sum(len(set(a) & set(b)) for a, b in zip(t5c[1:], t5u[1:])), 5 * (P - 1))
print("\n  all tables rebuilt from their tips equal the machine's piles (asserted at every bank, every text)")
print("\n  TIPS, the finished fringes (positions 1 on): share exactly equal | mean |change|")
for k in TIPB:
    print(f"    {k:8s}" + "".join(f"  {n}: {100*ST[('eq',k,n)][0]/ST[('eq',k,n)][1]:5.1f}% | {ST[('dt',k,n)][0]/ST[('dt',k,n)][1]:5.2f}" for n in ("own", "prev", "ctl")))
print("\n  tips exactly equal by place pair, fc.6 (light place down 15..0, plate place across 15..0), %:")
for n in ("own", "prev", "ctl"):
    E = 100 * ST[("eqab", "fc.6", n)] / ST[("nab", "fc.6", n)]
    print(f"    {n}:"); [print("      L%2d " % a + " ".join(f"{E[a, b]:4.0f}" for b in range(15, -1, -1))) for a in range(15, -1, -1) if a in (15, 14, 13, 12, 11, 10, 8, 5, 0)]
print("\n  PILES keeping their side (bright/dark), by layer, %   own / prev / control")
for kind in ("qkv", "proj", "fc", "mproj"):
    print(f"    {kind:6s}" + "".join(f" {l:2d}:" + "/".join(f"{100*ST[('side',f'{kind}.{l}',n)][0]/ST[('side',f'{kind}.{l}',n)][1]:.0f}" for n in ("own", "prev", "ctl")) for l in range(12)))
print("\n  TOP 5 words shared: " + "  ".join(f"{n}: {100*ST[('top5',n)][0]/ST[('top5',n)][1]:.1f}%" for n in ("own", "prev", "ctl")))
