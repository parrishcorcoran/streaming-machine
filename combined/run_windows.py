"""RUN 2: THE 300 JUDGE5 WINDOWS (24 tokens each; every position goes through every layer; the top read is at positions
3..22). Done in pieces, each under ten minutes, each saving its part.

  stream a=0 b=25   windows a..b-1. For each window the plain machine (base2.py's own steps) and the combined machine
                    both run. Every pile and value of all 12 layers, the stream and the light at the top are compared
                    exactly. The light at the top is also compared with collapse_top/light_top.npz (saved from the plain
                    machine). Saves combined/parts/light_a_b.npz.
  top a=0 b=1200    the collapsing top read at judged positions a..b-1 (of 6,000), on the combined machine's light.
                    Saves combined/parts/top_a_b.npz.
  report            all parts together:
                      - the top 5 (set and order) against judge5's reference (judge5_ref.npz);
                      - judge5's two counts: the machine's word in the reference top 5, and the same top word.
                    Saves combined/windows_top.npz for combined/judge_check.py (judge5.judge itself).
    python3 combined/run_windows.py stream a=0 b=25 | top a=0 b=1200 | report
"""
import sys, os, glob, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
t0 = time.time()
def log(s): print(s, flush=True)
MODE = sys.argv[1]; A = dict(a.split("=") for a in sys.argv[2:]); a, b = int(A.get("a", 0)), int(A.get("b", 0))
HERE = "/Users/abundancemachine/streaming-machine"; PARTS = f"{HERE}/combined/parts"; os.makedirs(PARTS, exist_ok=True)

if MODE == "stream":
    import machine as K
    K.build_lineups(); log(f"loaded; line-ups stored ({time.time() - t0:.0f}s, {K.rss()} MB)")
    L = np.load(f"{HERE}/collapse_top/light_top.npz"); SRC, WIN = L["src"], L["win"]
    X, EX, LS, BAD, SAME = [], [], [], [], []; rows = np.arange(K.FIRST, K.LAST)
    for i in range(a, b):
        ids = K.WINDOWS[i]["ids"]; ref = K.plain_run(ids, keep_top=False); rec = {}
        K.forward(ids, None, rec)
        bad = K.same_as_plain(rec, ref); BAD.append(len(bad))
        x16, ex, ls = rec["top light"]; X.append(x16[rows]); EX.append(ex[rows]); LS.append(ls[rows])
        gi = np.flatnonzero((SRC == 1) & (WIN == i))
        SAME.append(int(np.array_equal(x16[rows], L["x16"][gi]) and np.array_equal(ex[rows], L["ex"][gi]) and np.array_equal(ls[rows], L["LS"][gi])))
        del ref, rec
        log(f"  window {i}: every pile and value identical: {not bad}{'' if not bad else ' -- ' + ', '.join(bad[:6])}; "
            f"light at the top = the saved light: {bool(SAME[-1])} ({time.time() - t0:.0f}s, {K.rss()} MB)")
    np.savez(f"{PARTS}/light_{a:03d}_{b:03d}.npz", X=np.concatenate(X), EX=np.concatenate(EX), LS=np.concatenate(LS),
             windows=np.arange(a, b), bad=np.array(BAD), same=np.array(SAME))
    log(f"windows {a}..{b - 1}: every pile and value identical in {sum(x == 0 for x in BAD)}/{b - a}; light at the top = "
        f"saved light in {sum(SAME)}/{b - a} ({time.time() - t0:.0f}s, max memory {K.rss()} MB)")

elif MODE == "top":
    import machine as K
    parts = sorted(glob.glob(f"{PARTS}/light_*.npz")); Z = [np.load(p) for p in parts]
    W_ = np.concatenate([z["windows"] for z in Z]); assert np.array_equal(W_, np.arange(300)), "windows missing"
    X, EX, LS = (np.concatenate([z[k] for z in Z]) for k in ("X", "EX", "LS"))
    T = K.top_read(X[a:b], EX[a:b], LS[a:b], None, log=log)
    np.savez(f"{PARTS}/top_{a:04d}_{b:04d}.npz", **{k: v for k, v in T.items() if k != "bad"},
             bad=np.array([T["bad"][k] for k in ("bound", "final_vs_plain", "pairs_sum")]), a=a, b=b)
    log(f"top read at judged positions {a}..{b - 1}: bound broken {T['bad']['bound']}, words read to the end != plain "
        f"{T['bad']['final_vs_plain']}, pairs != plain {T['bad']['pairs_sum']} ({time.time() - t0:.0f}s, max memory {K.rss()} MB)")

elif MODE == "report":
    pct = lambda x, y: (200 * x + y) // (2 * y)
    Z = [np.load(p) for p in sorted(glob.glob(f"{PARTS}/light_*.npz"))]
    bad = np.concatenate([z["bad"] for z in Z]); same = np.concatenate([z["same"] for z in Z])
    log(f"every pile and value of all 12 layers, the stream and the light at the top identical to the plain machine: "
        f"{int((bad == 0).sum())}/{len(bad)} windows; light at the top = collapse_top/light_top.npz: {int(same.sum())}/{len(same)}")
    T = sorted((np.load(p) for p in glob.glob(f"{PARTS}/top_*.npz")), key=lambda z: int(z["a"]))
    assert [int(z["a"]) for z in T] == [0] + [int(z["b"]) for z in T[:-1]] and int(T[-1]["b"]) == 6000, "top parts missing"
    cat = lambda k: np.concatenate([z[k] for z in T])
    top5, win, ptop5, surv, work, plain = cat("top5"), cat("win"), cat("plain_top5"), cat("surv"), cat("work"), cat("plain")
    fin_idx, fin_sc = cat("fin_idx"), cat("fin_sc"); offs = [0]
    for z in T: offs += list(offs[-1] + z["fin_off"][1:])
    badt = sum(z["bad"] for z in T)
    R = np.load(f"{HERE}/judge5_ref.npz")["top5"].reshape(-1, 5); n = len(R)
    log(f"TOP READ at {n} judged positions: bound broken {badt[0]} times; words read to the end != plain read {badt[1]}; "
        f"stages' pairs != plain meetings at {badt[2]} positions")
    log(f"  top 5 (set and order) = the plain read's here: {int((top5 == ptop5).all(1).sum())}/{n}; = judge5's reference: "
        f"{int((top5 == R).all(1).sum())}/{n}")
    log(f"  judge5: in the reference top 5 {int((R == win[:, None]).any(1).sum())}/{n}, same top word {int((R[:, 0] == win).sum())}/{n}")
    for s, k in enumerate((4, 8, 12)):
        v = np.sort(surv[:, s]); log(f"  words still in after the light's top {k:2d} places: median {v[len(v) // 2]}, least {v[0]}, most {v[-1]}")
    log(f"  top read meetings per position: {(int(work.sum()) + n // 2) // n:,} against plain {(int(plain.sum()) + n // 2) // n:,}"
        f" = {pct(int(work.sum()), int(plain.sum()))}%")
    np.savez(f"{HERE}/combined/windows_top.npz", top5=top5, win=win, fin_idx=fin_idx, fin_sc=fin_sc, fin_off=np.array(offs),
             surv=surv, work=work, plain=plain, n=300)
    log("saved combined/windows_top.npz")
