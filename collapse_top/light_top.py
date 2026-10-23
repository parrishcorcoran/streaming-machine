"""THE LIGHT AT THE TOP, saved for every read position (the machine itself, unchanged: judge5's steps, base2's parts).
  refs:    the 30 refs texts, ids + ref[:16], read positions len(ids)-1 .. +15 (as collapse.py): 480 positions
  windows: the 300 judge5 windows, read positions 3..22 (as judge5.py): 6,000 positions
For each position: the light x16 (768 whole numbers, its own power of two ex) and LS (log2 of its water level), exactly
what the machine's top read meets. Also the machine's own top read there (its winner, by race), for the refs texts.
    python3 collapse_top/light_top.py        -> collapse_top/light_top.npz
"""
import sys, os, json, time, resource, warnings, numpy as np
warnings.filterwarnings("ignore", message=".*encountered in matmul")
HERE = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, HERE); os.chdir(HERE)
import judge5 as J
M, W = J.M, J.W
WTE = M["wte"]; WPE = M["wpe"]                       # read once (judge5 reads the same arrays on every call)
refs = json.load(open(f"{HERE}/refs_many.json"))
t0 = time.time()

def light_at_top(ids, rows):                          # judge5.scores_all, stopped just before the top read
    P = len(ids); R = WTE[ids] + WPE[:P]
    for l in range(12):
        R = R + J.plain_bank(J.attention(J.gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
        R = R + J.plain_bank(J.edge(J.gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
    return J.gauge_log(R[rows])                       # row by row: the same as gauge_log(R)[rows]

w, we, b = W["head"]
X, EX, LS, SRC, WIN = [], [], [], [], []
for t, r in enumerate(refs):
    ids = r["ids"] + r["ref"][:16]; rows = np.arange(len(r["ids"]) - 1, len(ids) - 1)
    x16, ex, ls = light_at_top(ids, rows)
    X.append(x16); EX.append(ex); LS.append(ls); SRC += [0] * len(rows); WIN += [t] * len(rows)
win_ref = np.array([J.race(J.power(J.product(x[None, :], w), (we + e + J.ES) * 65536 - l)[0] + b)
                    for xs, es, lss in zip(X, EX, LS) for x, e, l in zip(xs, es, lss)])
print(f"refs: {sum(len(x) for x in X)} positions ({time.time() - t0:.0f}s)", flush=True)
for i, win in enumerate(J.WINDOWS):
    x16, ex, ls = light_at_top(win["ids"], np.arange(J.FIRST, J.LAST))
    X.append(x16); EX.append(ex); LS.append(ls); SRC += [1] * len(ex); WIN += [i] * len(ex)
    if i % 50 == 49: print(f"  {i + 1} windows ({time.time() - t0:.0f}s, {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB)", flush=True)
X = np.concatenate(X); assert np.abs(X).max() <= 1 << 15
np.savez(f"{HERE}/collapse_top/light_top.npz", x16=X.astype(np.int32), ex=np.concatenate(EX), LS=np.concatenate(LS),
         src=np.array(SRC), win=np.array(WIN), refs_winner=win_ref)
print(f"saved collapse_top/light_top.npz: {len(X)} positions, light places used up to {int(np.abs(X).max()).bit_length() - 1}"
      f" ({time.time() - t0:.0f}s, {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB)")
