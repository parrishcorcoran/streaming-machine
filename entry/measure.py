"""MEASURE THE STORED ENTRY: bytes on disk, meetings at qkv.0 per token, and real wall-clock time, against base2.py.

Meetings are in the unit of every earlier count (extra/extra_layer0.py): a meeting is light photons x plate photons
for the lines actually met (the plate's photons summed over its 2,304 neurons, per line). Stored reads are counted
apart: the photons of the stored rows added, the photons of the rounded rows taken off (the split), and bytes read.
Over the 648 positions of the 30 refs texts (ids + ref).

PREDICTIONS (written before the run; the time ones from the short scratch probe of the same functions):
  count:   plain qkv.0 61,818,770 photons a token (the pilot's figure). Entry: about 2.7M (the pilot's 2,855,854 less
           position 0, which now reads its whole pile and meets nothing). Lit lines about 190 of 768 (24.8%).
           The residue layout (one unrounded tip per word, s digits of every line meet the plate) about 4x the carry.
           CONTROL (split with another word's rounded row): more than plain, about 63M.
  time:    base2's qkv.0 product about 310 us a text (about 14 us a position) and 60 us for one position alone.
           Entry, narrow carry: about 1.3x faster both ways. Literal pour: slower than base2's product both ways.
  forward: no difference beyond noise: qkv.0's product is well under 1% of a forward's time.
    python3 entry/measure.py parts=count,time,forward,forward2   (forward2: the word table already in memory)
"""
import sys, os, json, time, gc, platform, subprocess, resource, warnings, numpy as np
warnings.filterwarnings("ignore")
HERE = "/Users/abundancemachine/streaming-machine"; sys.path.insert(0, HERE); os.chdir(HERE)
PARTS = dict(a.split("=") for a in sys.argv[1:]).get("parts", "count,time,forward,forward2").split(",")
import base2_entry as E
J = E.J; t0 = time.time()
def gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
def say(*a): print(*a, flush=True)
WTE = J.M["wte"]; WPE = J.M["wpe"]; WQ = J.W["qkv.0"][0]; P16 = J.M["qkv.0.w"]
REFS = json.load(open(f"{HERE}/refs_many.json")); TEXTS = [np.array(r["ids"] + r["ref"]) for r in REFS]
NPOS = sum(len(t) for t in TEXTS)
POP = np.array([bin(i).count("1") for i in range(1 << 16)], np.int64)
def ones(A):
    A = np.abs(np.asarray(A).astype(np.int64)); n = np.zeros(A.shape, np.int64)
    while A.any(): n += POP[A & 0xFFFF]; A = A >> 16
    return n
WROW = ones(P16).sum(1)                               # the plate's photons per line (over its 2,304 neurons)
LIGHT = []                                            # (ids, A, B, x16, ex) for every text: the same light both ways
for ids in TEXTS:
    A = WTE[ids]; B = WPE[:len(ids)]; x16, ex, LS = J.gauge_log(A + B); LIGHT.append((ids, A, B, x16, ex))

def settings():
    env = {k: os.environ.get(k, "(unset)") for k in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    ncpu = subprocess.run(["sysctl", "-n", "hw.ncpu"], capture_output=True, text=True).stdout.strip()
    try: blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
    except Exception: blas = "unknown"
    say(f"settings: {cpu}, {ncpu} cores, macOS {platform.mac_ver()[0]}, Python {platform.python_version()}, numpy {np.__version__}, "
        f"BLAS/LAPACK {blas}; threads env {env}")

def part_count():
    MAN = json.load(open(f"{HERE}/entry/manifest.json"))
    say("BYTES ON DISK (entry/):")
    for name, t in MAN["tables"].items():
        b = sum(MAN["files"][f]["bytes"] for f in t["files"]) + MAN["files"][t["ids"]]["bytes"]
        say(f"  {name:6s} {t['rows']:6,} rows x 2,304, {t['kind']} ({t['bits']} bits signed needed): {b:>14,} bytes")
    tot = MAN["bytes on disk, tables and ids"]; sv = MAN["files"]["survey_s.npz"]["bytes"]
    say(f"  tables + ids {tot:,} bytes; with the survey {tot + sv:,}; manifest {os.path.getsize(f'{HERE}/entry/manifest.json'):,}")
    # the residue layout, for the choice: one unrounded tip per word and per position
    top_w = 0
    for i in range(0, 50257, 4096):
        top_w = max(top_w, int(np.abs((WTE[i:i + 4096].astype(np.float64) @ WQ)).max()))
    top_p = int(np.abs(WPE.astype(np.float64) @ WQ).max())
    bw = (max(top_w, top_p).bit_length() + 1 + 7) // 8
    alt = (50257 + 1024) * 2304 * bw
    say(f"  the residue layout would need: largest word tip {top_w:,}, largest position tip {top_p:,} -> {bw} bytes a value, "
        f"{alt:,} bytes ({100 * alt / tot:.0f}% of the chosen layout)")
    # meetings per token
    C = dict(plain=0, carry=0, residue=0, control=0, lit=0, dark=0, up=0, down=0, read_ph=0, split_ph=0, read_bytes=0, z_pos=0)
    rng = np.random.default_rng(3); bypos = {}
    for ids, A, B, x16, ex in LIGHT:
        s = ex + J.ES
        C["plain"] += int((ones(x16) @ WROW).sum())
        for i in range(len(ids)):
            sv = int(s[i])
            r = np.zeros(768, np.int64) if sv == 0 else np.bitwise_and(A[i] + B[i] + (1 << (sv - 1)), (1 << sv) - 1)
            C["residue"] += int(ones(r) @ WROW)
            other = WTE[rng.integers(0, 50257)]                          # CONTROL: another word's rounded row taken off
            C["control"] += int(ones(x16[i] - J.rshift(other, sv) - J.rshift(B[i], sv)) @ WROW)
            if sv in (4, 0):
                z = (E.ZP0 if sv == 4 else E.Z1023).read([ids[i]])[0]
                C["read_ph"] += int(ones(z).sum()); C["read_bytes"] += 2304 * 5; C["z_pos"] += 1; meet = 0; lit = 0
            else:
                a_s, b_s = J.rshift(A[i], sv), J.rshift(B[i], sv); c = x16[i] - a_s - b_s
                t, p = E.T[sv].read([ids[i]])[0], E.P[sv].read([i])[0]
                meet = int(ones(c) @ WROW); lit = int((c != 0).sum())
                C["read_ph"] += int(ones(t).sum() + ones(p).sum()); C["read_bytes"] += 2 * 2304 * 5
                C["split_ph"] += int(ones(a_s).sum() + ones(b_s).sum())
                C["up"] += int((c == 1).sum()); C["down"] += int((c == -1).sum())
            C["carry"] += meet; C["lit"] += lit
            bypos.setdefault(i, []).append(meet)
    n = NPOS; pl = C["plain"] / n; ca = C["carry"] / n
    say(f"\nMEETINGS AT qkv.0 PER TOKEN ({n} positions of the 30 refs texts; light photons x plate photons):")
    say(f"  plain (base2.py: the whole light meets the plate): {pl:16,.0f}")
    say(f"  entry (only the carry meets the plate):            {ca:16,.0f}   = {100 * ca / pl:.2f}% of plain, {pl / ca:.1f}x fewer")
    say(f"     lines lit by the carry: {C['lit'] / n:.1f} of 768 a token on average ({100 * C['lit'] / (768 * n):.1f}%); "
        f"down (-1) {100 * C['down'] / (768 * n):.2f}%, up (+1) {100 * C['up'] / (768 * n):.2f}% of lines; "
        f"{C['z_pos']} positions read their whole pile and meet nothing")
    say(f"  stored reads, apart: stored rows added {C['read_ph'] / n:,.0f} photons a token; rounded rows taken off (the split) "
        f"{C['split_ph'] / n:,.0f} photons; bytes read {C['read_bytes'] / n:,.0f} a token "
        f"(the plain meeting reads the whole plate: {P16.nbytes:,} bytes as int16)")
    say(f"  the residue layout (s digits of every line meet the plate): {C['residue'] / n:16,.0f}   = {C['residue'] / C['carry']:.1f}x the carry")
    say(f"  CONTROL (split with a random other word's rounded row):     {C['control'] / n:16,.0f}   = {100 * C['control'] / C['plain']:.1f}% of plain")
    whole = 3666596259
    say(f"  saving at qkv.0: {pl - ca:,.0f} photons a token = {100 * (pl - ca) / whole:.2f}% of the whole plain machine "
        f"(3,666,596,259 a token, extra/extra_layer0.py on the same 648 positions)")
    say("  carry meeting by position (mean): " + "  ".join(f"p{p} {np.mean(v) / 1e6:.2f}M" for p, v in sorted(bypos.items()) if p % 3 == 0))

def timeall(fns, reps=5):                          # runs interleaved, the order reversed every other round
    for f in fns: f()                                 # one untimed pass of each first (pages and caches warm)
    runs = [[] for _ in fns]
    for r in range(reps):
        for k in (range(len(fns)) if r % 2 == 0 else reversed(range(len(fns)))):
            gc.collect(); gc.disable(); t = time.perf_counter_ns(); fns[k](); runs[k].append(time.perf_counter_ns() - t); gc.enable()
    return [(float(np.median(r)), r) for r in runs]
def with_meet(how, fn):
    def g(): E.MEET = how; fn(); E.MEET = "narrow"
    return g
def timeit(fn, reps=5):
    fn(); runs = []                                   # one untimed pass first (pages and caches warm)
    for _ in range(reps):
        gc.collect(); gc.disable(); t = time.perf_counter_ns(); fn(); runs.append(time.perf_counter_ns() - t); gc.enable()
    return float(np.median(runs)), runs
def part_time():
    settings()
    pos = [np.arange(len(L[0])) for L in LIGHT]
    def base_text(): [J.product(x16, WQ) for ids, A, B, x16, ex in LIGHT]
    def entry_text(): [E.entry_pile(x16, ex, A, B, ids, p) for (ids, A, B, x16, ex), p in zip(LIGHT, pos)]
    def base_one(): [J.product(x16[i:i + 1], WQ) for ids, A, B, x16, ex in LIGHT for i in range(len(ids))]
    def entry_one(): [E.entry_pile(x16[i:i + 1], ex[i:i + 1], A[i:i + 1], B[i:i + 1], ids[i:i + 1], [i]) for ids, A, B, x16, ex in LIGHT for i in range(len(ids))]
    # the parts of one position's entry work, timed apart (positions with s = 1 or 2)
    one = [(ids[i:i + 1], A[i:i + 1], B[i:i + 1], x16[i:i + 1], int(ex[i] + J.ES), i) for ids, A, B, x16, ex in LIGHT for i in range(len(ids)) if ex[i] + J.ES in (1, 2)]
    carries = [x - J.rshift(a, s) - J.rshift(b, s) for w, a, b, x, s, i in one]
    def reads(): [E.T[s].read(w) + E.P[s].read([i]) for w, a, b, x, s, i in one]
    def splits(): [x - J.rshift(a, s) - J.rshift(b, s) for w, a, b, x, s, i in one]
    def meets(): [E.carry_meet(c) for c in carries]
    say(f"\nWALL-CLOCK, qkv.0's pile from the same light (x16 and the water level's shift given), {NPOS} positions, 5 runs each, interleaved, median:")
    rows = []
    for label, fb, fe in (("whole text at once (base2.py's forward runs every position together)", base_text, entry_text),
                          ("one position at a time (the stream: each new token alone)", base_one, entry_one)):
        (mb, rb), rn, rl = timeall([fb, with_meet("narrow", fe), with_meet("lit", fe)])
        res = {"narrow": rn, "lit": rl}
        say(f"  {label}:")
        say(f"     base2.py product (float64, whole light x plate): {mb / 1e3 / NPOS:8.2f} us a position   (runs, ms total: {', '.join(f'{r / 1e6:.1f}' for r in rb)})")
        for how in ("narrow", "lit"):
            m, r = res[how]
            say(f"     entry, carry met '{how:6s}':                    {m / 1e3 / NPOS:8.2f} us a position = {mb / m:5.2f}x base2's speed   (runs: {', '.join(f'{x / 1e6:.1f}' for x in r)})")
    n1 = len(one)
    for label, fn in (("read the stored rows (T_s + P_s, 5-byte values put together)", reads),
                      ("split: rounded word and position rows taken off -> the carry", splits),
                      ("the carry met with the plate ('narrow')", meets)):
        m, _ = timeit(fn); say(f"     part, one position at a time, {label}: {m / 1e3 / n1:7.2f} us")
    E.MEET = "lit"; m, _ = timeit(meets); E.MEET = "narrow"
    say(f"     part, one position at a time, the carry met with the plate ('lit'): {m / 1e3 / n1:7.2f} us   ({n1} positions with s = 1 or 2)")

def part_forward():
    settings()
    say(f"\nWALL-CLOCK, the whole forward on the 30 refs texts ({NPOS} positions), 5 runs each, interleaved, median:")
    def base_fwd(): return [J.forward(list(ids)) for ids in TEXTS]
    def entry_fwd(): return [E.forward(list(ids)) for ids in TEXTS]
    assert base_fwd() == entry_fwd()
    (mb, rb), (me, re_) = timeall([base_fwd, entry_fwd])
    say(f"  base2.py forward (as written: reads the word table from base2_gpt2.npz on every call): {mb / 1e9:7.2f} s   runs {', '.join(f'{r / 1e9:.2f}' for r in rb)}")
    say(f"  base2_entry forward (the same):                                                     {me / 1e9:7.2f} s   runs {', '.join(f'{r / 1e9:.2f}' for r in re_)}   = {mb / me:.3f}x")
    def read_table():                                 # each read let go before the next (memory under 3 GB)
        for _ in TEXTS: x = J.M["wte"]; del x
    mw, _ = timeit(read_table)
    say(f"     of which reading the word table, 30 times: {mw / 1e9:.2f} s")
def part_forward2():
    settings()
    def head(R):
        w, we, b = J.W["head"]; x16, ex, LS = J.gauge_log(R[-1:])
        return J.race(J.power(J.product(x16, w), (we[None, :] + ex[:, None] + J.ES) * 65536 - LS[:, None])[0] + b)
    def base_layers(ids):                              # base2.py's forward lines, the word table already in memory
        R = WTE[ids] + WPE[:len(ids)]
        for l in range(12):
            R = R + J.plain_bank(J.attention(J.gauged_bank(R, f"qkv.{l}")), f"proj.{l}")
            R = R + J.plain_bank(J.edge(J.gauged_bank(R, f"fc.{l}")), f"mproj.{l}")
        return head(R)
    def b2(): return [base_layers(ids) for ids in TEXTS]
    def e2(): return [head(E.stream(ids, WTE)) for ids in TEXTS]
    assert b2() == e2()
    (mb, rb), (me, re_) = timeall([b2, e2])
    say(f"  with the word table already in memory: base2.py {mb / 1e9:.2f} s, base2_entry {me / 1e9:.2f} s = {mb / me:.3f}x   "
        f"(runs {', '.join(f'{r / 1e9:.2f}' for r in rb)} / {', '.join(f'{r / 1e9:.2f}' for r in re_)})")

for p in PARTS: {"count": part_count, "time": part_time, "forward": part_forward, "forward2": part_forward2}[p]()
say(f"\nmax memory {gb():.2f} GB, {time.time() - t0:.0f}s")
