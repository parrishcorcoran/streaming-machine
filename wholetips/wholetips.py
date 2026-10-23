"""THE WHOLE MACHINE CUT TO ITS TIPS, AGAINST THE CRUDE GRID.

HIS WORDS (2026-10-08), the only sentences quoted as his:
  "The WHOLE MODEL has to shrink together, JUST like quantization. The embedding etc all of it, JUST like quantization
  BUT with our precision. The tips needed!"
  Before: "every weight is a vector ... we break down each weight into all its base 2 circles ... only a couple of
  those circles hold the vector and most cancel." "The Lower circles don't matter. And since it's 90% that's what
  quantization does. It crudely cuts out most of the canceling parts cheaply, but it's not exact."

THE BUILD (the session's restatement): every stored number of GPT-2 (124,439,808: the word table, which is also the top
read, the position rows, the 48 plates, every bias, every water-level gain and offset) is written in base 2 as single
waves (each lit place up +1 or a hole -1, no two lit places side by side) and keeps only its own top k lit places, k =
1, 2, 3, fixed, wherever they sit; every lower circle dropped. Against it, the crude grid: one even grid per row (a
plate's row = one neuron's inputs; a word's 768 values; a position's 768; each bias or gain vector one row), step =
the row's largest number / (2^(b-1) - 1) written as a 16-bit float, round to nearest, b = 2, 3, 4, 5, 6.
The cut numbers are exact binary values, so fp32 GPT-2 (HF transformers, CPU) runs them as they are.

THE WORDS: judge5's 300 windows (24 tokens, positions 3..22 judged: 6,000 positions), against the uncut fp32 machine:
  the full machine's word in the version's top 5; the version's word in the full machine's top 5 (judge5's own rule);
  the same top word. The 30 refs texts: 16 greedy words each, the version running free, against refs_many.json
  (fp32's own words): words kept of 480, texts kept whole, words before the first difference. The record's judge
  (shrink/): the next word after the first 5 tokens of each refs text, of 30.
STORED SIZE at the floor, the same way for every version, for every stored number:
  tips: per tensor, static counts of (top lit place, sign) or dark; then for each further kept circle, (gap down from
    the circle above, up or hole) or stop, one count table per round. Grid: per tensor, static counts of the levels,
    plus 16 bits a row scale. Every used count cell charged 32 bits. bits = sum of -log2(count / total).
CONTROL (error placement): each version's errors (stored minus cut) shuffled among the numbers of the same row; the
  same errors, put on other numbers.
BY PART: each version with only one part cut (word table / position rows / plates / small pieces), the rest uncut.
    python3 wholetips/wholetips.py [versions=name,...]
"""
import sys, os, json, time, resource, warnings, numpy as np
warnings.filterwarnings("ignore")
import torch
from transformers import GPT2LMHeadModel
torch.set_grad_enabled(False)
HERE = "/Users/abundancemachine/streaming-machine"; OUT = f"{HERE}/wholetips"
ARG = dict(a.split("=", 1) for a in sys.argv[1:])
LOG = open(f"{OUT}/wholetips.log", "a"); T0 = time.time()
def log(s=""): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def mb():
    now = int(os.popen(f"ps -o rss= -p {os.getpid()}").read().strip() or 0) // 1000
    return f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 10 ** 6} peak, {now} now MB"

def wait_free():                                    # one heavy process at a time
    while True:
        ps = os.popen("ps -axo pid=,rss=,comm=").read().split("\n")
        other = [p for p in ps if p.strip() and "ython" in p and int(p.split()[0]) != os.getpid() and int(p.split()[1]) > 1_500_000]
        if not other: return
        log(f"  waiting: another python process over 1.5 GB is running ({other[0].split()[0]})"); time.sleep(60)

# ---------------- the machine and its stored numbers ----------------
MODEL = GPT2LMHeadModel.from_pretrained("gpt2").eval()
PARAMS = dict(MODEL.named_parameters())             # the top read is tied to the word table: one stored table
NAMES = list(PARAMS)
ORIG = {n: p.detach().numpy().copy() for n, p in PARAMS.items()}
NV = sum(a.size for a in ORIG.values()); assert NV == 124_439_808
def part(n):
    if n.endswith("wte.weight"): return "word table"
    if n.endswith("wpe.weight"): return "position rows"
    if ORIG[n].ndim == 2: return "plates"
    return "small pieces"
PARTS = ["word table", "position rows", "plates", "small pieces"]
PART_NV = {p: sum(ORIG[n].size for n in NAMES if part(n) == p) for p in PARTS}
def rows_of(n, A):                                  # the grid's rows: a plate's neuron, a word, a position, a whole vector
    if A.ndim == 1: return A[None, :]
    if part(n) == "plates": return A.T              # HF plates are (lines in, neurons out)
    return A
def from_rows(n, R, shape):
    if len(shape) == 1: return R[0]
    if part(n) == "plates": return R.T
    return R

# ---------------- single waves ----------------
def naf_split(a):                                   # a >= 0 -> (up, hole), a = up - hole, no two lit side by side
    ah = a >> 1; a3 = a + ah; c = ah ^ a3
    return a3 & c, ah & c
def topbit(v): return np.frexp(v.astype(np.float64))[1] - 1     # v > 0, < 2^53

OFF = 200                                           # absolute place offset for the count tables
def tips_cut(A, k, acc):
    """A float32 -> each number's own top k lit places (single waves), float32, exact. acc: count tables (per tensor)."""
    x = A.astype(np.float64).ravel(); neg = x < 0
    mant, ex = np.frexp(np.abs(x)); m = np.rint(mant * 2.0 ** 24).astype(np.int64)
    assert np.array_equal(np.ldexp(m.astype(np.float64), ex - 24), np.abs(x))
    up, hole = naf_split(m); lit = up | hole
    assert np.array_equal(up - hole, m)
    cnt = np.bitwise_count(lit).astype(np.int64)
    while (cnt > k).any():
        sel = cnt > k; lit = np.where(sel, lit & (lit - 1), lit); cnt = cnt - sel      # put out the lowest lit place
    kept = (up & lit) - (hole & lit)
    v = np.ldexp(kept.astype(np.float64), ex - 24); v = np.where(neg, -v, v)
    out = v.astype(np.float32); assert np.array_equal(out.astype(np.float64), v)        # exact in fp32
    # sizes: round 1 (top place + sign, or dark); round r (gap + up/hole, or stop)
    rem = lit.copy(); alive = rem > 0
    t = np.where(alive, topbit(np.maximum(rem, 1)), 0); place = t + ex - 24
    assert (place[alive] + OFF > 0).all() and (place[alive] + OFF < 400).all()
    s1 = np.where(alive, 1 + 2 * (place + OFF) + neg, 0)
    acc["r1"] += np.bincount(s1, minlength=801)[:801]
    rem = np.where(alive, rem ^ (np.int64(1) << t), 0); prev = t
    for r in range(2, k + 1):
        go = alive & (rem > 0)
        tt = np.where(go, topbit(np.maximum(rem, 1)), 0); hb = (hole >> tt) & 1
        gap = prev - tt
        assert (gap[go] >= 2).all() and (gap[go] < 40).all()
        sr = np.where(go, 1 + 2 * (gap - 2) + hb, 0)
        acc[f"r{r}"] += np.bincount(sr[alive], minlength=80)[:80]                     # only numbers still lit ask
        rem = np.where(go, rem ^ (np.int64(1) << tt), 0); prev = np.where(go, tt, prev); alive = go
    return out.reshape(A.shape)

def grid_cut(R, b):
    """R (rows, n) float32 -> (cut float32, levels int, scales float16) on one even grid per row"""
    L = 2 ** (b - 1) - 1
    mx = np.abs(R).max(1).astype(np.float64)
    sc = (mx / L).astype(np.float16); scf = sc.astype(np.float32)
    assert ((scf > 0) | (mx == 0)).all()
    q = np.where(scf[:, None] > 0, np.rint(R / np.where(scf > 0, scf, 1)[:, None]), 0)
    q = np.clip(q, -L, L).astype(np.int64)
    v = (q.astype(np.float32) * scf[:, None]).astype(np.float32)
    assert np.array_equal(v.astype(np.float64), q.astype(np.float64) * scf.astype(np.float64)[:, None])
    return v, q, sc

def floor_bits(counts):
    c = np.asarray(counts, np.float64); c = c[c > 0]; N = c.sum()
    return float(-(c * np.log2(c / N)).sum()), int(c.size)
CELL = 32                                           # bits charged for each used count cell

def build(spec):
    """-> CUT {name: float32}, size rows per tensor"""
    CUT, SIZE = {}, {}
    for n in NAMES:
        A = ORIG[n]
        if spec["kind"] == "tips":
            k = spec["k"]; acc = {"r1": np.zeros(801, np.int64), **{f"r{r}": np.zeros(80, np.int64) for r in range(2, k + 1)}}
            flat = A.ravel(); out = np.empty_like(flat); CH = 1 << 21
            for i in range(0, flat.size, CH): out[i:i + CH] = tips_cut(flat[i:i + CH], k, acc)
            CUT[n] = out.reshape(A.shape)
            bits = cells = 0.0; byround = {}
            for key, c in acc.items():
                b_, c_ = floor_bits(c); bits += b_; cells += c_; byround[key] = b_ / A.size
            SIZE[n] = dict(n=A.size, entropy_bits=bits, cells=cells, scale_bits=0, total_bits=bits + CELL * cells, byround=byround)
        elif spec["kind"] == "grid":
            R = rows_of(n, A); V = np.empty(R.shape, np.float32); qc = np.zeros(129, np.int64)
            for r0 in range(0, R.shape[0], 4096):           # a block of rows at a time (memory)
                v, q, sc = grid_cut(R[r0:r0 + 4096], spec["b"]); V[r0:r0 + 4096] = v
                qc += np.bincount(q.ravel() + 64, minlength=129)
            CUT[n] = np.ascontiguousarray(from_rows(n, V, A.shape)); del V
            b_, c_ = floor_bits(qc)
            SIZE[n] = dict(n=A.size, entropy_bits=b_, cells=c_, scale_bits=16 * R.shape[0],
                           total_bits=b_ + CELL * c_ + 16 * R.shape[0], plain_bits=spec["b"] * A.size + 16 * R.shape[0])
        elif spec["kind"] == "fp16":
            CUT[n] = A.astype(np.float16).astype(np.float32)
            SIZE[n] = dict(n=A.size, entropy_bits=16 * A.size, cells=0, scale_bits=0, total_bits=16 * A.size)
        elif spec["kind"] == "full":
            CUT[n] = A; SIZE[n] = dict(n=A.size, entropy_bits=32 * A.size, cells=0, scale_bits=0, total_bits=32 * A.size)
    return CUT, SIZE

def error_share(CUT):                               # the error left, as a share of the numbers' own size, by part
    out = {}
    for p in PARTS:
        e2 = x2 = 0.0
        for n in NAMES:
            if part(n) != p: continue
            o, c = ORIG[n].ravel(), CUT[n].ravel()
            for i in range(0, o.size, 1 << 22):
                a = o[i:i + (1 << 22)].astype(np.float64); d = a - c[i:i + (1 << 22)].astype(np.float64)
                e2 += float((d * d).sum()); x2 += float((a * a).sum())
        out[p] = (e2 / x2) ** 0.5
    return out

def shuffled_errors(CUT, seed):
    """CONTROL: within every row, the errors (stored minus cut) put on other numbers of the same row"""
    rng = np.random.default_rng(seed); C2 = {}
    for n in NAMES:
        A = ORIG[n]; RA = rows_of(n, A); RC = rows_of(n, CUT[n]); OUTR = np.empty(RA.shape, np.float32)
        for r0 in range(0, RA.shape[0], 4096):              # a block of rows at a time (memory)
            R = RA[r0:r0 + 4096].astype(np.float64); E = R - RC[r0:r0 + 4096].astype(np.float64)
            perm = np.argsort(rng.random(E.shape), axis=1)
            OUTR[r0:r0 + 4096] = (R - np.take_along_axis(E, perm, 1)).astype(np.float32)
        C2[n] = np.ascontiguousarray(from_rows(n, OUTR, A.shape)); del OUTR
    return C2

def install(CUT, parts=None):
    for n in NAMES:
        src = CUT[n] if (parts is None or part(n) in parts) else ORIG[n]
        PARAMS[n].data.copy_(torch.from_numpy(np.ascontiguousarray(src)))

# ---------------- the words ----------------
WINDOWS = json.load(open(f"{HERE}/judge5_texts.json")); REFS = json.load(open(f"{HERE}/refs_many.json"))
WIN_IDS = torch.tensor([w["ids"] for w in WINDOWS], dtype=torch.long)
REF5_B2 = np.load(f"{HERE}/judge5_ref.npz")["top5"]                 # the base-2 machine's own top 5 (for the uncut line)
RECORD_IN = [(r["ids"] + r["ref"])[:5] for r in REFS]
def run_words():
    T5 = []
    for i in range(0, 300, 25):
        lg = MODEL(WIN_IDS[i:i + 25]).logits[:, 3:23, :]
        T5.append(torch.topk(lg, 5, dim=-1).indices.numpy())
    T5 = np.concatenate(T5)                                        # (300, 20, 5), sorted
    gen = []
    for r in REFS:
        seq = list(r["ids"]); g = []
        for _ in range(16):
            nxt = int(MODEL(torch.tensor([seq])).logits[0, -1].argmax()); seq.append(nxt); g.append(nxt)
        gen.append(g)
    rec = [int(MODEL(torch.tensor([ids])).logits[0, -1].argmax()) for ids in RECORD_IN]
    return T5, gen, rec

def score(T5, gen, rec, REF):
    rT5, rgen, rrec = REF
    a = T5.reshape(-1, 5); f = rT5.reshape(-1, 5)
    full_in = int((a == f[:, :1]).any(1).sum()); ver_in = int((f == a[:, :1]).any(1).sum()); same = int((a[:, 0] == f[:, 0]).sum())
    want = [r["ref"] for r in REFS]
    words = sum(int(x == y) for g, w in zip(gen, want) for x, y in zip(g, w))
    texts = sum(g == w for g, w in zip(gen, want))
    first = sum(next((i for i, (x, y) in enumerate(zip(g, w)) if x != y), 16) for g, w in zip(gen, want))
    record = sum(int(x == y) for x, y in zip(rec, rrec))
    return dict(full_word_in_version_top5=full_in, version_word_in_full_top5=ver_in, same_top=same, judged=a.shape[0],
                greedy_words=words, greedy_texts=texts, greedy_before_first_diff=first, record_next_word=record)

VERSIONS = {"full": dict(kind="full"), "fp16": dict(kind="fp16"),
            **{f"tips_k{k}": dict(kind="tips", k=k) for k in (1, 2, 3)},
            **{f"grid_b{b}": dict(kind="grid", b=b) for b in (2, 3, 4, 5, 6)}}

def pc(a, b): return f"{100.0 * a / b:.2f}%"
def line(tag, s):
    return (f"{tag}: full word in its top 5 {s['full_word_in_version_top5']}/6000 ({pc(s['full_word_in_version_top5'], 6000)}); "
            f"its word in full top 5 {s['version_word_in_full_top5']} ({pc(s['version_word_in_full_top5'], 6000)}); same top word "
            f"{s['same_top']} ({pc(s['same_top'], 6000)}); greedy {s['greedy_words']}/480 words, {s['greedy_texts']}/30 texts, "
            f"{s['greedy_before_first_diff']} before first difference; record's next word {s['record_next_word']}/30")

if __name__ == "__main__":
    names = ARG["versions"].split(",") if "versions" in ARG else list(VERSIONS)
    RES = f"{OUT}/results.json"; res = json.load(open(RES)) if os.path.exists(RES) else {}
    log(f"=== WHOLETIPS {time.strftime('%Y-%m-%d %H:%M:%S')}  versions {names}  stored numbers {NV:,}  ({mb()})")
    wait_free()
    install(ORIG); REF = run_words()
    rT5, rgen, rrec = REF
    b2 = int((REF5_B2.reshape(-1, 5) == rT5.reshape(-1, 5)[:, :1]).any(1).sum())
    log(f"uncut fp32: greedy vs refs_many {sum(int(x == y) for g, r in zip(rgen, REFS) for x, y in zip(g, r['ref']))}/480; "
        f"its word in the base-2 machine's top 5 (judge5_ref) {b2}/6000; same top word as base-2 "
        f"{int((REF5_B2.reshape(-1, 5)[:, 0] == rT5.reshape(-1, 5)[:, 0]).sum())}/6000  ({int(time.time() - T0)}s, {mb()})")
    for name in names:
        wait_free(); spec = VERSIONS[name]; tc = time.time()
        CUT, SIZE = build(spec)
        bits_part = {p: sum(SIZE[n]["total_bits"] for n in NAMES if part(n) == p) for p in PARTS}
        tot_bits = sum(bits_part.values())
        row = dict(spec=spec, bits_a_number=tot_bits / NV, MB=tot_bits / 8e6,
                   bits_by_part={p: bits_part[p] / PART_NV[p] for p in PARTS},
                   table_and_scale_bits_a_number=sum(CELL * SIZE[n]["cells"] + SIZE[n]["scale_bits"] for n in NAMES) / NV,
                   changed=sum(int((CUT[n] != ORIG[n]).sum()) for n in NAMES) / NV, error_share=error_share(CUT))
        if spec["kind"] == "grid": row["plain_bits_a_number"] = sum(SIZE[n]["plain_bits"] for n in NAMES) / NV
        if spec["kind"] == "tips":
            row["bits_by_round"] = {key: sum(SIZE[n]["byround"][key] * SIZE[n]["n"] for n in NAMES) / NV for key in SIZE[NAMES[0]]["byround"]}
        log(f"[{name}] built ({int(time.time() - tc)}s, {mb()}): {row['bits_a_number']:.3f} bits a number, {row['MB']:.2f} MB; by part "
            + ", ".join(f"{p} {row['bits_by_part'][p]:.3f}" for p in PARTS)
            + f"; count tables and scales {row['table_and_scale_bits_a_number']:.4f}; changed {100 * row['changed']:.1f}%; error share "
            + ", ".join(f"{p} {row['error_share'][p]:.4f}" for p in PARTS)
            + (f"; by round {row['bits_by_round']}" if "bits_by_round" in row else "")
            + (f"; plain {row['plain_bits_a_number']:.3f}" if "plain_bits_a_number" in row else ""))
        install(CUT); row["whole"] = score(*run_words(), REF); log(line(f"[{name}] WHOLE", row["whole"]))
        if spec["kind"] in ("tips", "grid"):
            row["by_part"] = {}
            for p in PARTS:
                install(CUT, {p}); row["by_part"][p] = score(*run_words(), REF); log(line(f"[{name}] only {p}", row["by_part"][p]))
            C2 = shuffled_errors(CUT, 101); install(C2); del C2
            row["control_errors_shuffled_in_row"] = score(*run_words(), REF)
            log(line(f"[{name}] CONTROL errors shuffled within rows", row["control_errors_shuffled_in_row"]))
        install(ORIG); del CUT
        row["seconds"] = int(time.time() - tc); row["mem"] = mb()
        res[name] = row; json.dump(res, open(RES, "w"), indent=1)
        log(f"[{name}] done ({row['seconds']}s, {mb()})")
    log(f"=== done ({int(time.time() - T0)}s, {mb()})")
