"""The soup, built from the all-base-2 machine's own parts (base2.py, loaded the way table_bits.py loads it), keeping
every number it makes so it can be watched: the stream R at all 25 snapshots (the entry and after each of the 24
writes), the whole numbers that meet every bank, every bank's piles, the bend's inputs and outputs, and the arrivals.

Nothing here changes the machine: every product, shift, table read and race is base2.py's own function. Two steps
are done for every position at once instead of one position at a time, and both are checked bit for bit against
base2.py's own code (check()):
  - the race in the arrival runs on every row together: the same digit-by-digit drop-out, where the winner's digit
    at each place is 1 exactly when anyone still in has a 1 there;
  - each head's arrivals meet the values as one product over all rows (a row's arrivals past its own position are 0).

Photons: the 1s of a value's size (sign apart), np.bitwise_count. A meeting is one light value meeting one weight;
its work is (light photons) x (weight photons). A bank's work for one position is therefore
    sum over lines i of photons(x_i) x (photons of line i's weights, summed over every neuron it meets)
which is computed exactly in whole numbers.
"""
import sys, numpy as np
HERE = "/Users/abundancemachine/streaming-machine/"
_src = open(HERE + "base2.py").read(); _src = _src[:_src.index("tok = GPT2Tokenizer")]
_argv = sys.argv; sys.argv = ["x"]; exec(_src); sys.argv = _argv

WTE = M["wte"].copy()                                 # the entry table, read once (base2.forward reads it from the file each call)
WPE = M["wpe"][:64].copy()
BANKS = [f"{k}.{l}" for l in range(12) for k in ("qkv", "proj", "fc", "mproj")] + ["head"]
bc = lambda X: np.bitwise_count(np.asarray(X)).astype(np.int64)   # photons of |x|, per value


def _rowph():                                         # per line in: photons of its weights over all neurons it meets
    return {n: np.concatenate([bc(W[n][0][i:i+64].astype(np.int64)).sum(1) for i in range(0, W[n][0].shape[0], 64)]) for n in BANKS}
ROWPH = _rowph()
WPH = {n: int(ROWPH[n].sum()) for n in BANKS}


def relabel_lines(seed=11):
    """Control (a): a random order for every line and neuron (fixed seed), moved consistently through the whole machine
    (the stream's 768 lines, each head's 64 lines for query/key and for value, the 3,072 neurons of every layer).
    The words read at the top are not relabelled. Every count must come out identical."""
    global WTE, WPE, ROWPH, WPH
    g = np.random.default_rng(seed)
    s = g.permutation(768)                            # the stream's lines
    WTE = WTE[:, s].copy(); WPE = WPE[:, s].copy()
    for l in range(12):
        tau = [g.permutation(64) for _ in range(12)]; rho = [g.permutation(64) for _ in range(12)]
        qk = np.concatenate([h * 64 + tau[h] for h in range(12)]); vv = np.concatenate([h * 64 + rho[h] for h in range(12)])
        cols = np.concatenate([qk, 768 + qk, 1536 + vv])
        w, e, b = W[f"qkv.{l}"]; W[f"qkv.{l}"] = (w[s][:, cols].copy(), e[cols].copy(), b[cols].copy())
        w, e, b = W[f"proj.{l}"]; W[f"proj.{l}"] = (w[vv][:, s].copy(), e[s].copy(), b[s].copy())
        pi = g.permutation(3072)
        w, e, b = W[f"fc.{l}"]; W[f"fc.{l}"] = (w[s][:, pi].copy(), e[pi].copy(), b[pi].copy())
        w, e, b = W[f"mproj.{l}"]; W[f"mproj.{l}"] = (w[pi][:, s].copy(), e[s].copy(), b[s].copy())
    w, e, b = W["head"]; W["head"] = (w[s].copy(), e, b)
    ROWPH = _rowph(); WPH = {n: int(ROWPH[n].sum()) for n in BANKS}


def race_rows(vals, alive):
    """base2.race on every row at once: digits from the top; at each digit, if anyone still in has a 1, everyone with a
    0 drops out. Returns the winner's value per row (its digits are the places where anyone still in had a 1)."""
    u = vals.astype(np.int64).view(np.uint64) ^ np.uint64(1 << 63)
    alive = alive.copy(); win = np.zeros(len(u), dtype=np.uint64)
    for bit in range(63, -1, -1):
        has = alive & (((u >> np.uint64(bit)) & np.uint64(1)) == 1)
        anyone = has.any(1)
        alive = np.where(anyone[:, None], has, alive)
        win |= np.where(anyone, np.uint64(1) << np.uint64(bit), np.uint64(0))
    return (win ^ np.uint64(1 << 63)).view(np.int64)


def arrival(QKV, l, rec):
    P = QKV.shape[0]; out = np.zeros((P, 768), dtype=np.int64); low = np.tril(np.ones((P, P), dtype=bool))
    for h in range(12):
        q, k, v = QKV[:, h*64:(h+1)*64], QKV[:, 768+h*64:768+(h+1)*64], QKV[:, 1536+h*64:1536+(h+1)*64]
        q16, eq = to16(q, -ES); k16, ek = to16(k, -ES)
        sc = rshift(product(q16, k16.T.astype(np.float64)), -(eq[:, None] + ek[None, :] - 3 + SC))
        top = race_rows(sc, low)
        d = top[:, None] - sc
        a = np.where(low & (d < len(ARR)), ARR[np.clip(d, 0, len(ARR) - 1)], 0)
        num = product(a, v.astype(np.float64))
        out[:, h*64:(h+1)*64] = power(num, -log2fix(a.sum(1))[:, None])
        rec["q16"][l, h] = q16; rec["k16"][l, h] = k16; rec["v"][l, h] = v; rec["a"][l, h] = a
    return out


def light(T, reads):
    """One pass of the whole machine over the words T (every layer, every position), reading the top at `reads`."""
    T = np.asarray(T); P = len(T)
    rec = dict(snaps=np.zeros((25, P, 768), np.int64), ins={}, piles={}, H=np.zeros((12, P, 3072), np.int64),
               G=np.zeros((12, P, 3072), np.int64), q16=np.zeros((12, 12, P, 64), np.int64), k16=np.zeros((12, 12, P, 64), np.int64),
               v=np.zeros((12, 12, P, 64), np.int64), a=np.zeros((12, 12, P, P), np.int64), gauge={})
    R = WTE[T] + WPE[:P]; rec["snaps"][0] = R
    for l in range(12):
        w, we, b = W[f"qkv.{l}"]; x16, ex, LS = gauge_log(R); pile = product(x16, w)
        rec["ins"][f"qkv.{l}"] = x16; rec["piles"][f"qkv.{l}"] = pile
        att = arrival(power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b, l, rec)
        w, we, b = W[f"proj.{l}"]; a16, ea = to16(att, -ES); pile = product(a16, w)
        rec["ins"][f"proj.{l}"] = a16; rec["piles"][f"proj.{l}"] = pile
        R = R + (rshift(pile, -(we[None, :] + ea[:, None] + ES)) + b); rec["snaps"][2 * l + 1] = R
        w, we, b = W[f"fc.{l}"]; x16, ex, LS = gauge_log(R); pile = product(x16, w)
        rec["ins"][f"fc.{l}"] = x16; rec["piles"][f"fc.{l}"] = pile; rec["gauge"][f"fc.{l}"] = (ex, LS)
        H = power(pile, (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b; G = edge(H)
        rec["H"][l] = H; rec["G"][l] = G
        w, we, b = W[f"mproj.{l}"]; g16, eg = to16(G, -ES); pile = product(g16, w)
        rec["ins"][f"mproj.{l}"] = g16; rec["piles"][f"mproj.{l}"] = pile
        R = R + (rshift(pile, -(we[None, :] + eg[:, None] + ES)) + b); rec["snaps"][2 * l + 2] = R
    reads = list(reads); hin = np.zeros((P, 768), np.int64)
    if reads:
        w, we, b = W["head"]; x16, ex, LS = gauge_log(R[reads])
        lg = power(product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None]) + b
        rec["tops"] = [race(lg[i]) for i in range(len(reads))]; hin[reads] = x16
    else:
        rec["tops"] = []
    rec["ins"]["head"] = hin
    return rec


def bank_work(rec):
    """(P, 49): photon meetings if each position's whole light is poured through each bank."""
    return np.stack([bc(rec["ins"][n]) @ ROWPH[n] for n in BANKS], 1)


def bank_delta(new, old, pair=None):
    """(P, 49): photon meetings if only the change of each bank's input is poured. pair: position q paired with p."""
    out = []
    for n in BANKS:
        o = old["ins"][n] if pair is None else old["ins"][n][pair]
        out.append(bc(new["ins"][n] - o) @ ROWPH[n])
    return np.stack(out, 1)


def lines_moved(new, old):
    return np.stack([(new["ins"][n] != old["ins"][n]).sum(1) for n in BANKS], 1)


def att_work(rec):
    """(P, 12): light-on-light meetings in the arrival, per layer (query x key for every earlier position, and every
    arrival x every value line)."""
    P = rec["a"].shape[-1]; low = np.tril(np.ones((P, P), dtype=np.int64)); out = np.zeros((P, 12), np.int64)
    for l in range(12):
        for h in range(12):
            pq, pk = bc(rec["q16"][l, h]), bc(rec["k16"][l, h])
            out[:, l] += ((pq @ pk.T) * low).sum(1) + bc(rec["a"][l, h]) @ bc(rec["v"][l, h]).sum(1)
    return out


def att_delta(new, old, pair=None):
    """(P, 12): the arrival's meetings if only changes are poured: q'k' - qk = dq k' + q dk, a'v' - av = da v' + a dv.
    pair: position q paired with p (shuffle control)."""
    P = new["a"].shape[-1]; low = np.tril(np.ones((P, P), dtype=bool)); out = np.zeros((P, 12), np.int64)
    pr = np.arange(P) if pair is None else pair
    for l in range(12):
        for h in range(12):
            q0, k0, v0 = old["q16"][l, h][pr], old["k16"][l, h][pr], old["v"][l, h][pr]
            a0 = old["a"][l, h][pr]
            q1, k1, v1, a1 = new["q16"][l, h], new["k16"][l, h], new["v"][l, h], new["a"][l, h]
            sc = bc(q1 - q0) @ bc(k1).T + bc(q0) @ bc(k1 - k0).T
            da = np.where(low, a1 - a0, 0)
            out[:, l] += (sc * low).sum(1) + bc(da) @ bc(v1).sum(1) + bc(np.where(low, a0, 0)) @ bc(v1 - v0).sum(1)
    return out


def oneword(prompt, N):
    """The plain machine: the prompt poured, then one word at a time; each new position poured once, the top read
    at the last position only. Returns the words, the poured positions' lights and their work."""
    T = list(prompt); S = len(T); bank = np.zeros((S + N - 1, 49), np.int64); att = np.zeros((S + N - 1, 12), np.int64)
    snaps = np.zeros((25, S + N - 1, 768), np.int64)
    for k in range(N):
        rec = light(T, [len(T) - 1])
        new = range(0, S) if k == 0 else [len(T) - 1]
        bw, aw = bank_work(rec), att_work(rec)
        for p in new: bank[p], att[p], snaps[:, p] = bw[p], aw[p], rec["snaps"][:, p]
        T.append(rec["tops"][0])
    return T[S:], bank, att, snaps


def check(T, S):
    """light() against base2.py's own functions (gauged_bank, plain_bank, attention, edge, race), bit for bit."""
    rec = light(T, list(range(S - 1, len(T) - 1)))
    P = len(T); R = WTE[T] + WPE[:P]; ok = True                 # WTE, WPE: copies of M["wte"], M["wpe"]
    for l in range(12):
        R = R + plain_bank(attention(gauged_bank(R, f"qkv.{l}")), f"proj.{l}"); ok &= np.array_equal(R, rec["snaps"][2 * l + 1])
        R = R + plain_bank(edge(gauged_bank(R, f"fc.{l}")), f"mproj.{l}"); ok &= np.array_equal(R, rec["snaps"][2 * l + 2])
    w, we, b = W["head"]; tops = []
    for p in range(S - 1, P - 1):
        x16, ex, LS = gauge_log(R[p:p+1])
        tops.append(race(power(product(x16, w), (we[None, :] + ex[:, None] + ES) * 65536 - LS[:, None])[0] + b))
    return ok and tops == rec["tops"]
