"""CHECKER, own code: the lead's reverse light at each newest bend output, exactly, with own roundings and own constants.
Uses only machine.py's recorded forward (record) and the machine's tables; the reverse, the residues, the meetings and
the split are written here from the forward's own arithmetic:
   top      score_w = round(pile_w x EXP2[f_w] x 2^(I_w-16)) + standing amount, x16 = round(R_t x 2^-sh)
   plain    out = round(pile x 2^-s) + bias, x16 = round(X x 2^-sh)
   gauged   out = round(pile x EXP2[f] x 2^(I-16)) + bias, x16 = round(R x 2^-sh)
   bend     out = round(H x share x 2^-16)
   arrival  out = round(num x EXP2[f] x 2^(I-16)), num_p = sum_j a_pj v_j
Each rounding's residue (out minus the exact value) is met with the light at its own step and gathered in c.
Checks per text: bottom (embedding numbers met with the light + c == lead) and at every layer
   lead == [newest 3,072 parts] + [earlier positions' bend outputs met with light] + [Rmid met with light above] + c_now
    python3 reverse/film_chk_split.py [n_texts]   -> reverse/film_chk_split.npz
"""
import sys, time; sys.path.insert(0, "/Users/abundancemachine/streaming-machine/reverse")
import numpy as np
np.seterr(all="ignore")
from fractions import Fraction
from machine import record, machine_scores, TEXTS, W, HEADW, HEADE, HEADB, EXP2, race, word

def obj(a): return np.asarray(a).astype(np.int64).astype(object) if np.asarray(a).dtype != object else np.asarray(a)
def pw(d):
    d = np.asarray(d, np.int64); return np.array([1 << int(x) for x in d.ravel()], dtype=object).reshape(d.shape)

class Lt:                                              # light: N (object whole numbers) x 2^E
    def __init__(s, N, E): s.N, s.E = N, int(E)
def scale(G, m, k):
    k = np.broadcast_to(np.asarray(k, np.int64), G.N.shape); k0 = int(k.min())
    return Lt(G.N * obj(np.broadcast_to(np.asarray(m, np.int64), G.N.shape)) * pw(k - k0), G.E + k0)
def add(A, B):
    E = min(A.E, B.E); return Lt(A.N * (1 << (A.E - E)) + B.N * (1 << (B.E - E)), E)
def limbs(N):
    flat = N.ravel(); mags = [abs(int(x)) for x in flat]; nb = max(1, max(m.bit_length() for m in mags)); K = -(-nb // 24)
    u = np.frombuffer(b"".join(m.to_bytes(3 * K, "little") for m in mags), np.uint8).reshape(len(flat), K, 3).astype(np.int64)
    lm = u[:, :, 0] + (u[:, :, 1] << 8) + (u[:, :, 2] << 16)
    sg = np.array([(x > 0) - (x < 0) for x in flat], np.int64)
    return (lm * sg[:, None]).T.reshape((K,) + N.shape).astype(np.float64)
def plate(G, wT):                                      # light on the plate's input = light on its pile @ w.T, exact
    Lm = limbs(G.N); K = Lm.shape[0]; P = G.N.shape[0]
    out = (Lm.reshape(K * P, -1) @ wT).reshape(K, P, wT.shape[1]); assert np.abs(out).max() < 2.0 ** 53
    oi = out.astype(np.int64); acc = oi[K - 1].astype(object)
    for k in range(K - 2, -1, -1): acc = acc * (1 << 24) + oi[k].astype(object)
    return Lt(acc, G.E)
def meet(G, V, k=0):                                   # sum of light x V x 2^k, exact Fraction
    k = np.broadcast_to(np.asarray(k, np.int64), G.N.shape); k0 = int(k.min())
    tot = (G.N * obj(np.broadcast_to(V, G.N.shape)) * pw(k - k0)).sum()
    return Fraction(int(tot)) * Fraction(2) ** (G.E + k0)
def residue(out, X, m, k):
    """out - X x m x 2^k as (numerator, exponent T): out x 2^-T - X x m x 2^(k-T); also checks |residue| <= 1/2"""
    k = np.broadcast_to(np.asarray(k, np.int64), np.shape(out)); T = min(int(k.min()), 0)
    rN = obj(out) * (1 << -T) - obj(X) * obj(np.broadcast_to(np.asarray(m, np.int64), np.shape(out))) * pw(k - T)
    assert all(2 * abs(int(x)) <= (1 << -T) for x in rN.ravel()), "a rounding over half a unit"
    return rN, T
def meet_res(G, out, X, m, k):
    rN, T = residue(out, X, m, k); return meet(G, rN, T)

def run(i, verbose):
    ids = TEXTS[i]; rec = record(ids); S = machine_scores(ids); assert (rec["top"]["scores"] == S).all()
    order = np.argsort(-S, kind="stable"); a, b = int(order[0]), int(order[1]); assert a == race(S)
    lead = int(S[a]) - int(S[b]); P, t = rec["P"], rec["t"]; tp = rec["top"]
    # ---- top
    x16 = tp["x16"]; sh = int(tp["sh"])
    rows, c = [], Fraction(0)
    for w_, sg in ((a, 1), (b, -1)):
        F = (int(HEADE[w_]) + sh) * 65536 - int(tp["LS"][0]); I, f = F >> 16, F & 0xFFFF
        pile = int((x16.astype(object) * HEADW[:, w_].astype(np.int64).astype(object)).sum())
        exact = Fraction(pile * int(EXP2[f])) * Fraction(2) ** (I - 16)
        c += sg * (int(HEADB[w_]) + (int(S[w_]) - int(HEADB[w_]) - exact))
        rows.append((sg, HEADW[:, w_].astype(np.int64) * int(EXP2[f]), I - 16))
    E0 = min(r[2] for r in rows)
    Nt = sum(obj(r[1]) * (r[0] * (1 << (r[2] - E0))) for r in rows)
    N = np.zeros((P, 768), dtype=object); N[:] = 0; N[t] = Nt; G = Lt(N, E0)          # light on the top's x16
    Z = np.zeros((P, 768), np.int64); X16 = Z.copy(); X16[t] = x16; RT = Z.copy(); RT[t] = tp["Rt"]
    SH = np.zeros((P, 768), np.int64); SH[t] = -sh
    c += meet_res(G, X16, RT, 1, SH)
    G = scale(G, 1, SH)                                                                 # light on the stream after layer 11
    ok_top = meet(G, rec["R12"]) + c == lead
    lights = np.zeros((12, 3072), dtype=object); parts = np.zeros((12, 3072), dtype=object); Ebo = np.zeros(12, np.int64); oks = []; partsum = np.zeros(12)
    for l in range(11, -1, -1):
        Ly = rec["layers"][l]; Rabove = rec["layers"][l + 1]["R"] if l < 11 else rec["R12"]
        ok_stream = meet(G, Rabove) + c == lead
        # mproj (plain bank)
        mp = Ly["mproj"]; w, we, bb = W[mp["name"]]
        c += meet(G, np.broadcast_to(bb, (P, 768)))
        c += meet_res(G, mp["out"] - bb, mp["pile"], 1, -mp["s"])
        Gx = plate(scale(G, 1, -mp["s"]), np.ascontiguousarray(w.T))
        c += meet_res(Gx, mp["x16"], Ly["bend"]["out"], 1, -mp["sh"][:, None] * np.ones((1, 3072), np.int64))
        Gbo = scale(Gx, 1, -mp["sh"][:, None] * np.ones((1, 3072), np.int64))
        Bout = Ly["bend"]["out"]
        pn = Gbo.N[t] * obj(Bout[t]); parts[l] = pn; Ebo[l] = Gbo.E; lights[l] = Gbo.N[t]
        Bz = Bout.copy(); Bz[t] = 0
        newest = Fraction(int(pn.sum())) * Fraction(2) ** Gbo.E
        tot = newest + meet(Gbo, Bz) + meet(G, Ly["Rmid"]) + c
        oks.append(bool(tot == lead) and bool(ok_stream)); partsum[l] = float(newest)
        if verbose:
            D = -Gbo.E
            for fr in (meet(Gbo, Bz), meet(G, Ly["Rmid"]), c): D = max(D, fr.denominator.bit_length() - 1)
            sN = int(pn.sum()) << (Gbo.E + D); rest = (tot - newest) * (1 << D); assert rest.denominator == 1
            print(f"   layer {l:2d}: newest part {float(newest):12.1f} ({float(newest) / lead:+.3f} leads); over 2^{D}: "
                  f"parts {sN} + rest {int(rest)} = {sN + int(rest)}, lead x 2^{D} = {lead << D}: "
                  f"{'EXACT' if sN + int(rest) == lead << D else 'DIFFERENT'}; stream above + c == lead: {ok_stream}", flush=True)
        # bend (shares held)
        bd = Ly["bend"]
        c += meet_res(Gbo, Bout, bd["H"], bd["share"], -16)
        GH = scale(Gbo, bd["share"], -16)
        # fc (gauged)
        fc = Ly["fc"]; w, we, bb = W[fc["name"]]; I_, f_ = fc["F"] >> 16, fc["F"] & 0xFFFF
        c += meet(GH, np.broadcast_to(bb, (P, 3072)))
        c += meet_res(GH, fc["out"] - bb, fc["pile"], EXP2[f_], I_ - 16)
        Gq = plate(scale(GH, EXP2[f_], I_ - 16), np.ascontiguousarray(w.T))
        shq = -fc["sh"][:, None] * np.ones((1, 768), np.int64)
        c += meet_res(Gq, fc["x16"], Ly["Rmid"], 1, shq)
        G = add(G, scale(Gq, 1, shq))                                                    # light on Rmid
        # proj (plain)
        pj = Ly["proj"]; w, we, bb = W[pj["name"]]
        c += meet(G, np.broadcast_to(bb, (P, 768)))
        c += meet_res(G, pj["out"] - bb, pj["pile"], 1, -pj["s"])
        Gx = plate(scale(G, 1, -pj["s"]), np.ascontiguousarray(w.T))
        shp = -pj["sh"][:, None] * np.ones((1, 768), np.int64)
        c += meet_res(Gx, pj["x16"], pj["X"], 1, shp)
        GA = scale(Gx, 1, shp)
        # arrival (shares held)
        ar = Ly["arr"]; Fa = np.repeat(-ar["L"].T, 64, axis=1); Ia, fa = Fa >> 16, Fa & 0xFFFF
        c += meet_res(GA, ar["out"], ar["num"], EXP2[fa], Ia - 16)
        Gn = scale(GA, EXP2[fa], Ia - 16)
        V = np.zeros((P, 768), dtype=object); V[:] = 0
        for h in range(12):
            cs = slice(h * 64, (h + 1) * 64)
            for j in range(P):
                for p in range(j, P):
                    av = int(ar["A"][h, p, j])
                    if av: V[j, cs] = V[j, cs] + Gn.N[p, cs] * av
        Gv = Lt(V, Gn.E)
        # qkv value columns (gauged)
        q = Ly["qkv"]; w, we, bb = W[q["name"]]; Fv = q["F"][:, 1536:]; Iv, fv = Fv >> 16, Fv & 0xFFFF
        c += meet(Gv, np.broadcast_to(bb[1536:], (P, 768)))
        c += meet_res(Gv, q["out"][:, 1536:] - bb[1536:], q["pile"][:, 1536:], EXP2[fv], Iv - 16)
        Gr = plate(scale(Gv, EXP2[fv], Iv - 16), np.ascontiguousarray(w[:, 1536:].T))
        shr = -q["sh"][:, None] * np.ones((1, 768), np.int64)
        c += meet_res(Gr, q["x16"], Ly["R"], 1, shr)
        G = add(G, scale(Gr, 1, shr))
    ok_bottom = meet(G, rec["R0"]) + c == lead
    return dict(lights=lights, a=a, b=b, lead=lead, top5=order[:5], parts=parts, Ebo=Ebo, oks=oks[::-1], ok_top=ok_top,
                ok_bottom=ok_bottom, partsum=partsum, Bout=np.stack([rec["layers"][l]["bend"]["out"][t] for l in range(12)]),
                share=np.stack([rec["layers"][l]["bend"]["share"][t] for l in range(12)]), H=np.stack([rec["layers"][l]["fc"]["out"][t] for l in range(12)]))

if __name__ == "__main__":
    NT = int(sys.argv[1]) if len(sys.argv) > 1 else 30; t0 = time.time()
    out = dict(parts50=np.zeros((NT, 12, 3072), np.int64), light50=np.zeros((NT, 12, 3072), np.int64),
               Bout=np.zeros((NT, 12, 3072), np.int64), share=np.zeros((NT, 12, 3072), np.int64), H=np.zeros((NT, 12, 3072), np.int64),
               ans=np.zeros(NT, np.int64), second=np.zeros(NT, np.int64), top5=np.zeros((NT, 5), np.int64), lead=np.zeros(NT, np.int64),
               partsum=np.zeros((NT, 12)), lightE=np.zeros((NT, 12), np.int64))
    allok = 0
    for i in range(NT):
        r = run(i, verbose=i < 2)
        for l in range(12):
            pn = r["parts"][l]; mb = max(abs(int(x)).bit_length() for x in pn); out["parts50"][i, l] = [int(x) >> max(mb - 50, 0) for x in pn]
            g = r["lights"][l]; gb = max(abs(int(x)).bit_length() for x in g); out["light50"][i, l] = [int(x) >> max(gb - 50, 0) for x in g]
        out["Bout"][i], out["share"][i], out["H"][i] = r["Bout"], r["share"], r["H"]
        out["ans"][i], out["second"][i], out["top5"][i], out["lead"][i], out["partsum"][i] = r["a"], r["b"], r["top5"], r["lead"], r["partsum"]
        ok = all(r["oks"]) and r["ok_top"] and r["ok_bottom"]; allok += ok
        print(f" text {i:2d} {word(r['a'])!r:13s} lead {r['lead']:7d}: top {r['ok_top']}, 12 layer splits exact {sum(r['oks'])}/12, "
              f"bottom {r['ok_bottom']}  ({time.time() - t0:.0f}s)", flush=True)
    np.savez("/Users/abundancemachine/streaming-machine/reverse/film_chk_split.npz", **out)
    print(f"all checks exact on {allok}/{NT} texts  ({time.time() - t0:.0f}s)")
