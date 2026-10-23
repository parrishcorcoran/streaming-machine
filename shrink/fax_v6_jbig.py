"""FAX family, version 6: the JBIG way. Every pixel of every place's page is coded (no position lists) by an adaptive
range coder (fax_rc.c, whole numbers only) with odds learned per context -- the context is what is already known:
  - the pages above (coded first, top place down): has this value lit higher up? how far above? (0 / 1 / 2 / 3+)
  - for a value still dark: its line (the column: 768 lines of a word / neurons of a plate) and how many of its four
    neighbours (left, right, above, below in the table) already lit higher up -- a local "exposure" of the page
  - two pixels of this same page already coded: the one to the left and the one above (the JBIG template)
Dark pixels in sparse pages cost almost nothing, lit pixels pay for their position; half-lit pages cost a bit a pixel.
Sign page last: context = its line, the value is zero or not, template left/above.
Decoded and checked equal.     python3 fax_v6_jbig.py E|T|fc.0 [shuffle=line|all] [orient=T] [flat=1] [refine=1]
  flat=1   : version 8a, flat field: each line's dark level (whole number, the mean of its first 2^m values by a shift,
             stored as side info) taken off before the pages are made -- a camera's column fixed-pattern correction
  refine=1 : version 8b, the two digits right below a value's top lit place are coded knowing their line
  shuffle=line : control, each line's values put in a random order (seed 0) -- same values per line, no neighbours
  shuffle=all  : control, all values of the table in a random order -- same values, no lines, no neighbours
"""
import sys, json, time, numpy as np
sys.path.insert(0, "/Users/abundancemachine/streaming-machine/shrink")
from fax_common import *

REFINE_LINE = False
def contexts(top, k, C):
    """top: (R, C) int8, top lit place + 1 found so far (0 = still dark above page k)."""
    lit = top > 0
    n4 = np.zeros(top.shape, np.uint8)
    n4[:, 1:] += lit[:, :-1]; n4[:, :-1] += lit[:, 1:]; n4[1:] += lit[:-1]; n4[:-1] += lit[1:]
    d = np.where(lit, np.minimum(top.astype(np.int16) - 1 - k, 3), 0).astype(np.uint32)
    line = np.broadcast_to(np.arange(C, dtype=np.uint32), top.shape)
    # still dark: (n4, line) -> 5*C contexts; lit above at depth d: (d, n4) -> 15 more
    ctx = np.where(d == 0, n4.astype(np.uint32) * C + line, 5 * C + (d - 1) * 5 + n4)
    if REFINE_LINE:                    # v8b: the digits 1 and 2 below the top lit place also know their line
        ctx = np.where((d == 1) | (d == 2), 5 * C + 15 + (d - 1) * C + line, ctx)
        return ctx.ravel(), 7 * C + 15
    return ctx.ravel(), 5 * C + 15

def sign_ctx(a, C):
    line = np.broadcast_to(np.arange(C, dtype=np.uint32), a.shape)
    return np.where(a != 0, line, C).ravel().astype(np.uint32), C + 1

def code(A):
    R, C = A.shape; a = np.abs(A); nb = nplanes(A); top = np.zeros(A.shape, np.int8); bufs = []
    for k in range(nb - 1, -1, -1):
        bit = ((a >> k) & 1).astype(np.uint8)
        ctx, nc = contexts(top, k, C)
        bufs.append(rc_encode(bit.ravel(), ctx, nc, tbits=2, width=C)); del ctx
        top[(bit == 1) & (top == 0)] = k + 1
    ctx, nc = sign_ctx(a, C)
    bufs.append(rc_encode((A < 0).ravel(), ctx, nc, tbits=2, width=C))
    return bufs, nb

def decode(bufs, nb, shape):
    R, C = shape; a = np.zeros(shape, np.int32); top = np.zeros(shape, np.int8)
    for i in range(nb):
        k = nb - 1 - i
        ctx, nc = contexts(top, k, C)
        bit = rc_decode(bufs[i], ctx, nc, tbits=2, width=C).reshape(shape); del ctx
        a |= bit.astype(np.int32) << k; top[(bit == 1) & (top == 0)] = k + 1
    ctx, nc = sign_ctx(a, C)
    neg = rc_decode(bufs[nb], ctx, nc, tbits=2, width=C).reshape(shape).astype(bool)
    a[neg] *= -1
    return a

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "E"
    opt = dict(x.split("=") for x in sys.argv[2:])
    A = load(name); tag = ""
    if opt.get("orient") == "T": A = np.ascontiguousarray(A.T); tag += " (transposed)"
    rng = np.random.default_rng(0)
    off = None
    if opt.get("flat"):                # v8a: each line's dark level (mean of its first 2^m values, a shift) taken off
        n2 = 1 << (A.shape[0].bit_length() - 1); sh = n2.bit_length() - 1
        off = ((A[:n2].astype(np.int64).sum(0) + (n2 >> 1)) >> sh).astype(np.int32)
        A = A - off[None, :]; tag += " flat field (dark level per line off)"
    if opt.get("refine"): REFINE_LINE = True; tag += " + line-aware digits below the top"
    if opt.get("shuffle") == "line":
        A = np.ascontiguousarray(rng.permuted(A, axis=0)); tag += " CONTROL: each line shuffled"
    elif opt.get("shuffle") == "all":
        A = rng.permutation(A.ravel()).reshape(A.shape); tag += " CONTROL: all shuffled"
    t = time.time()
    bufs, nb = code(A); nbytes = sum(len(b) for b in bufs); side = 2 * 32 + 5 + 64 * len(bufs)
    if off is not None: side += A.shape[1] * (int(np.abs(off).max()).bit_length() + 1)
    print(f"     coded ({time.time() - t:.0f} s); per page bits/value: " +
          " ".join(f"{len(b) * 8 / A.size:.3f}" for b in bufs), flush=True)
    B = decode(bufs, nb, A.shape)
    if off is not None: B = B + off[None, :]; A = A + off[None, :]
    ok = bool(np.array_equal(B, A)); del B
    res = [report(f"v6 JBIG contexts{tag}", name, nbytes, side, A.size, ok)]
    print(f"     ({time.time() - t:.0f} s)")
    json.dump(res, open(f"/Users/abundancemachine/streaming-machine/shrink/fax_out_v6_{name}{'_' + '_'.join(opt.values()) if opt else ''}.json", "w"))
