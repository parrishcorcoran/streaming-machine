"""A binary range coder in whole numbers only (the LZMA/JBIG kind: a 32-bit range cut in two by a 16-bit probability,
probabilities nudged by shifts). Contexts are small whole numbers; each context keeps its own probability."""

PB = 16                 # probability on 16 places
ADAPT = 6               # nudge by 1/64 of the gap each symbol
TOP = 1 << 24


def encode(bits, ctxs, nctx):
    probs = [1 << (PB - 1)] * nctx                         # P(bit = 0), starts at one half
    low, rng, cache, csize = 0, 0xFFFFFFFF, 0, 1
    out = bytearray()
    full = 1 << PB
    for b, c in zip(bits, ctxs):
        p = probs[c]
        bound = (rng >> PB) * p
        if b:
            low += bound; rng -= bound; probs[c] = p - (p >> ADAPT)
        else:
            rng = bound; probs[c] = p + ((full - p) >> ADAPT)
        while rng < TOP:
            rng <<= 8
            if low < 0xFF000000 or low >= (1 << 32):        # shift a byte out (carry kept in 'cache')
                carry = low >> 32; t = cache
                while True:
                    out.append((t + carry) & 0xFF); t = 0xFF; csize -= 1
                    if csize == 0: break
                cache = (low >> 24) & 0xFF
            csize += 1
            low = (low & 0x00FFFFFF) << 8
    for _ in range(5):
        if low < 0xFF000000 or low >= (1 << 32):
            carry = low >> 32; t = cache
            while True:
                out.append((t + carry) & 0xFF); t = 0xFF; csize -= 1
                if csize == 0: break
            cache = (low >> 24) & 0xFF
        csize += 1
        low = (low & 0x00FFFFFF) << 8
    return bytes(out)


def decode(data, ctxs, nctx):
    probs = [1 << (PB - 1)] * nctx
    full = 1 << PB
    rng, code, pos = 0xFFFFFFFF, 0, 0
    for _ in range(5):
        code = (code << 8) | data[pos]; pos += 1
    out = bytearray(len(ctxs))
    n = len(data)
    for i, c in enumerate(ctxs):
        p = probs[c]
        bound = (rng >> PB) * p
        if code < bound:
            rng = bound; probs[c] = p + ((full - p) >> ADAPT)
        else:
            code -= bound; rng -= bound; probs[c] = p - (p >> ADAPT); out[i] = 1
        while rng < TOP:
            rng <<= 8
            code = ((code << 8) | (data[pos] if pos < n else 0)) & 0xFFFFFFFF; pos += 1
    return out
