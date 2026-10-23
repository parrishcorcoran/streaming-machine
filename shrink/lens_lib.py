"""LENS family, shared pieces. The top table T is the entry table E seen through a lens: the last water level's gain gf
on each line, the centring C (each word's DC blocked), then the whole table put on 16 bits with one power of two per word.
base2_convert.py:  head.w = cols16(C @ (gf * wte.T)),  head.b = stream(bfin @ wte.T),  wte = stream(wte) (2^16).
Every encode/decode step here is whole numbers: shifts, adds, subtracts, ANDs, table reads, long division done as
shift-and-subtract. The lens itself (768 gains) is written once as whole numbers from the converter's own gain, the same
way the converter wrote every other number once; after that no float is touched.
"""
import lzma, numpy as np
from safetensors import safe_open

NPZ = "/Users/abundancemachine/streaming-machine/base2_gpt2.npz"
ST = "/Users/abundancemachine/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors"


def load():
    M = np.load(NPZ)
    E = M["wte"]                                         # (50257, 768) int64 at 2^16
    T = np.ascontiguousarray(M["head.w"].T)              # (50257, 768) int16, word's power of two in TE
    return E, T, M["head.e"].copy(), M["head.b"].copy(), M["wpe"].copy()


def lens_float(name):                                    # the converter's own float lens, read once
    f = safe_open(ST, "np")
    return f.get_tensor(name).astype(np.float64)


def whole(x, k):                                         # a float written once as a whole number at 2^-k
    return np.round(x * 2.0 ** k).astype(np.int64)


def div_round(S, d):
    """round(S / d) for whole numbers S (int64 array) and a positive whole d, by restoring long division:
    shift a digit in, subtract d if it fits. Sign kept apart, half of d added first (round half up in magnitude)."""
    S = np.asarray(S, dtype=np.int64)
    neg = S < 0
    A = (np.abs(S) + (d >> 1)).astype(np.uint64)
    q = np.zeros_like(A); r = np.zeros_like(A); D = np.uint64(d)
    for b in range(63, -1, -1):
        r = (r << np.uint64(1)) | ((A >> np.uint64(b)) & np.uint64(1))
        fit = r >= D
        r = np.where(fit, r - D, r)
        q = q | (fit.astype(np.uint64) << np.uint64(b))
    q = q.astype(np.int64)
    return np.where(neg, -q, q)


def rshift_round(X, s):                                  # X / 2^s rounded (add half, arithmetic shift)
    return (X + (np.int64(1) << np.int64(s - 1))) >> np.int64(s)


def zigzag(v):                                           # signed -> unsigned: 0,-1,1,-2,2 -> 0,1,2,3,4 (shift, XOR)
    v = np.asarray(v, dtype=np.int64)
    return ((v << 1) ^ (v >> 63)).astype(np.uint64)


def unzigzag(z):
    z = np.asarray(z, dtype=np.uint64)
    return ((z >> np.uint64(1)).astype(np.int64)) ^ (-(z & np.uint64(1)).astype(np.int64))


def pack(v, preset=6):
    """signed whole numbers -> bytes: zigzag, split into byte planes (low byte first), each plane lzma'd on its own.
    Returns (list of compressed planes, header). Header = shape + plane count (counted as side information)."""
    v = np.asarray(v)
    z = zigzag(v.ravel())
    top = int(z.max()) if z.size else 0
    nb = max(1, (top.bit_length() + 7) // 8)
    blobs = [lzma.compress(((z >> np.uint64(8 * b)) & np.uint64(255)).astype(np.uint8).tobytes(), preset=preset)
             for b in range(nb)]
    return blobs, (v.shape, nb)


def unpack(blobs, header):
    shape, nb = header
    z = np.zeros(int(np.prod(shape)), dtype=np.uint64)
    for b, blob in enumerate(blobs):
        z |= np.frombuffer(lzma.decompress(blob), dtype=np.uint8).astype(np.uint64) << np.uint64(8 * b)
    return unzigzag(z).reshape(shape)


def nbytes(blobs, header=None):
    hdr = 0 if header is None else 4 * (len(header[0]) + 1)
    return sum(len(b) for b in blobs) + hdr


def mb(n): return n / 1e6


_POP = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.int64)


def ones(A):                                             # photons (1s) per value, sign apart (same as table_bits.ones)
    A = np.abs(np.asarray(A).astype(np.int64)); n = np.zeros(A.shape, np.int64)
    while A.any(): n += _POP[A & 0xFFFF]; A = A >> 16
    return n


def naf(A): A = np.abs(np.asarray(A).astype(np.int64)); return ones(A ^ (3 * A))


def order0_bits(v):                                      # Shannon floor per value, own histogram (reference only)
    _, c = np.unique(np.asarray(v).ravel(), return_counts=True); p = c / c.sum(); return float(-(p * np.log2(p)).sum())


def pack_u(z, preset=6):                                 # unsigned whole numbers -> byte planes, each lzma'd
    z = np.asarray(z).astype(np.uint64)
    top = int(z.max()) if z.size else 0
    nb = max(1, (top.bit_length() + 7) // 8)
    return [lzma.compress(((z.ravel() >> np.uint64(8 * b)) & np.uint64(255)).astype(np.uint8).tobytes(), preset=preset)
            for b in range(nb)], (z.shape, nb)


def unpack_u(blobs, header):
    shape, nb = header
    z = np.zeros(int(np.prod(shape)), dtype=np.uint64)
    for b, blob in enumerate(blobs):
        z |= np.frombuffer(lzma.decompress(blob), dtype=np.uint8).astype(np.uint64) << np.uint64(8 * b)
    return z.reshape(shape)
