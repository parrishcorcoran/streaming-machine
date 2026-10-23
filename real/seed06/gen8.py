"""gen8: pure-numpy, bit-exact reimplementation of torch <= 2.9 CPU `Tensor.normal_` on a contiguous bf16 tensor
(the 8-bit-uniform Box-Muller of aten/src/ATen/native/cpu/DistributionTemplates.h::normal_fill<BFloat16>).

No torch import.  Stream = numpy legacy MT19937 (init_genrand(seed & 0xffffffff)), identical to torch's CPU mt19937.

Layout (torch 2.6 source + real draws, see g1_map.log):
  call normal_(mean, std) on n >= 16 contiguous bf16 elements, starting at stream offset O
  (O = number of u32 outputs consumed since manual_seed):
    pass 1: element i gets uniform k_i/256, k_i = y[O+i] & 0xFF          (y = tempered mt19937 output)
    pass 2: for each full block b (elements 16b..16b+15), j in 0..7:
        k1 = k[16b+j]   (radius uniform, u1 = 1 - k1/256)
        k2 = k[16b+j+8] (angle  uniform, u2 = k2/256)
        out[16b+j]   = TC[k1,k2] = bf16(bf16(bf16(R(k1)*C(k2))*std16) + mean16)
        out[16b+j+8] = TS[k1,k2] = bf16(bf16(bf16(R(k1)*S(k2))*std16) + mean16)
    if n % 16 != 0: elements n-16..n-1 are overwritten by one more block fed by y[O+n .. O+n+15].
    consumption = n + 16*[n % 16 != 0]
  R(k1) = bf16(sqrt(bf16(-2*bf16(log(bf16(1-k1/256))))))   theta = bf16(float32(2*pi*k2/256))
  C = bf16(cos(theta)), S = bf16(sin(theta)), std16 = bf16(float32(std)), mean16 = bf16(float32(mean)).
  All bf16 roundings are round-to-nearest-even of the float32 result (c10 round_to_nearest_even).
"""
import numpy as np

TWO_PI_D = 2.0 * 3.141592653589793  # (2.0f * c10::pi<double>) evaluated in double


# ---------------------------------------------------------------- bf16 helpers (bit exact, c10 semantics)
def bf16_bits(x32):
    """float32 array -> bf16 bit patterns (uint16), round-to-nearest-even like c10::detail::round_to_nearest_even."""
    x32 = np.asarray(x32, dtype=np.float32)
    u = x32.view(np.uint32).astype(np.uint64)
    out = ((u + (((u >> 16) & 1) + 0x7FFF)) >> 16).astype(np.uint16)
    out = np.where(np.isnan(x32), np.uint16(0x7FC0), out)
    return out


def bf16_to_f32(b):
    return (np.asarray(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def rb(x32):
    """round float32 values to bf16, returned as float32."""
    return bf16_to_f32(bf16_bits(x32))


def f32(x):
    return np.asarray(x, dtype=np.float64).astype(np.float32)


# ---------------------------------------------------------------- the 256 x 256 output table
def radius_table():
    k = np.arange(256, dtype=np.float64)
    U = f32(k / 256)                                  # exact (8 bits)
    u1 = rb(np.float32(1.0) - U)                      # 1 - data[j] in BFloat16 arithmetic: exact
    L = rb(f32(np.log(u1.astype(np.float64))))        # std::log(BFloat16) -> bf16(logf(float))
    m2L = rb(np.float32(-2.0) * L)                    # -2 * BFloat16 -> BFloat16 (exact)
    R = rb(np.sqrt(m2L))                              # std::sqrt(BFloat16) -> bf16(sqrtf); sqrtf correctly rounded
    return R


def angle_tables():
    k = np.arange(256, dtype=np.float64)
    u2 = f32(k / 256)
    theta = rb(f32(TWO_PI_D * u2.astype(np.float64)))  # double -> float -> bf16
    C = rb(f32(np.cos(theta.astype(np.float64))))
    S = rb(f32(np.sin(theta.astype(np.float64))))
    return theta, C, S


_TABLE_CACHE = {}


def table(std=0.02, mean=0.0):
    """Returns (TC, TS): uint16 bf16 bit patterns [256 (k1), 256 (k2)]."""
    key = (float(std), float(mean))
    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key]
    R = radius_table()
    _, C, S = angle_tables()
    std16 = rb(f32(std))
    mean16 = rb(f32(mean))
    out = []
    for A in (C, S):
        p = rb(R[:, None] * A[None, :])               # radius * cos(theta): BFloat16*BFloat16
        p = rb(p * std16)                             # * std (BFloat16)
        p = rb(p + mean16)                            # + mean (BFloat16); turns -0 into +0 when mean = 0
        out.append(bf16_bits(p))
    _TABLE_CACHE[key] = tuple(out)
    return _TABLE_CACHE[key]


# ---------------------------------------------------------------- the stream
def mt_stream(seed, offset, count, chunk=1 << 24):
    """Raw tempered mt19937 outputs y[offset : offset+count] (uint32) for torch.manual_seed(seed)."""
    bg = np.random.MT19937(0)
    bg._legacy_seeding(int(seed) & 0xFFFFFFFF)
    left = int(offset)
    while left > 0:
        m = min(left, chunk)
        bg.random_raw(m)
        left -= m
    return bg.random_raw(int(count)).astype(np.uint32)


def consumed(n):
    """u32 outputs consumed by one torch<=2.9 bf16 normal_ call on n >= 16 contiguous elements."""
    return n + (16 if n % 16 else 0)


def from_stream(y, n, std=0.02, mean=0.0):
    """Apply the bf16 normal_fill to a given u32 stream y (len >= consumed(n)); returns uint16 bf16 bits."""
    if n < 16:
        raise NotImplementedError('n < 16 uses at::normal_distribution<double> (random64 + cached sample); not emulated')
    TC, TS = table(std, mean)
    k = (np.asarray(y[:consumed(n)], dtype=np.uint32) & 0xFF).astype(np.intp)
    out = np.empty(n, dtype=np.uint16)
    nb = n // 16
    blk = k[:16 * nb].reshape(nb, 16)
    o = out[:16 * nb].reshape(nb, 16)
    o[:, :8] = TC[blk[:, :8], blk[:, 8:]]
    o[:, 8:] = TS[blk[:, :8], blk[:, 8:]]
    if n % 16:
        kt = k[n:n + 16]
        out[n - 16:n - 8] = TC[kt[:8], kt[8:]]
        out[n - 8:n] = TS[kt[:8], kt[8:]]
    return out


def gen8(seed, offset, n, std=0.02, mean=0.0):
    """bf16 bits (uint16) of: torch.manual_seed(seed); <consume `offset` u32>; torch.empty(n, dtype=bf16).normal_(mean, std)
    under torch <= 2.9 CPU.  Use bf16_to_f32() for values."""
    y = mt_stream(seed, offset, consumed(n))
    return from_stream(y, n, std, mean)


# ---------------------------------------------------------------- element -> generator bits
def element_bits(t, P):
    """For element t (flat index inside a tensor whose element 0 sits at stream offset P; P and the tensor start must make
    t's block start = P + 16*(t//16), i.e. the tensor's own flat index is a multiple of 16 at its start; true for every
    full block of the call) return (u32 index of the radius output, u32 index of the angle output, sign spec).
    sign spec: ('b7^b6') if t%16 < 8 (cos element): [W_t < 0] = bit7(y[angle]) xor bit6(y[angle])
               ('b7')    if t%16 >= 8 (sin element): [W_t < 0] = bit7(y[angle])
    valid when W_t != 0 (k1 != 0) and the angle is not an axis angle (k2 not in {0,64,128,192}); axis pairs are detected
    by min/max(|W_j|,|W_{j+8}|) <= 0.00636 (all other angles >= 0.02295; threshold 0.012).
    bit b of y = (y >> b) & 1 of the TEMPERED 32-bit output (the value torch's mt19937() returns)."""
    t = np.asarray(t, dtype=np.int64)
    base = P + 16 * (t // 16)
    j = t % 8
    rad = base + j
    ang = base + 8 + j
    is_cos = (t % 16) < 8
    return rad, ang, is_cos


def sign_equations(w_bits, P=0):
    """Exact GF(2) equations from the signs of a bf16 block-aligned draw (or of bf16(c*draw), c > 0).
    Returns (u32_index, mask, value): parity(y[u32_index] & mask) == value.  mask 0x80: bit7, 0xC0: bit7^bit6,
    0x60: bit6^bit5 (|W_{j+8}| > |W_j|), 0x01..0x80 with value 0 for both-zero pairs (k1 == 0, radius low byte 0)."""
    w = bf16_to_f32(np.asarray(w_bits, np.uint16)).astype(np.float64).reshape(-1, 16)
    w1 = w[:, :8].ravel(); w2 = w[:, 8:].ravel()
    p = np.arange(w1.size); rad = P + 16 * (p // 8) + p % 8; ang = rad + 8
    both0 = (w1 == 0) & (w2 == 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.minimum(np.abs(w1), np.abs(w2)) / np.maximum(np.abs(w1), np.abs(w2))
    keep = ~both0 & (r > 0.012); extra = ~both0 & (r < 0.970)
    idx = [ang[keep], ang[keep], ang[extra]] + [rad[both0]] * 8
    msk = [np.full(keep.sum(), 0x80), np.full(keep.sum(), 0xC0), np.full(extra.sum(), 0x60)] + [np.full(both0.sum(), 1 << b) for b in range(8)]
    val = [(w2[keep] < 0), (w1[keep] < 0), (np.abs(w2[extra]) > np.abs(w1[extra]))] + [np.zeros(both0.sum(), bool)] * 8
    return np.concatenate(idx), np.concatenate(msk).astype(np.uint32), np.concatenate(val).astype(np.uint8)


def _parity(x):
    x = x.astype(np.uint32)
    x ^= x >> 16; x ^= x >> 8; x ^= x >> 4; x ^= x >> 2; x ^= x >> 1
    return (x & 1).astype(np.uint8)


if __name__ == '__main__':
    TC, TS = table()
    v = bf16_to_f32(np.concatenate([TC.ravel(), TS.ravel()]))
    print('table: distinct values', len(np.unique(v)), 'max|v|', np.abs(v).max())
    # self-test of element_bits / sign_equations on a draw at offset P (positive control) and a shifted stream (negative)
    seed, P, n = 2024, 123456, 5120 * 4
    y = mt_stream(seed, 0, P + n + 64)
    w = from_stream(y[P:], n)
    i, m, val = sign_equations(w, P)
    print('sign_equations self-test:', len(val), 'equations; hold', float(np.mean(_parity(y[i] & m) == val)) * 100, '% ; with y shifted by 1:',
          float(np.mean(_parity(y[i + 1] & m) == val)) * 100, '%')
    rad, ang, is_cos = element_bits(np.arange(n), P)
    k1 = y[rad] & 0xFF; k2 = y[ang] & 0xFF
    rebuilt = np.where(is_cos, TC[k1, k2], TS[k1, k2])
    print('element_bits self-test: value == table[k1(rad), k2(ang)] on', float(np.mean(rebuilt == w)) * 100, '% of', n)
