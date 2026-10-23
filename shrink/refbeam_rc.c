/* REFERENCE BEAM family: an adaptive binary range coder for whole numbers, written as photons.
   Each value v is written as: its top place k (the bit length of |v|, 0 for a dark value), its direction (up or a
   hole: the sign, only when lit), then the places below the top one, top first.
     - the top place k: 5 yes/no answers down a little tree, odds kept per context c
     - the direction:   one answer, odds kept per (c, k)
     - the 3 places just below the top: odds kept per (cm, k, the places above already said), cm = c or 0
     - the deeper places: odds kept per (k, place) -- they come out at about 1 bit each, as they should
   Odds are whole numbers (chance of a 0 out of 2^31; the coder uses the top 16 bits). After each answer they move
   toward what was seen by a shift: 1/4 of the way at first, then 1/8, 1/16 ... (the shift is the bit length of the
   context's count, so the odds settle into a running average), never slower than 1/2^13.
   Everything is shifts, adds, compares, and the coder's one whole-number product (range >> 16) * p.
   The context of value i:  c = (i / cdiv) % cmod, or ctx[i] when a context array is given.
   Encoder and decoder see the same contexts, so decoding is exact.
       cc -O2 -shared -o refbeam_rc.dylib refbeam_rc.c
*/
#include <stdint.h>
#include <stdlib.h>

typedef struct { uint64_t low; uint32_t range; uint8_t cache; uint64_t csize; uint8_t *out; int64_t pos; } Enc;
typedef struct { uint32_t range, code; const uint8_t *in; int64_t pos, len; } Dec;

static void shift_low(Enc *e) {
    if ((uint32_t)e->low < 0xFF000000u || (e->low >> 32) != 0) {
        uint8_t carry = (uint8_t)(e->low >> 32), t = e->cache;
        do { e->out[e->pos++] = (uint8_t)(t + carry); t = 0xFF; } while (--e->csize != 0);
        e->cache = (uint8_t)(e->low >> 24);
    }
    e->csize++;
    e->low = (e->low & 0x00FFFFFFu) << 8;
}

static inline uint32_t SH(uint32_t n) { uint32_t b = 0; n += 1; while (n) { b++; n >>= 1; } b += 1; return b > 13 ? 13 : b; }

typedef struct { uint32_t *p; uint16_t *n; } Odds;
static Odds odds_new(uint64_t k) {
    Odds o; o.p = malloc(k * 4); o.n = malloc(k * 2);
    for (uint64_t i = 0; i < k; i++) { o.p[i] = 1u << 30; o.n[i] = 0; }
    return o;
}
static void odds_free(Odds o) { free(o.p); free(o.n); }

static inline void ebit(Enc *e, Odds o, uint64_t c, int bit) {
    uint32_t p = o.p[c] >> 15, sh = SH(o.n[c]);
    if (p < 1) p = 1; if (p > 65535) p = 65535;
    if (o.n[c] < 65535) o.n[c]++;
    uint32_t bound = (e->range >> 16) * p;
    if (!bit) { e->range = bound; o.p[c] += (0x7FFFFFFFu - o.p[c]) >> sh; }
    else { e->low += bound; e->range -= bound; o.p[c] -= o.p[c] >> sh; }
    while (e->range < (1u << 24)) { e->range <<= 8; shift_low(e); }
}
static inline int dbit(Dec *d, Odds o, uint64_t c) {
    uint32_t p = o.p[c] >> 15, sh = SH(o.n[c]); int bit;
    if (p < 1) p = 1; if (p > 65535) p = 65535;
    if (o.n[c] < 65535) o.n[c]++;
    uint32_t bound = (d->range >> 16) * p;
    if (d->code < bound) { d->range = bound; bit = 0; o.p[c] += (0x7FFFFFFFu - o.p[c]) >> sh; }
    else { d->code -= bound; d->range -= bound; bit = 1; o.p[c] -= o.p[c] >> sh; }
    while (d->range < (1u << 24)) { d->range <<= 8; d->code = (d->code << 8) | (d->pos < d->len ? d->in[d->pos++] : 0); }
    return bit;
}
static inline uint32_t bitlen(uint64_t m) { uint32_t k = 0; while (m) { k++; m >>= 1; } return k; }

#define TOPN 3
int64_t rb_encode(const int64_t *v, int64_t n, const uint32_t *ctx, int64_t cdiv, uint32_t cmod, int mant_ctx,
                  uint8_t *out) {
    uint32_t nc = cmod, ncm = mant_ctx ? cmod : 1;
    Odds K = odds_new((uint64_t)nc * 32), S = odds_new((uint64_t)nc * 64),
         MT = odds_new((uint64_t)ncm * 64 * 8), ML = odds_new(64 * 64);
    Enc e = {0, 0xFFFFFFFFu, 0, 1, out, 0};
    for (int64_t i = 0; i < n; i++) {
        uint32_t c = ctx ? ctx[i] : (uint32_t)((i / cdiv) % cmod);
        int64_t x = v[i]; uint64_t m = x < 0 ? (uint64_t)(-x) : (uint64_t)x; uint32_t k = bitlen(m);
        uint32_t node = 1;
        for (int b = 4; b >= 0; b--) { int bit = (k >> b) & 1; ebit(&e, K, (uint64_t)c * 32 + node, bit); node = node * 2 + bit; }
        if (k == 0) continue;
        ebit(&e, S, (uint64_t)c * 64 + k, x < 0);
        uint32_t cm = mant_ctx ? c : 0, pre = 1;
        for (int j = (int)k - 2, t = 0; j >= 0; j--, t++) {
            int bit = (m >> j) & 1;
            if (t < TOPN) { ebit(&e, MT, ((uint64_t)cm * 64 + k) * 8 + pre, bit); pre = pre * 2 + bit; }
            else ebit(&e, ML, (uint64_t)k * 64 + j, bit);
        }
    }
    for (int k = 0; k < 5; k++) shift_low(&e);
    odds_free(K); odds_free(S); odds_free(MT); odds_free(ML);
    return e.pos;
}

int64_t rb_decode(const uint8_t *in, int64_t inlen, int64_t n, const uint32_t *ctx, int64_t cdiv, uint32_t cmod,
                  int mant_ctx, int64_t *v) {
    uint32_t nc = cmod, ncm = mant_ctx ? cmod : 1;
    Odds K = odds_new((uint64_t)nc * 32), S = odds_new((uint64_t)nc * 64),
         MT = odds_new((uint64_t)ncm * 64 * 8), ML = odds_new(64 * 64);
    Dec d = {0xFFFFFFFFu, 0, in, 0, inlen};
    for (int k = 0; k < 5; k++) d.code = (d.code << 8) | (d.pos < d.len ? d.in[d.pos++] : 0);
    for (int64_t i = 0; i < n; i++) {
        uint32_t c = ctx ? ctx[i] : (uint32_t)((i / cdiv) % cmod);
        uint32_t node = 1;
        for (int b = 4; b >= 0; b--) { int bit = dbit(&d, K, (uint64_t)c * 32 + node); node = node * 2 + bit; }
        uint32_t k = node - 32;
        if (k == 0) { v[i] = 0; continue; }
        int neg = dbit(&d, S, (uint64_t)c * 64 + k);
        uint32_t cm = mant_ctx ? c : 0, pre = 1; uint64_t m = 1;
        for (int j = (int)k - 2, t = 0; j >= 0; j--, t++) {
            int bit;
            if (t < TOPN) { bit = dbit(&d, MT, ((uint64_t)cm * 64 + k) * 8 + pre); pre = pre * 2 + bit; }
            else bit = dbit(&d, ML, (uint64_t)k * 64 + j);
            m = m * 2 + bit;
        }
        v[i] = neg ? -(int64_t)m : (int64_t)m;
    }
    odds_free(K); odds_free(S); odds_free(MT); odds_free(ML);
    return d.pos;
}
