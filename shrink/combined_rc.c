/* COMBINED family: the entry table's coder. Round one's develop_rc nudges every chance by 1/32 of the gap after every
   bit, so its chances never settle (3.3% above the per-line floor at c=35). Here each node of each context keeps a
   count n of the answers it has seen, and its chance moves by 1/2, 1/4, 1/8, ... of the gap -- a running share,
   as a counter would keep it -- until the step reaches 1/2^LIM, then stays there (plates_rc's rule).
   Binary range coder, whole numbers only: shifts, adds, compares and one product (range >> 16) x chance.
   Symbols of nbits places go top place first down a bit tree; the context is the line (the reader has it).
   cc -O2 -shared -o combined_rc.dylib combined_rc.c                                                      */
#include <stdint.h>
#include <stdlib.h>

typedef struct { uint32_t p; uint32_t n; } Prob;            /* p = chance of a 0, out of 2^16 */
typedef struct { uint64_t low; uint32_t range; uint8_t cache; uint64_t csize; uint8_t *out; int64_t pos, cap; } Enc;

static void shift_low(Enc *e) {
    if ((uint32_t)e->low < 0xFF000000u || (e->low >> 32) != 0) {
        uint8_t carry = (uint8_t)(e->low >> 32), t = e->cache;
        do { if (e->pos < e->cap) e->out[e->pos] = (uint8_t)(t + carry); e->pos++; t = 0xFF; } while (--e->csize != 0);
        e->cache = (uint8_t)(e->low >> 24);
    }
    e->csize++;
    e->low = (e->low & 0x00FFFFFFu) << 8;
}
static inline void learn(Prob *pr, int bit, int lim) {
    int sh = 1; while (((pr->n + 2) >> (sh + 1)) && sh < lim) sh++;
    if (pr->n < 0xFFFFFFF) pr->n++;
    if (!bit) pr->p += (65536 - pr->p) >> sh; else pr->p -= pr->p >> sh;
    if (pr->p < 32) pr->p = 32; if (pr->p > 65504) pr->p = 65504;
}
static Prob *table(int64_t nctx, int nbits) {
    int64_t m = nctx << nbits; Prob *t = (Prob *)malloc(sizeof(Prob) * m);
    for (int64_t i = 0; i < m; i++) { t[i].p = 32768; t[i].n = 0; }
    return t;
}
int64_t crc_encode(const uint16_t *sym, const uint16_t *ctx, int64_t n, int nbits, int64_t nctx, int lim,
                   uint8_t *out, int64_t cap) {
    Prob *pr = table(nctx, nbits); Enc e = {0, 0xFFFFFFFFu, 0, 1, out, 0, cap};
    for (int64_t i = 0; i < n; i++) {
        Prob *t = pr + ((int64_t)ctx[i] << nbits); uint32_t s = sym[i], node = 1;
        for (int b = nbits - 1; b >= 0; b--) {
            int bit = (s >> b) & 1; Prob *p = t + node; uint32_t bound = (e.range >> 16) * p->p;
            if (!bit) e.range = bound; else { e.low += bound; e.range -= bound; }
            while (e.range < (1u << 24)) { e.range <<= 8; shift_low(&e); }
            learn(p, bit, lim); node = (node << 1) | bit;
        }
    }
    for (int i = 0; i < 5; i++) shift_low(&e);
    free(pr); return e.pos;
}
int64_t crc_decode(const uint8_t *in, int64_t inlen, const uint16_t *ctx, int64_t n, int nbits, int64_t nctx, int lim,
                   uint16_t *sym) {
    Prob *pr = table(nctx, nbits); uint32_t range = 0xFFFFFFFFu, code = 0; int64_t pos = 0;
    for (int i = 0; i < 5; i++) { code = (code << 8) | (pos < inlen ? in[pos] : 0); pos++; }
    for (int64_t i = 0; i < n; i++) {
        Prob *t = pr + ((int64_t)ctx[i] << nbits); uint32_t node = 1;
        for (int b = nbits - 1; b >= 0; b--) {
            Prob *p = t + node; uint32_t bound = (range >> 16) * p->p; int bit;
            if (code < bound) { range = bound; bit = 0; } else { code -= bound; range -= bound; bit = 1; }
            while (range < (1u << 24)) { range <<= 8; code = (code << 8) | (pos < inlen ? in[pos] : 0); pos++; }
            learn(p, bit, lim); node = (node << 1) | bit;
        }
        sym[i] = (uint16_t)(node - (1u << nbits));
    }
    free(pr); return pos;
}
