/* TABLEDIFFUSE family: a binary range coder whose chances learn by COUNTING (whole numbers only).
   Round one's coder (develop_rc.c) nudges every chance by 1/32 of the gap after every bit: it never settles, and its
   jitter costs 2.6-3.3% over the per-line floor on the developed tables. Here each node of each context keeps its
   chance at 2^16 and a count n: the nudge is 1/(n + 2) of the gap (a table read of 2^16 / (n + 2)), so the chance is
   the running share of zeros -- a counter -- until n reaches NMAX, after which it keeps a slow 1/(NMAX + 2) nudge.
   A 32-bit range is cut by the chance of a 0: bound = (range >> 16) x p. Symbols of nbits places go top place first
   down a bit tree; contexts are whole numbers the reader has before it reads the symbol (the line). No float.
   cc -O2 -shared -o tablediffuse_rc.dylib tablediffuse_rc.c                                                    */
#include <stdint.h>
#include <stdlib.h>

#define TOPV (1u << 24)
#define PMIN 32
#define PMAX (65536 - 32)

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

static int32_t RATE[65536];
static void rates(int nmax) { for (int n = 0; n <= nmax; n++) RATE[n] = (int32_t)(65536 / (n + 2)); }

static inline void learn(uint16_t *p, uint16_t *n, int bit, int nmax) {
    int32_t P = *p, r = RATE[*n];
    P += (int32_t)(((int64_t)((bit ? 0 : 65536) - P) * r) >> 16);
    if (P < PMIN) P = PMIN; if (P > PMAX) P = PMAX;
    *p = (uint16_t)P; if (*n < nmax) (*n)++;
}

static inline void enc_bit(Enc *e, uint16_t *p, uint16_t *n, int bit, int nmax) {
    uint32_t bound = (e->range >> 16) * (*p);
    if (!bit) e->range = bound; else { e->low += bound; e->range -= bound; }
    learn(p, n, bit, nmax);
    while (e->range < TOPV) { e->range <<= 8; shift_low(e); }
}

/* sym[i] in [0, 2^nbits) (nbits <= 16), ctx[i] in [0, nctx). Returns bytes written (or needed, if > cap). */
int64_t rc2_encode(const uint16_t *sym, const uint16_t *ctx, int64_t n, int nbits, int64_t nctx, int nmax,
                   uint8_t *out, int64_t cap) {
    rates(nmax);
    int64_t tree = (int64_t)1 << nbits;
    uint16_t *probs = (uint16_t *)malloc(sizeof(uint16_t) * nctx * tree);
    uint16_t *cnt = (uint16_t *)calloc(nctx * tree, sizeof(uint16_t));
    for (int64_t i = 0; i < nctx * tree; i++) probs[i] = 32768;
    Enc e = {0, 0xFFFFFFFFu, 0, 1, out, 0, cap};
    for (int64_t i = 0; i < n; i++) {
        int64_t base = (int64_t)ctx[i] * tree; uint32_t s = sym[i]; uint32_t node = 1;
        for (int b = nbits - 1; b >= 0; b--) {
            int bit = (s >> b) & 1; enc_bit(&e, probs + base + node, cnt + base + node, bit, nmax);
            node = (node << 1) | bit;
        }
    }
    for (int i = 0; i < 5; i++) shift_low(&e);
    free(probs); free(cnt);
    return e.pos;
}

int64_t rc2_decode(const uint8_t *in, int64_t inlen, const uint16_t *ctx, int64_t n, int nbits, int64_t nctx, int nmax,
                   uint16_t *sym) {
    rates(nmax);
    int64_t tree = (int64_t)1 << nbits;
    uint16_t *probs = (uint16_t *)malloc(sizeof(uint16_t) * nctx * tree);
    uint16_t *cnt = (uint16_t *)calloc(nctx * tree, sizeof(uint16_t));
    for (int64_t i = 0; i < nctx * tree; i++) probs[i] = 32768;
    uint32_t range = 0xFFFFFFFFu, code = 0; int64_t pos = 0;
    for (int i = 0; i < 5; i++) { code = (code << 8) | (pos < inlen ? in[pos] : 0); pos++; }
    for (int64_t i = 0; i < n; i++) {
        int64_t base = (int64_t)ctx[i] * tree; uint32_t node = 1;
        for (int b = nbits - 1; b >= 0; b--) {
            uint16_t *p = probs + base + node; uint32_t bound = (range >> 16) * (*p); int bit;
            if (code < bound) { range = bound; bit = 0; } else { code -= bound; range -= bound; bit = 1; }
            learn(p, cnt + base + node, bit, nmax);
            while (range < TOPV) { range <<= 8; code = (code << 8) | (pos < inlen ? in[pos] : 0); pos++; }
            node = (node << 1) | bit;
        }
        sym[i] = (uint16_t)(node - (uint32_t)tree);
    }
    free(probs); free(cnt);
    return pos;
}
