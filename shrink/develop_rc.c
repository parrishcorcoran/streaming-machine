/* DEVELOP family: an adaptive binary range coder in whole numbers only (the LZMA kind).
   A 32-bit range is cut in two by an 11-bit chance of a 0; the chance is nudged by a shift (1/32 of the gap)
   after every bit. A symbol of nbits places is sent top place first down a bit tree: each node of the tree,
   in each context, keeps its own chance. Contexts are whole numbers the reader already has before it reads
   the symbol (the line's number, the place of the photon before). No float anywhere.
   cc -O2 -shared -o develop_rc.dylib develop_rc.c                                                          */
#include <stdint.h>
#include <stdlib.h>

#define PB 11
#define MOVE 5
#define TOPV (1u << 24)

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

static inline void enc_bit(Enc *e, uint16_t *p, int bit) {
    uint32_t bound = (e->range >> PB) * (*p);
    if (!bit) { e->range = bound; *p += ((1u << PB) - *p) >> MOVE; }
    else { e->low += bound; e->range -= bound; *p -= *p >> MOVE; }
    while (e->range < TOPV) { e->range <<= 8; shift_low(e); }
}

/* sym[i] in [0, 2^nbits) (nbits <= 16), ctx[i] in [0, nctx) (nctx <= 65536). Returns bytes written (or needed, if > cap). */
int64_t rc_encode(const uint16_t *sym, const uint16_t *ctx, int64_t n, int nbits, int64_t nctx, uint8_t *out, int64_t cap) {
    int64_t tree = (int64_t)1 << nbits;
    uint16_t *probs = (uint16_t *)malloc(sizeof(uint16_t) * nctx * tree);
    for (int64_t i = 0; i < nctx * tree; i++) probs[i] = 1u << (PB - 1);
    Enc e = {0, 0xFFFFFFFFu, 0, 1, out, 0, cap};
    for (int64_t i = 0; i < n; i++) {
        uint16_t *pr = probs + (int64_t)ctx[i] * tree; uint32_t s = sym[i]; uint32_t node = 1;
        for (int b = nbits - 1; b >= 0; b--) { int bit = (s >> b) & 1; enc_bit(&e, pr + node, bit); node = (node << 1) | bit; }
    }
    for (int i = 0; i < 5; i++) shift_low(&e);
    free(probs);
    return e.pos;
}

int64_t rc_decode(const uint8_t *in, int64_t inlen, const uint16_t *ctx, int64_t n, int nbits, int64_t nctx, uint16_t *sym) {
    int64_t tree = (int64_t)1 << nbits;
    uint16_t *probs = (uint16_t *)malloc(sizeof(uint16_t) * nctx * tree);
    for (int64_t i = 0; i < nctx * tree; i++) probs[i] = 1u << (PB - 1);
    uint32_t range = 0xFFFFFFFFu, code = 0; int64_t pos = 0;
    for (int i = 0; i < 5; i++) { code = (code << 8) | (pos < inlen ? in[pos] : 0); pos++; }
    for (int64_t i = 0; i < n; i++) {
        uint16_t *pr = probs + (int64_t)ctx[i] * tree; uint32_t node = 1;
        for (int b = nbits - 1; b >= 0; b--) {
            uint16_t *p = pr + node; uint32_t bound = (range >> PB) * (*p); int bit;
            if (code < bound) { range = bound; *p += ((1u << PB) - *p) >> MOVE; bit = 0; }
            else { code -= bound; range -= bound; *p -= *p >> MOVE; bit = 1; }
            while (range < TOPV) { range <<= 8; code = (code << 8) | (pos < inlen ? in[pos] : 0); pos++; }
            node = (node << 1) | bit;
        }
        sym[i] = (uint16_t)(node - (uint32_t)tree);
    }
    free(probs);
    return pos;
}
