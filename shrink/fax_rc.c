/* Adaptive binary range coder for the FAX family (the JBIG idea: each pixel of a binary page is coded with the odds
   learned so far in its "context" -- a small whole number naming what is already known around it).
   Whole numbers only: shifts, adds, subtracts, compares, and the coder's one whole-number product (range>>16)*p.
   Odds are whole numbers: the chance the pixel is dark, out of 2^31 (the coder uses its top 16 bits). After each
   pixel the odds move toward what was seen by a shift: 1/4 of the way at first, then 1/8, 1/16 ... as the context
   sees more pixels (the shift is the bit length of its count, so the odds settle like a running average over
   everything the context has seen), never slower than 1/65536.

   ctx[i]  : the context the caller computed for pixel i (from places already known: planes above, line number ...)
   tbits   : 0, 1, 2 or 3 extra template bits taken from the same page, pixels already coded:
             bit0 = pixel i-1 (left), bit1 = pixel i-width (above), bit2 = pixel i-2
   Encoder and decoder see exactly the same contexts, so decoding is exact.
   cc -O2 -shared -o fax_rc.dylib fax_rc.c
*/
#include <stdint.h>
#include <stdlib.h>

typedef struct { uint64_t low; uint32_t range; uint8_t cache; uint64_t csize; uint8_t *out; int64_t pos; } Enc;

static void shift_low(Enc *e) {
    if ((uint32_t)e->low < 0xFF000000u || (e->low >> 32) != 0) {
        uint8_t carry = (uint8_t)(e->low >> 32), t = e->cache;
        do { e->out[e->pos++] = (uint8_t)(t + carry); t = 0xFF; } while (--e->csize != 0);
        e->cache = (uint8_t)(e->low >> 24);
    }
    e->csize++;
    e->low = (e->low & 0x00FFFFFFu) << 8;
}

static inline uint32_t SH(uint32_t n) { uint32_t b = 0; n += 1; while (n) { b++; n >>= 1; } b += 1; return b > 16 ? 16 : b; }

static inline uint32_t tmpl(const uint8_t *b, int64_t i, int64_t width, int tbits) {
    uint32_t t = 0;
    if (tbits >= 1) t |= (i >= 1 ? b[i - 1] : 0);
    if (tbits >= 2) t |= (uint32_t)(i >= width ? b[i - width] : 0) << 1;
    if (tbits >= 3) t |= (uint32_t)(i >= 2 ? b[i - 2] : 0) << 2;
    return t;
}

int64_t fax_encode(const uint8_t *bits, const uint32_t *ctx, int64_t n, uint32_t nctx, int tbits, int64_t width,
                   uint8_t *out) {
    uint64_t nc = (uint64_t)nctx << tbits;
    uint32_t *pp = malloc(nc * 4); uint16_t *cn = malloc(nc * 2);
    for (uint64_t c = 0; c < nc; c++) { pp[c] = 1u << 30; cn[c] = 0; }
    Enc e = {0, 0xFFFFFFFFu, 0, 1, out, 0};
    for (int64_t i = 0; i < n; i++) {
        uint64_t c = ((uint64_t)ctx[i] << tbits) | tmpl(bits, i, width, tbits);
        uint32_t p = pp[c] >> 15, sh = SH(cn[c]);
        if (p < 1) p = 1; if (p > 65535) p = 65535;
        if (cn[c] < 65535) cn[c]++;
        uint32_t bound = (e.range >> 16) * p;
        if (!bits[i]) { e.range = bound; pp[c] += (0x7FFFFFFFu - pp[c]) >> sh; }
        else { e.low += bound; e.range -= bound; pp[c] -= pp[c] >> sh; }
        while (e.range < (1u << 24)) { e.range <<= 8; shift_low(&e); }
    }
    for (int k = 0; k < 5; k++) shift_low(&e);
    free(pp); free(cn);
    return e.pos;
}

int64_t fax_decode(const uint8_t *in, int64_t inlen, const uint32_t *ctx, int64_t n, uint32_t nctx, int tbits,
                   int64_t width, uint8_t *bits) {
    uint64_t nc = (uint64_t)nctx << tbits;
    uint32_t *pp = malloc(nc * 4); uint16_t *cn = malloc(nc * 2);
    for (uint64_t c = 0; c < nc; c++) { pp[c] = 1u << 30; cn[c] = 0; }
    uint32_t range = 0xFFFFFFFFu, code = 0; int64_t pos = 0;
    for (int k = 0; k < 5; k++) code = (code << 8) | (pos < inlen ? in[pos++] : 0);
    for (int64_t i = 0; i < n; i++) {
        uint64_t c = ((uint64_t)ctx[i] << tbits) | tmpl(bits, i, width, tbits);
        uint32_t p = pp[c] >> 15, sh = SH(cn[c]);
        if (p < 1) p = 1; if (p > 65535) p = 65535;
        if (cn[c] < 65535) cn[c]++;
        uint32_t bound = (range >> 16) * p;
        if (code < bound) { range = bound; bits[i] = 0; pp[c] += (0x7FFFFFFFu - pp[c]) >> sh; }
        else { code -= bound; range -= bound; bits[i] = 1; pp[c] -= pp[c] >> sh; }
        while (range < (1u << 24)) { range <<= 8; code = (code << 8) | (pos < inlen ? in[pos++] : 0); }
    }
    free(pp); free(cn);
    return pos;
}
