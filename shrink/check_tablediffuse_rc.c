/* CHECKER's own decoder for the TDF counting range coder (written from the format description, whole numbers only).
   Each context has a bit tree of 2^nbits nodes; each node a 16-bit chance of a 0 (start 32768) and a count n.
   After each bit the chance moves by (target - p) * floor(65536/(n+2)) >> 16, clamped to [32, 65504]; n stops at nmax.
   Range 32 bits, bound = (range >> 16) * p; normalise below 2^24 by reading a byte. First 5 bytes prime the code. */
#include <stdint.h>
#include <stdlib.h>

int64_t chk_decode(const uint8_t *in, int64_t inlen, const uint16_t *ctx, int64_t n, int nbits, int64_t nctx,
                   int nmax, uint16_t *sym) {
    int32_t *rate = (int32_t *)malloc(sizeof(int32_t) * (nmax + 1));
    for (int k = 0; k <= nmax; k++) rate[k] = 65536 / (k + 2);
    int64_t T = (int64_t)1 << nbits;
    uint16_t *p = (uint16_t *)malloc(2 * nctx * T);
    uint16_t *c = (uint16_t *)calloc(nctx * T, 2);
    for (int64_t i = 0; i < nctx * T; i++) p[i] = 32768;
    uint32_t range = 0xFFFFFFFFu, code = 0; int64_t pos = 0;
    for (int i = 0; i < 5; i++) { code = (code << 8) | (pos < inlen ? in[pos] : 0); pos++; }
    for (int64_t i = 0; i < n; i++) {
        if (ctx[i] >= nctx) { free(p); free(c); free(rate); return -1; }
        uint16_t *pp = p + (int64_t)ctx[i] * T, *cc = c + (int64_t)ctx[i] * T;
        uint32_t node = 1;
        for (int b = 0; b < nbits; b++) {
            uint32_t P = pp[node], bound = (range >> 16) * P; int bit;
            if (code < bound) { range = bound; bit = 0; } else { code -= bound; range -= bound; bit = 1; }
            int32_t Q = (int32_t)P, r = rate[cc[node]];
            Q += (int32_t)(((int64_t)((bit ? 0 : 65536) - Q) * r) >> 16);
            if (Q < 32) Q = 32; if (Q > 65504) Q = 65504;
            pp[node] = (uint16_t)Q; if (cc[node] < nmax) cc[node]++;
            while (range < (1u << 24)) { range <<= 8; code = (code << 8) | (pos < inlen ? in[pos] : 0); pos++; }
            node = (node << 1) | (uint32_t)bit;
        }
        sym[i] = (uint16_t)(node - (uint32_t)T);
    }
    free(p); free(c); free(rate);
    return pos;   /* bytes consumed (may run up to 4 past the end: the primer) */
}
