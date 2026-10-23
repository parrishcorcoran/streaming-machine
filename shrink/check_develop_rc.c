/* CHECKER for the develop family: an independent coder (not develop_rc.c). A carry-less range coder (Subbotin kind)
   with adaptive symbol counts, one count table per context (the line). Whole numbers only: adds, subtracts, shifts,
   whole-number divides of the range by the total count, compares. Counts start at 1, grow by INC, halve past 2^16.
   cc -O2 -shared -o check_develop_rc.dylib check_develop_rc.c                                                  */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define TOP (1u << 24)
#define BOT (1u << 16)
#define INC 24
#define LIM (1u << 16)

typedef struct { uint32_t *f; uint32_t tot; } Model;

static Model *models_new(int64_t nctx, int K) {
    Model *m = (Model *)malloc(sizeof(Model) * nctx);
    for (int64_t c = 0; c < nctx; c++) {
        m[c].f = (uint32_t *)malloc(sizeof(uint32_t) * K);
        for (int k = 0; k < K; k++) m[c].f[k] = 1;
        m[c].tot = K;
    }
    return m;
}
static void models_free(Model *m, int64_t nctx) { for (int64_t c = 0; c < nctx; c++) free(m[c].f); free(m); }
static void bump(Model *m, int K, int s) {
    m->f[s] += INC; m->tot += INC;
    if (m->tot > LIM) { uint32_t t = 0; for (int k = 0; k < K; k++) { m->f[k] = (m->f[k] + 1) >> 1; t += m->f[k]; } m->tot = t; }
}

int64_t ck_encode(const uint16_t *sym, const uint16_t *ctx, int64_t n, int K, int64_t nctx, uint8_t *out, int64_t cap) {
    Model *m = models_new(nctx, K);
    uint32_t low = 0, range = 0xFFFFFFFFu; int64_t pos = 0;
    for (int64_t i = 0; i < n; i++) {
        Model *md = m + ctx[i]; int s = sym[i]; uint32_t cum = 0;
        for (int k = 0; k < s; k++) cum += md->f[k];
        range /= md->tot; low += cum * range; range *= md->f[s];
        while ((low ^ (low + range)) < TOP || (range < BOT && ((range = (0u - low) & (BOT - 1)), 1))) {
            if (pos < cap) out[pos] = (uint8_t)(low >> 24); pos++; low <<= 8; range <<= 8;
        }
        bump(md, K, s);
    }
    for (int i = 0; i < 4; i++) { if (pos < cap) out[pos] = (uint8_t)(low >> 24); pos++; low <<= 8; }
    models_free(m, nctx);
    return pos;
}

void ck_decode(const uint8_t *in, int64_t inlen, const uint16_t *ctx, int64_t n, int K, int64_t nctx, uint16_t *sym) {
    Model *m = models_new(nctx, K);
    uint32_t low = 0, range = 0xFFFFFFFFu, code = 0; int64_t pos = 0;
    for (int i = 0; i < 4; i++) { code = (code << 8) | (pos < inlen ? in[pos] : 0); pos++; }
    for (int64_t i = 0; i < n; i++) {
        Model *md = m + ctx[i];
        range /= md->tot;
        uint32_t v = (code - low) / range; if (v >= md->tot) v = md->tot - 1;
        int s = 0; uint32_t cum = 0;
        while (cum + md->f[s] <= v) { cum += md->f[s]; s++; }
        low += cum * range; range *= md->f[s];
        while ((low ^ (low + range)) < TOP || (range < BOT && ((range = (0u - low) & (BOT - 1)), 1))) {
            code = (code << 8) | (pos < inlen ? in[pos] : 0); pos++; low <<= 8; range <<= 8;
        }
        sym[i] = (uint16_t)s;
        bump(md, K, s);
    }
    models_free(m, nctx);
}
