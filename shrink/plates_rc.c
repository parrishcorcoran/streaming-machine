/* PLATES family coder. A plate is written one weight at a time, lines in (rows) outer, neurons out (columns) inner.
   Each weight is written as its photons in the single-wave writing (up = +1, hole = -1, never two lit side by side),
   the way a fax codes a line as runs: where the top photon is, its sign, then for each further photon "is there
   another one?" and how many dark places until it, and its sign. Nothing else is stored -- only the lit places.

   One adaptive binary range coder, whole numbers only: shifts, adds, subtracts, ANDs, compares, and the coder's one
   whole-number product (range >> 16) * chance. The chance of a dark/lit answer lives in a context made only of what
   has already been written:
     - the top place is read against two exposure meters: a running average of the top places along this line
       (the line's brightness) and one per neuron down its column (the neuron's brightness), each updated by a shift;
     - the further photons: how many photons so far, the place of the last one, how far below it we are.
       (the place of the last photon matters near the top: inside one octave the values are not spread evenly)
   After each answer the chance moves toward it by a shift: 1/2, 1/4, 1/8 ... as the context sees more, never slower
   than 1/2^LIM. Encoder and decoder run the SAME function (dec = 0 / 1), so they see the same contexts.
   ctx = 0 turns the meters off (one context for the top place): the control for "do the meters see anything".

   cc -O2 -shared -o plates_rc.dylib plates_rc.c
*/
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int LIM = 11;
void set_lim(int l) { LIM = l; }
typedef struct { uint32_t p; uint32_t n; } Prob;          /* p = chance of a dark answer (0), out of 2^28 */
typedef struct {
    int dec;
    uint64_t low; uint32_t range; uint8_t cache; uint64_t csize;
    uint32_t code;
    uint8_t *buf; int64_t pos, cap, len;
} RC;

static void enc_shift(RC *r) {
    if ((uint32_t)r->low < 0xFF000000u || (r->low >> 32) != 0) {
        uint8_t carry = (uint8_t)(r->low >> 32), t = r->cache;
        do { if (r->pos < r->cap) r->buf[r->pos] = (uint8_t)(t + carry); r->pos++; t = 0xFF; } while (--r->csize != 0);
        r->cache = (uint8_t)(r->low >> 24);
    }
    r->csize++;
    r->low = (r->low & 0x00FFFFFFu) << 8;
}
static inline uint32_t in_byte(RC *r) { uint32_t b = r->pos < r->len ? r->buf[r->pos] : 0; r->pos++; return b; }

static inline int bit(RC *r, Prob *pr, int b) {
    uint32_t p16 = pr->p >> 12; if (p16 < 1) p16 = 1; if (p16 > 65535) p16 = 65535;
    uint32_t bound = (r->range >> 16) * p16;
    if (r->dec) {
        if (r->code < bound) { r->range = bound; b = 0; } else { r->code -= bound; r->range -= bound; b = 1; }
        while (r->range < (1u << 24)) { r->range <<= 8; r->code = (r->code << 8) | in_byte(r); }
    } else {
        if (!b) r->range = bound; else { r->low += bound; r->range -= bound; }
        while (r->range < (1u << 24)) { r->range <<= 8; enc_shift(r); }
    }
    int n = pr->n, sh = 1; while ((n + 2) >> (sh + 1)) sh++;   /* sh = bit length of (n+2) - 1 */
    if (sh > LIM) sh = LIM;
    if (pr->n < 65535) pr->n++;
    if (!b) pr->p += ((1u << 28) - pr->p) >> sh; else pr->p -= pr->p >> sh;
    return b;
}

#define NB 40
typedef struct {
    Prob tree[NB][32];
    Prob sign;
    Prob more[5][20];
    Prob gap[5][20][12];
    Prob psign[5][20][2];
} Model;

static void model_init(Model *m) {
    Prob *p = (Prob *)m; size_t n = sizeof(Model) / sizeof(Prob);
    for (size_t i = 0; i < n; i++) { p[i].p = 1u << 27; p[i].n = 0; }
}

/* single-wave digits of a (a >= 0): dig[i] in {-1,0,1}; returns top place + 1 (0 if a == 0) */
static int naf(uint32_t a, int *dig) {
    int i = 0, top = 0;
    for (int k = 0; k < 24; k++) dig[k] = 0;
    while (a) {
        if (a & 1) { int d = 2 - (int)(a & 3); dig[i] = d; a -= (uint32_t)d; top = i + 1; }
        a >>= 1; i++;
    }
    return top;
}

static int code_plate(RC *r, int32_t *v, int64_t R, int64_t C, int ctx) {
    Model *m = (Model *)malloc(sizeof(Model)); model_init(m);
    int32_t *cm = (int32_t *)malloc(sizeof(int32_t) * C);
    for (int64_t j = 0; j < C; j++) cm[j] = 12 << 4;
    int32_t rm = 12 << 4;
    int dig[24];
    for (int64_t i = 0; i < R; i++) {
        for (int64_t j = 0; j < C; j++) {
            int32_t val = r->dec ? 0 : v[i * C + j];
            int T = 0;
            if (!r->dec) T = naf((uint32_t)(val < 0 ? -val : val), dig);
            int nb = ctx ? ((rm + cm[j]) >> 4) : 0;
            if (nb < 0) nb = 0; if (nb >= NB) nb = NB - 1;
            int node = 1, Td = 0;
            for (int b = 4; b >= 0; b--) { int x = bit(r, &m->tree[nb][node], (T >> b) & 1); node = 2 * node + x; Td = 2 * Td + x; }
            T = Td;
            int32_t out = 0;
            if (T) {
                int sg = bit(r, &m->sign, val < 0);
                int p = T - 1, k = 1, prevs = 0; int32_t a = (int32_t)1 << p;
                while (p >= 2) {
                    int kk = k < 4 ? k : 4, has = 0;
                    if (!r->dec) for (int q = p - 2; q >= 0; q--) if (dig[q]) { has = 1; break; }
                    if (!bit(r, &m->more[kk][p], has)) break;
                    int q = p - 2, d = 0;
                    while (q > 0) {
                        if (bit(r, &m->gap[kk][p][d < 11 ? d : 11], r->dec ? 0 : (dig[q] != 0))) break;
                        q--; d++;
                    }
                    int s = bit(r, &m->psign[kk][p][prevs], r->dec ? 0 : (dig[q] < 0));
                    a += s ? -((int32_t)1 << q) : ((int32_t)1 << q);
                    prevs = s; p = q; k++;
                }
                out = sg ? -a : a;
            }
            if (r->dec) v[i * C + j] = out;
            else if (out != val) { free(m); free(cm); return -1; }
            rm += ((T << 4) - rm) >> 4;
            cm[j] += ((T << 4) - cm[j]) >> 3;
        }
    }
    free(m); free(cm);
    return 0;
}

int64_t plate_encode(int32_t *v, int64_t R, int64_t C, uint8_t *out, int64_t cap, int ctx) {
    RC r; memset(&r, 0, sizeof r); r.dec = 0; r.buf = out; r.cap = cap; r.range = 0xFFFFFFFFu; r.csize = 1;
    if (code_plate(&r, v, R, C, ctx) < 0) return -1;
    for (int i = 0; i < 5; i++) enc_shift(&r);
    return r.pos <= cap ? r.pos : -2;
}

int plate_decode(uint8_t *in, int64_t len, int64_t R, int64_t C, int32_t *v, int ctx) {
    RC r; memset(&r, 0, sizeof r); r.dec = 1; r.buf = in; r.len = len; r.range = 0xFFFFFFFFu;
    for (int i = 0; i < 5; i++) r.code = (r.code << 8) | in_byte(&r);
    return code_plate(&r, v, R, C, ctx);
}
