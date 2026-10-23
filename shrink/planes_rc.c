/* PLANES family: each place (bit plane) of a table is a black-and-white image, coded top place first, the way JBIG
   and JPEG2000 code bit planes. One adaptive binary range coder, whole numbers only: shifts, adds, subtracts, ANDs,
   compares, a leading-zero count, and the coder's one whole-number product (range >> 16) * chance.

   The chance a pixel is dark (0) lives in a context: a whole number made only of pixels already coded --
     - higher places of the same value: is it still dark above this place? if lit, how far below its top lit place,
       and is the place just above lit;
     - the same place / higher places of the neighbours: the line (column) number, how many of the 4 neighbours
       (left/right line in the word, word above/below in the line) are already lit, and two exposure meters:
       how many lines of this word, and how many words of this line, are already lit above this place.
   After each pixel the chance moves toward what was seen by a shift: 1/4 of the way at first, then 1/8, 1/16 ... as
   the context sees more pixels (shift = bit length of count + 2), never slower than 1/65536 -- a running average.
   Encoder and decoder run the SAME function (dec = 0 / 1), so they see the same contexts: decoding is exact.

   cc -O2 -shared -o planes_rc.dylib planes_rc.c
*/
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ the range coder ------------------------- */
typedef struct {
    int dec;
    uint64_t low; uint32_t range; uint8_t cache; uint64_t csize;   /* encoder */
    uint32_t code;                                                 /* decoder */
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
static void rc_init(RC *r, int dec, uint8_t *buf, int64_t cap_or_len) {
    memset(r, 0, sizeof *r); r->dec = dec; r->buf = buf; r->range = 0xFFFFFFFFu;
    if (dec) { r->len = cap_or_len; for (int k = 0; k < 5; k++) r->code = (r->code << 8) | in_byte(r); }
    else { r->cap = cap_or_len; r->csize = 1; }
}
static int64_t rc_finish(RC *r) { if (!r->dec) for (int k = 0; k < 5; k++) enc_shift(r); return r->pos; }

typedef struct { uint32_t p; uint32_t n; } Ctx;          /* p: chance of a 0, out of 2^31; n: pixels seen */
static inline uint32_t blen(uint32_t x) { return x ? 32u - (uint32_t)__builtin_clz(x) : 0u; }

static inline int rc_bit(RC *r, Ctx *c, int bit) {
    uint32_t p = c->p >> 15; if (p < 1) p = 1; if (p > 65535) p = 65535;
    uint32_t sh = blen(c->n + 2); if (sh > 16) sh = 16;
    if (c->n < 65535) c->n++;
    uint32_t bound = (r->range >> 16) * p;
    if (r->dec) bit = r->code >= bound;
    if (!bit) { r->range = bound; c->p += (0x7FFFFFFFu - c->p) >> sh; }
    else {
        if (r->dec) r->code -= bound; else r->low += bound;
        r->range -= bound; c->p -= c->p >> sh;
    }
    while (r->range < (1u << 24)) { r->range <<= 8; if (r->dec) r->code = (r->code << 8) | in_byte(r); else enc_shift(r); }
    return bit;
}
static Ctx *new_ctx(int64_t n) { Ctx *c = malloc(sizeof(Ctx) * n); for (int64_t i = 0; i < n; i++) { c[i].p = 1u << 30; c[i].n = 0; } return c; }

/* ------------------------------------------------------------------ modes ----------------------------------- */
#define M_HIGH      1   /* higher places of the same value (else: one chance per place for every pixel)          */
#define M_LINE      2   /* the line (column) number, for a value still dark above                                */
#define M_NEIGH     4   /* how many of the 4 neighbours are already lit (0..4), for a value still dark above     */
#define M_SIGNC     8   /* sign: context = line and the value's top lit place (else one chance for all signs)    */
#define M_SIGNN    16   /* sign: also the signs of the left and the upper neighbour (0 none, 1 up, 2 hole)        */
#define M_WEXP     32   /* exposure meter of the word: bit length of (lines of this word lit above this place)   */
#define M_LEXP     64   /* exposure meter of the line: bit length of (words of this line lit above this place)   */
#define M_SKIP    128   /* 64 x 64 blocks: the block's empty top places are side info and are not coded         */
#define M_SIGNLIT 256   /* sign coded the moment a value first lights (inside its place), not as a last plane    */
#define M_SIGNTOP 512   /* the +-1 writing: sign place first, as its own image, then magnitude m = v or -v-1;     */
                        /* the dark-above context also knows the sign                                            */
#define M_REFLINE 1024  /* the two places right below the top lit place also know their line                     */

/* sign context index for value i at top place tp (0..nb) */
static inline int64_t sign_ctx(int mode, const int32_t *A, const uint8_t *neg, const int8_t *top, int64_t i,
                               int64_t r, int64_t c, int64_t C, int nb, int tp, int known_all) {
    int64_t s = 0;
    if (mode & M_SIGNC) s = c * (nb + 1) + tp;
    if (mode & M_SIGNN) {
        int sl = 0, sa = 0;
        if (c > 0 && (known_all || top[i - 1] > 0) && (A ? A[i - 1] != 0 : top[i - 1] > 0)) sl = 1 + neg[i - 1];
        if (r > 0 && (known_all || top[i - C] > 0) && (A ? A[i - C] != 0 : top[i - C] > 0)) sa = 1 + neg[i - C];
        s = s * 9 + sl * 3 + sa;
    }
    return s;
}

/* Sign-magnitude planes (or the +-1 writing with M_SIGNTOP).
   dec = 0: A holds the values, buf receives bytes (cap = room). dec = 1: buf holds cap bytes, A receives the values.
   blocktop: per 64x64 block, number of places that may be lit (M_SKIP; side info the reader is given).
   mark[k]: bytes written when place k finished (mark[nb] after the sign plane / sign first).
   Returns bytes written (enc) or read (dec). */
int64_t planes(int dec, int32_t *A, int64_t R, int64_t C, int nb, int mode, const uint8_t *blocktop,
               uint8_t *buf, int64_t cap, int64_t *mark) {
    int64_t N = R * C, BC = (C + 63) / 64;
    int32_t *mag = malloc(sizeof(int32_t) * N); int8_t *top = calloc(N, 1); uint8_t *neg = calloc(N, 1);
    uint32_t *wc = calloc(R, 4), *wc0 = calloc(R, 4), *lc = calloc(C, 4), *lc0 = calloc(C, 4);
    int signtop = (mode & M_SIGNTOP) != 0;
    if (!dec) for (int64_t i = 0; i < N; i++) {
        int32_t v = A[i];
        if (signtop) { neg[i] = v < 0; mag[i] = v < 0 ? -v - 1 : v; } else { neg[i] = v < 0; mag[i] = v < 0 ? -v : v; }
    } else memset(mag, 0, sizeof(int32_t) * N);
    int64_t SL = (mode & M_LINE) ? C : 1, SN = (mode & M_NEIGH) ? 5 : 1, SW = (mode & M_WEXP) ? 12 : 1,
            SE = (mode & M_LEXP) ? 18 : 1, SS = signtop ? 2 : 1;
    int64_t nsig = (int64_t)nb * SL * SN * SW * SE * SS, nref = (int64_t)nb * 8 * 2 * ((mode & M_REFLINE) ? C : 1);
    int64_t nsgn = ((mode & M_SIGNC) ? C * (nb + 1) : 1) * ((mode & M_SIGNN) ? 9 : 1);
    Ctx *sig = new_ctx(nsig), *ref = new_ctx(nref), *sgn = new_ctx(nsgn);
    RC rc; rc_init(&rc, dec, buf, cap);

    if (signtop) {                                   /* the sign place first: its own image, line + neighbour signs */
        for (int64_t r = 0; r < R; r++) for (int64_t c = 0; c < C; c++) {
            int64_t i = r * C + c, s = 0;
            if (mode & M_SIGNC) s = c;
            if (mode & M_SIGNN) s = s * 9 + (c > 0 ? 1 + neg[i - 1] : 0) * 3 + (r > 0 ? 1 + neg[i - C] : 0);
            neg[i] = (uint8_t)rc_bit(&rc, sgn + s, neg[i]);
        }
        if (mark) mark[nb] = rc.pos;
    }
    for (int k = nb - 1; k >= 0; k--) {
        memcpy(wc0, wc, R * 4); memcpy(lc0, lc, C * 4);
        for (int64_t r = 0; r < R; r++) {
            const uint8_t *bt = blocktop ? blocktop + (r >> 6) * BC : 0;
            for (int64_t c = 0; c < C; c++) {
                int64_t i = r * C + c;
                if ((mode & M_SKIP) && k >= bt[c >> 6]) continue;          /* whole block dark at this place: known */
                int t = top[i]; Ctx *cx;
                if (!(mode & M_HIGH)) cx = sig + k;
                else if (t == 0) {
                    int64_t f = k;
                    if (mode & M_LINE) f = f * C + c;
                    if (mode & M_NEIGH) f = f * 5 + (c > 0 && top[i - 1]) + (c < C - 1 && top[i + 1]) + (r > 0 && top[i - C]) + (r < R - 1 && top[i + C]);
                    if (mode & M_WEXP) f = f * 12 + blen(wc0[r]);
                    if (mode & M_LEXP) f = f * 18 + blen(lc0[c]);
                    if (signtop) f = f * 2 + neg[i];
                    cx = sig + f;
                } else {
                    int d = t - 1 - k; int dd = d > 7 ? 7 : d; int ab = (mag[i] >> (k + 1)) & 1;
                    int64_t f = ((int64_t)k * 8 + dd) * 2 + ab;
                    if ((mode & M_REFLINE)) f = f * C + (d <= 2 ? c : 0);
                    cx = ref + f;
                }
                int bit = rc_bit(&rc, cx, (mag[i] >> k) & 1);
                if (bit) {
                    if (dec) mag[i] |= (int32_t)1 << k;
                    if (t == 0) {
                        top[i] = (int8_t)(k + 1); wc[r]++; lc[c]++;
                        if ((mode & M_SIGNLIT) && !signtop) {
                            int64_t s = sign_ctx(mode, 0, neg, top, i, r, c, C, nb, k + 1, 0);
                            neg[i] = (uint8_t)rc_bit(&rc, sgn + s, neg[i]);
                        }
                    }
                }
            }
        }
        if (mark) mark[k] = rc.pos;
    }
    if (!signtop && !(mode & M_SIGNLIT)) {           /* the sign plane last: only for values that are not zero */
        for (int64_t r = 0; r < R; r++) for (int64_t c = 0; c < C; c++) {
            int64_t i = r * C + c; if (!mag[i]) continue;
            int64_t s = sign_ctx(mode, 0, neg, top, i, r, c, C, nb, top[i], 1);
            neg[i] = (uint8_t)rc_bit(&rc, sgn + s, neg[i]);
        }
        if (mark) mark[nb] = rc.pos;
    }
    int64_t n = rc_finish(&rc);
    if (dec) for (int64_t i = 0; i < N; i++) A[i] = signtop ? (neg[i] ? -mag[i] - 1 : mag[i]) : (neg[i] ? -mag[i] : mag[i]);
    free(mag); free(top); free(neg); free(wc); free(wc0); free(lc); free(lc0); free(sig); free(ref); free(sgn);
    return n;
}

/* Single-wave writing (up / hole, no two lit places side by side): each place of v is dark, up (+1) or a hole (-1).
   A lit place right above forces this place dark (not coded). Per place: lit? then, if lit, up or hole?
   The top lit place's direction is the sign of v. nb = places of the single-wave writing (one more than plain). */
int64_t naf_planes(int dec, int32_t *A, int64_t R, int64_t C, int nb, int mode, uint8_t *buf, int64_t cap, int64_t *mark) {
    int64_t N = R * C;
    uint32_t *lit = calloc(N, 4), *hole = calloc(N, 4); int8_t *top = calloc(N, 1);
    uint32_t *wc = calloc(R, 4), *wc0 = calloc(R, 4), *lc = calloc(C, 4), *lc0 = calloc(C, 4);
    uint32_t *tl = 0, *th = 0;
    if (!dec) {                                       /* the target writing: up = (3a & ~a) >> 1, hole = (a & ~3a) >> 1 */
        tl = malloc(4 * N); th = malloc(4 * N);
        for (int64_t i = 0; i < N; i++) {
            int64_t v = A[i], a = v < 0 ? -v : v, up = ((3 * a) & ~a) >> 1, ho = (a & ~(3 * a)) >> 1;
            if (v < 0) { int64_t x = up; up = ho; ho = x; }
            tl[i] = (uint32_t)(up | ho); th[i] = (uint32_t)ho;
        }
    }
    int64_t SL = (mode & M_LINE) ? C : 1, SN = (mode & M_NEIGH) ? 5 : 1, SW = (mode & M_WEXP) ? 12 : 1, SE = (mode & M_LEXP) ? 18 : 1;
    int64_t nsig = (int64_t)nb * SL * SN * SW * SE, nref = (int64_t)nb * 8 * 2, ndir = (int64_t)nb * 8 * 2;
    int64_t nsgn = ((mode & M_SIGNC) ? C * (nb + 1) : 1) * ((mode & M_SIGNN) ? 9 : 1);
    Ctx *sig = new_ctx(nsig), *ref = new_ctx(nref), *dir = new_ctx(ndir), *sgn = new_ctx(nsgn);
    RC rc; rc_init(&rc, dec, buf, cap);
    for (int k = nb - 1; k >= 0; k--) {
        memcpy(wc0, wc, R * 4); memcpy(lc0, lc, C * 4);
        for (int64_t r = 0; r < R; r++) for (int64_t c = 0; c < C; c++) {
            int64_t i = r * C + c; int t = top[i]; Ctx *cx;
            if (t > 0 && ((lit[i] >> (k + 1)) & 1)) continue;          /* lit right above: this place is dark */
            if (!(mode & M_HIGH)) cx = sig + k;
            else if (t == 0) {
                int64_t f = k;
                if (mode & M_LINE) f = f * C + c;
                if (mode & M_NEIGH) f = f * 5 + (c > 0 && top[i - 1]) + (c < C - 1 && top[i + 1]) + (r > 0 && top[i - C]) + (r < R - 1 && top[i + C]);
                if (mode & M_WEXP) f = f * 12 + blen(wc0[r]);
                if (mode & M_LEXP) f = f * 18 + blen(lc0[c]);
                cx = sig + f;
            } else {
                int d = t - 1 - k, dd = d > 7 ? 7 : d; int a2 = (lit[i] >> (k + 2)) & 1;
                cx = ref + ((int64_t)k * 8 + dd) * 2 + a2;
            }
            int b = rc_bit(&rc, cx, dec ? 0 : (int)((tl[i] >> k) & 1));
            if (!b) continue;
            int h;
            if (t == 0) {                              /* the top lit place: its direction is the sign of v */
                int64_t s = 0;
                if (mode & M_SIGNC) s = c * (nb + 1) + k + 1;
                if (mode & M_SIGNN) {
                    int sl = (c > 0 && top[i - 1]) ? 1 + (int)((hole[i - 1] >> (top[i - 1] - 1)) & 1) : 0;
                    int sa = (r > 0 && top[i - C]) ? 1 + (int)((hole[i - C] >> (top[i - C] - 1)) & 1) : 0;
                    s = s * 9 + sl * 3 + sa;
                }
                h = rc_bit(&rc, sgn + s, dec ? 0 : (int)((th[i] >> k) & 1));
                top[i] = (int8_t)(k + 1); wc[r]++; lc[c]++;
            } else {                                   /* a lower lit place: same direction as the lit place above it? */
                int d = t - 1 - k, dd = d > 7 ? 7 : d;
                uint32_t above = lit[i] >> (k + 1); int j = __builtin_ctz(above) + k + 1;   /* nearest lit place above */
                int hp = (hole[i] >> j) & 1;
                h = rc_bit(&rc, dir + ((int64_t)k * 8 + dd) * 2 + hp, dec ? 0 : (int)((th[i] >> k) & 1));
            }
            lit[i] |= 1u << k; if (h) hole[i] |= 1u << k;
        }
        if (mark) mark[k] = rc.pos;
    }
    int64_t n = rc_finish(&rc);
    if (dec) for (int64_t i = 0; i < N; i++) A[i] = (int32_t)((int64_t)(lit[i] & ~hole[i]) - (int64_t)hole[i]);
    free(lit); free(hole); free(top); free(wc); free(wc0); free(lc); free(lc0); free(sig); free(ref); free(dir); free(sgn);
    if (tl) { free(tl); free(th); }
    return n;
}
