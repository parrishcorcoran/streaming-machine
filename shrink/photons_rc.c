/* PHOTONS family coder: store ONLY the photons (lit places) of developed whole numbers.
   An array (rows x cols, raster order) of whole numbers is written value by value, as its photons:
     writing 1 = single wave: up (+1) and holes (-1), never two lit places side by side;
     writing 0 = plain base 2: only up photons, the value's sign once.
   Four ways to write the photons down (all whole numbers: shifts, adds, compares, ANDs, table reads; the range
   coder's one product (range >> 16) * chance):
     coder 0  FAX + range coder: the place of the top photon, its sign, then for each further photon "another one?"
              and the run of dark places down to it (and its direction). Each answer is a dark/lit decision with an
              adaptive chance kept per context (what the reader already has).
     coder 1  ELIAS: no learning at all. Top place + 1 in Elias gamma (dark value = "1"), sign as a raw bit, every
              further photon as its distance from the photon above in Elias gamma ("1" = no more photons).
     coder 2  RICE (Golomb, the camera way of JPEG-LS): top place as its distance from the meter's reading, in a
              Golomb-Rice code whose k follows a running count (A/N, whole numbers); runs of dark places between
              photons in Rice codes with their own running k; signs raw.
     coder 4  PAGES: one page per place, from the top place down; see code_pages.
     coder 3  PATTERN: the value's whole photon pattern as ONE symbol (zigzag whole number) down a binary tree,
              each node with its own adaptive chance per context; rare big patterns escape to the fax photons.
   Contexts for the top place (all known to the reader before it reads the value):
     meters 1 = one bucket (row meter + column meter) >> 4  (round one's plate coder)
     meters 2 = two buckets: row meter and column meter apart (8 x 8)
     meters 3 = the row meter alone (half places); meters 4 = the column meter alone
     cls mode 1 = a class per column (e.g. the line of a word table: 768 classes)
     cls mode 2 = the exposure residual H - 2s of the plates (H = t_j + off_i - A, s = max(0, H >> 1)), clipped -3..1
   Encoder and decoder run the SAME function (dec = 0 / 1).                   cc -O2 -shared -o photons_rc.dylib photons_rc.c */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ adaptive binary range coder */
static int LIM = 11;
typedef struct { uint32_t p; uint32_t n; } Prob;
typedef struct {
    int dec; uint64_t low; uint32_t range; uint8_t cache; uint64_t csize; uint32_t code;
    uint8_t *buf; int64_t pos, cap, len;
} RC;
static void enc_shift(RC *r) {
    if ((uint32_t)r->low < 0xFF000000u || (r->low >> 32) != 0) {
        uint8_t carry = (uint8_t)(r->low >> 32), t = r->cache;
        do { if (r->pos < r->cap) r->buf[r->pos] = (uint8_t)(t + carry); r->pos++; t = 0xFF; } while (--r->csize != 0);
        r->cache = (uint8_t)(r->low >> 24);
    }
    r->csize++; r->low = (r->low & 0x00FFFFFFu) << 8;
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
    int n = pr->n, sh = 1; while ((n + 2) >> (sh + 1)) sh++;
    if (sh > LIM) sh = LIM;
    if (pr->n < 65535) pr->n++;
    if (!b) pr->p += ((1u << 28) - pr->p) >> sh; else pr->p -= pr->p >> sh;
    return b;
}
static void probs_init(Prob *p, size_t n) { for (size_t i = 0; i < n; i++) { p[i].p = 1u << 27; p[i].n = 0; } }

/* ------------------------------------------------------------------ raw bit writer / reader (Elias, Rice) */
typedef struct { int dec; uint8_t *buf; int64_t cap, len; int64_t bitpos; } BW;
static inline void put_bit(BW *w, int b) {
    int64_t byte = w->bitpos >> 3;
    if (byte < w->cap) { if (b) w->buf[byte] |= (uint8_t)(0x80u >> (w->bitpos & 7)); }
    w->bitpos++;
}
static inline int get_bit(BW *w) {
    int64_t byte = w->bitpos >> 3; int b = 0;
    if (byte < w->len) b = (w->buf[byte] >> (7 - (w->bitpos & 7))) & 1;
    w->bitpos++; return b;
}
static inline int rw_bit(BW *w, int b) { if (w->dec) return get_bit(w); put_bit(w, b); return b; }
static uint32_t rw_gamma(BW *w, uint32_t x) {          /* x >= 1: (len-1) zeros, then x in len places */
    if (w->dec) {
        int z = 0; while (!get_bit(w)) z++;
        uint32_t v = 1; for (int i = 0; i < z; i++) v = (v << 1) | (uint32_t)get_bit(w);
        return v;
    }
    int L = 0; uint32_t t = x; while (t) { L++; t >>= 1; }
    for (int i = 0; i < L - 1; i++) put_bit(w, 0);
    for (int i = L - 1; i >= 0; i--) put_bit(w, (x >> i) & 1);
    return x;
}
#define RICE_CAP 24
static uint32_t rw_rice(BW *w, uint32_t x, int k) {    /* unary of x >> k (ones then a zero), then k low places;
                                                          past RICE_CAP ones: an Elias gamma of the rest (escape) */
    if (w->dec) {
        uint32_t q = 0; while (q < RICE_CAP && get_bit(w)) q++;
        if (q == RICE_CAP) q += rw_gamma(w, 0) - 1;
        uint32_t lo = 0; for (int i = 0; i < k; i++) lo = (lo << 1) | (uint32_t)get_bit(w);
        return (q << k) | lo;
    }
    uint32_t q = x >> k;
    if (q < RICE_CAP) { for (uint32_t i = 0; i < q; i++) put_bit(w, 1); put_bit(w, 0); }
    else { for (int i = 0; i < RICE_CAP; i++) put_bit(w, 1); rw_gamma(w, q - RICE_CAP + 1); }
    for (int i = k - 1; i >= 0; i--) put_bit(w, (x >> i) & 1);
    return x;
}
typedef struct { uint32_t A, N; } RiceCtx;               /* running sum of coded numbers and their count */
static inline int rice_k(RiceCtx *c) { int k = 0; while (((uint64_t)c->N << k) < c->A) k++; return k; }
static inline void rice_upd(RiceCtx *c, uint32_t x) { c->A += x; c->N++; if (c->N >= 64) { c->A = (c->A + 1) >> 1; c->N >>= 1; } }

/* ------------------------------------------------------------------ photons of one value */
/* writing 1: dig[k] in {-1,0,1}; writing 0: dig[k] in {0,1}. Returns top place + 1 (0 for 0). */
static int digits(uint32_t a, int *dig, int wave) {
    int i = 0, top = 0;
    for (int k = 0; k < 34; k++) dig[k] = 0;
    if (!wave) { while (a) { if (a & 1) { dig[i] = 1; top = i + 1; } a >>= 1; i++; } return top; }
    while (a) {
        if (a & 1) { int d = 2 - (int)(a & 3); dig[i] = d; a -= (uint32_t)d; top = i + 1; }
        a >>= 1; i++;
    }
    return top;
}

#define NB 40
#define NCLS_MAX 1024
typedef struct {
    int coder, wave, meters, clsmode, ncls, A, signcls, lim, tbits, lowcls, rowreset;
} Opt;

typedef struct {
    /* coder 0 (and coder 3's escape) */
    Prob *top;            /* [ncls][NB2][32] */
    Prob *sign;           /* [ncls] */
    Prob more[2][5][34];
    Prob gap[2][5][34][12];
    Prob psign[2][5][34][2];
    /* coder 3 */
    Prob *tree;           /* [ncls][NB2][1 << tbits] */
    Prob esc[40][2];
    /* coder 2 */
    RiceCtx rtop[NCLS_MAX], rgap[5][34];
} Model;

static int nb2(const Opt *o) { return o->meters == 2 ? 64 : (o->meters >= 1 ? NB : 1); }

static int ctx_cls(const Opt *o, int64_t i, int64_t j, const int32_t *colcls, const int32_t *colterm, const int32_t *rowterm) {
    if (o->clsmode == 1) return colcls[j];
    if (o->clsmode == 2) {
        int H = colterm[j] + rowterm[i] - o->A; int s = H >> 1; if (s < 0) s = 0;
        int r = H - 2 * s; if (r < -3) r = -3; if (r > 1) r = 1; return r + 3;
    }
    if (o->clsmode == 3) {   /* both: column class x exposure residual */
        int H = colterm[j] + rowterm[i] - o->A; int s = H >> 1; if (s < 0) s = 0;
        int r = H - 2 * s; if (r < -3) r = -3; if (r > 1) r = 1; return colcls[j] * 5 + r + 3;
    }
    return 0;
}
static int ctx_meter(const Opt *o, int32_t rm, int32_t cm) {
    if (o->meters == 1) { int nb = (rm + cm) >> 4; if (nb < 0) nb = 0; if (nb >= NB) nb = NB - 1; return nb; }
    if (o->meters == 2) {
        int a = rm >> 4, b = cm >> 4;          /* each meter in whole places, 16ths kept above */
        if (a > 7) a = 7; if (b > 7) b = 7; if (a < 0) a = 0; if (b < 0) b = 0; return a * 8 + b;
    }
    if (o->meters == 3 || o->meters == 4) {   /* one meter alone, in half places: 3 = the row's, 4 = the column's */
        int nb = (o->meters == 3 ? rm : cm) >> 3; if (nb < 0) nb = 0; if (nb >= NB) nb = NB - 1; return nb;
    }
    return 0;
}

/* the further photons under the top (fax runs), shared by coders 0 and 3 */
static int32_t fax_lower(RC *r, Model *m, int wave, int lc, int T, int *dig) {
    int p = T - 1, k = 1, prevs = 0; int32_t a = (int32_t)1 << p;
    int minstep = wave ? 2 : 1;
    while (p >= minstep) {
        int kk = k < 4 ? k : 4, has = 0;
        if (!r->dec) for (int q = p - minstep; q >= 0; q--) if (dig[q]) { has = 1; break; }
        if (!bit(r, &m->more[lc][kk][p], has)) break;
        int q = p - minstep, d = 0;
        while (q > 0) {
            if (bit(r, &m->gap[lc][kk][p][d < 11 ? d : 11], r->dec ? 0 : (dig[q] != 0))) break;
            q--; d++;
        }
        int s = 0;
        if (wave) s = bit(r, &m->psign[lc][kk][p][prevs], r->dec ? 0 : (dig[q] < 0));
        a += s ? -((int32_t)1 << q) : ((int32_t)1 << q);
        prevs = s; p = q; k++;
    }
    return a;
}

static int code_array(RC *r, BW *w, int32_t *v, int64_t R, int64_t C, const Opt *o,
                      const int32_t *colcls, const int32_t *colterm, const int32_t *rowterm) {
    Model *m = (Model *)calloc(1, sizeof(Model));
    int NBc = nb2(o); int ncls = o->ncls < 1 ? 1 : o->ncls;
    m->top = (Prob *)malloc(sizeof(Prob) * (size_t)ncls * NBc * 32); probs_init(m->top, (size_t)ncls * NBc * 32);
    m->sign = (Prob *)malloc(sizeof(Prob) * (size_t)ncls); probs_init(m->sign, ncls);
    probs_init(&m->more[0][0][0], sizeof(m->more) / sizeof(Prob));
    probs_init(&m->gap[0][0][0][0], sizeof(m->gap) / sizeof(Prob));
    probs_init(&m->psign[0][0][0][0], sizeof(m->psign) / sizeof(Prob));
    probs_init(&m->esc[0][0], sizeof(m->esc) / sizeof(Prob));
    size_t tsz = (size_t)1 << (o->coder == 3 ? o->tbits : 0);
    if (o->coder == 3) { m->tree = (Prob *)malloc(sizeof(Prob) * (size_t)ncls * NBc * tsz); probs_init(m->tree, (size_t)ncls * NBc * tsz); }
    for (int c = 0; c < NCLS_MAX; c++) { m->rtop[c].A = 4; m->rtop[c].N = 1; }
    for (int a = 0; a < 5; a++) for (int b = 0; b < 34; b++) { m->rgap[a][b].A = 2; m->rgap[a][b].N = 1; }
    int32_t *cm = (int32_t *)malloc(sizeof(int32_t) * C);
    for (int64_t j = 0; j < C; j++) cm[j] = 12 << 4;
    int dig[34]; int dec = r ? r->dec : w->dec;
    int minstep = o->wave ? 2 : 1;
    int32_t rm = 12 << 4;                                  /* the row meter: round one let it run on across rows */
    for (int64_t i = 0; i < R; i++) {
        if (o->rowreset) rm = 12 << 4;
        for (int64_t j = 0; j < C; j++) {
            int32_t val = dec ? 0 : v[i * C + j];
            uint32_t aval = (uint32_t)(val < 0 ? -val : val);
            int T = 0; if (!dec) T = digits(aval, dig, o->wave);
            int cls = ctx_cls(o, i, j, colcls, colterm, rowterm);
            int mc = ctx_meter(o, rm, cm[j]);
            int32_t out = 0;
            if (o->coder == 0) {
                Prob *tp = m->top + ((size_t)cls * NBc + mc) * 32;
                int node = 1, Td = 0;
                for (int b = 4; b >= 0; b--) { int x = bit(r, &tp[node], (T >> b) & 1); node = 2 * node + x; Td = 2 * Td + x; }
                T = Td;
                if (T) {
                    int sg = bit(r, &m->sign[o->signcls ? cls : 0], val < 0);
                    int32_t a = fax_lower(r, m, o->wave, o->lowcls ? (cls & 1) : 0, T, dig);
                    out = sg ? -a : a;
                }
            } else if (o->coder == 3) {
                uint32_t z = dec ? 0 : ((uint32_t)aval << 1) - (val < 0 ? 1u : 0u);  /* zigzag */
                uint32_t escs = (uint32_t)tsz - 1;                                   /* the escape symbol */
                uint32_t sym = z < escs ? z : escs;
                Prob *tr = m->tree + ((size_t)cls * NBc + mc) * tsz;
                uint32_t node = 1;
                for (int b = o->tbits - 1; b >= 0; b--) { int x = bit(r, &tr[node], (sym >> b) & 1); node = 2 * node + x; }
                sym = node - (uint32_t)tsz;
                if (sym < escs) {
                    z = sym; out = (z & 1) ? -(int32_t)((z + 1) >> 1) : (int32_t)(z >> 1);
                    int dd[34]; T = digits((uint32_t)(out < 0 ? -out : out), dd, o->wave);   /* what the meters read */
                }
                else {                                                                /* escape: fax photons */
                    Prob *tp = m->top;
                    int node2 = 1, Td = 0;
                    for (int b = 4; b >= 0; b--) { int x = bit(r, &tp[node2], (T >> b) & 1); node2 = 2 * node2 + x; Td = 2 * Td + x; }
                    T = Td;
                    int sg = bit(r, &m->sign[0], val < 0);
                    int32_t a = fax_lower(r, m, o->wave, 0, T, dig);
                    out = sg ? -a : a;
                }
            } else {                                                                  /* coders 1 and 2: raw bits */
                if (o->coder == 1) T = (int)rw_gamma(w, (uint32_t)T + 1) - 1;
                else {
                    int pred = (rm + cm[j] + 16) >> 5;                                 /* the two meters' average place */
                    int e = T - pred; uint32_t zz = e >= 0 ? (uint32_t)(2 * e) : (uint32_t)(-2 * e - 1);
                    RiceCtx *rc = &m->rtop[cls < NCLS_MAX ? cls : 0]; int k = rice_k(rc);
                    zz = rw_rice(w, zz, k); rice_upd(rc, zz);
                    e = (zz & 1) ? -(int)((zz + 1) >> 1) : (int)(zz >> 1); T = pred + e;
                }
                if (T) {
                    int sg = rw_bit(w, val < 0);
                    int p = T - 1, k = 1; int32_t a = (int32_t)1 << p;
                    while (p >= minstep) {
                        int q = -1;
                        if (!dec) for (int t = p - minstep; t >= 0; t--) if (dig[t]) { q = t; break; }
                        /* 0 = no more photons; else the run of dark places down to the next photon, plus one */
                        uint32_t x = dec ? 0 : (q < 0 ? 0u : (uint32_t)(p - minstep - q + 1));
                        if (o->coder == 1) x = rw_gamma(w, x + 1) - 1;
                        else { int kk = k < 4 ? k : 4; RiceCtx *rc = &m->rgap[kk][p]; int kr = rice_k(rc); x = rw_rice(w, x, kr); rice_upd(rc, x); }
                        if (x == 0) break;
                        q = p - minstep - (int)(x - 1);
                        int s = 0; if (o->wave) s = rw_bit(w, dec ? 0 : (dig[q] < 0));
                        a += s ? -((int32_t)1 << q) : ((int32_t)1 << q);
                        p = q; k++;
                    }
                    out = sg ? -a : a;
                }
            }
            if (dec) v[i * C + j] = out;
            else if (out != val) { free(m->top); free(m->sign); if (m->tree) free(m->tree); free(m); free(cm); return -1; }
            rm += ((T << 4) - rm) >> 4;
            cm[j] += ((T << 4) - cm[j]) >> 3;
        }
    }
    free(m->top); free(m->sign); if (m->tree) free(m->tree); free(m); free(cm);
    return 0;
}


/* ------------------------------------------------------------------ coder 4: PAGES
   The photons laid out as pages, one page per place, sent from the top place down, each page in raster order (the
   way a fax sends a page line by line). On page p each value answers:
     not lit yet:  "does my top photon sit on this page?" (then its direction = the value's sign), read against the
                   page, the class, and whether the neighbour to the left and the one above are already lit
                   (0 = not yet, 1 = lit on a page above, 2 = lit on this page) -- the photon pages' own texture;
     lit already:  "is there a photon here?" (then up or hole), read against the page, the distance down from its
                   last photon and how many it has; skipped when the last photon is on the page just above
                   (single wave: never two side by side).                                                      */
#define NPAGE 20
static int code_pages(RC *r, int32_t *v, int64_t R, int64_t C, const Opt *o,
                      const int32_t *colcls, const int32_t *colterm, const int32_t *rowterm) {
    int ncls = o->ncls < 1 ? 1 : o->ncls; int dec = r->dec; int minstep = o->wave ? 2 : 1;
    size_t nstart = (size_t)ncls * NPAGE * 9;
    Prob *pst = (Prob *)malloc(sizeof(Prob) * nstart); probs_init(pst, nstart);
    Prob *psg = (Prob *)malloc(sizeof(Prob) * ncls); probs_init(psg, ncls);
    static Prob plit[NPAGE][NPAGE + 2][5], pdir[NPAGE][2];
    probs_init(&plit[0][0][0], sizeof(plit) / sizeof(Prob)); probs_init(&pdir[0][0], sizeof(pdir) / sizeof(Prob));
    int64_t n = R * C;
    int8_t *last = (int8_t *)malloc(n); int8_t *kk = (int8_t *)malloc(n); int8_t *pd = (int8_t *)malloc(n);
    int8_t *sg = (int8_t *)malloc(n); int32_t *acc = (int32_t *)calloc(n, sizeof(int32_t));
    for (int64_t x = 0; x < n; x++) { last[x] = -1; kk[x] = 0; pd[x] = 0; sg[x] = 0; }
    if (!dec) for (int64_t x = 0; x < n; x++) {
        uint32_t a = (uint32_t)(v[x] < 0 ? -v[x] : v[x]); sg[x] = v[x] < 0;
        if ((uint64_t)a >> (NPAGE - 1)) { free(pst); free(psg); free(last); free(kk); free(pd); free(sg); free(acc); return -2; }
    }
    for (int p = NPAGE - 1; p >= 0; p--) {
        for (int64_t i = 0; i < R; i++) {
            for (int64_t j = 0; j < C; j++) {
                int64_t x = i * C + j;
                int d = 0;                                          /* encoder: this value's digit on page p */
                if (!dec) {
                    uint64_t a = (uint64_t)(v[x] < 0 ? -(int64_t)v[x] : v[x]);
                    if (o->wave) { uint64_t a3 = 3 * a; d = (int)(((a3 & ~a) >> (p + 1)) & 1) - (int)(((a & ~a3) >> (p + 1)) & 1); }
                    else d = (int)((a >> p) & 1);
                }
                if (last[x] < 0) {                                  /* not lit yet */
                    int nl = 0, nu = 0;
                    if (j > 0) { int64_t y = x - 1; nl = last[y] < 0 ? 0 : (last[y] > p ? 1 : 2); }
                    if (i > 0) { int64_t y = x - C; nu = last[y] < 0 ? 0 : (last[y] > p ? 1 : 2); }
                    int cls = ctx_cls(o, i, j, colcls, colterm, rowterm);
                    if (bit(r, &pst[((size_t)cls * NPAGE + p) * 9 + nl * 3 + nu], d != 0)) {
                        int s = bit(r, &psg[o->signcls ? cls : 0], dec ? 0 : sg[x]);
                        sg[x] = (int8_t)s; last[x] = (int8_t)p; kk[x] = 1; pd[x] = 0;
                        acc[x] = (int32_t)1 << p;
                    }
                } else {                                            /* lit already */
                    int gap = last[x] - p; if (gap < minstep) continue;
                    int k4 = kk[x] < 4 ? kk[x] : 4; int g = gap < NPAGE + 1 ? gap : NPAGE + 1;
                    if (bit(r, &plit[p][g][k4], d != 0)) {
                        int down = 0;
                        if (o->wave) down = bit(r, &pdir[p][(int)pd[x]], d < 0);
                        acc[x] += down ? -((int32_t)1 << p) : ((int32_t)1 << p);
                        pd[x] = (int8_t)down; last[x] = (int8_t)p; if (kk[x] < 100) kk[x]++;
                    }
                }
            }
        }
    }
    int bad = 0;
    for (int64_t x = 0; x < n; x++) {
        int32_t out = sg[x] ? -acc[x] : acc[x];
        if (dec) v[x] = out; else if (out != v[x]) bad = 1;
    }
    free(pst); free(psg); free(last); free(kk); free(pd); free(sg); free(acc);
    return bad ? -1 : 0;
}

static void opt_from(const int32_t *oa, Opt *o) {
    o->coder = oa[0]; o->wave = oa[1]; o->meters = oa[2]; o->clsmode = oa[3]; o->ncls = oa[4]; o->A = oa[5];
    o->signcls = oa[6]; o->lim = oa[7]; o->tbits = oa[8]; o->lowcls = oa[9]; o->rowreset = oa[10];
}

/* rows reset the row meter (a new line of the plate / a new word of the table) */
int64_t ph_encode(int32_t *v, int64_t R, int64_t C, const int32_t *oa, const int32_t *colcls, const int32_t *colterm,
                  const int32_t *rowterm, uint8_t *out, int64_t cap) {
    Opt o; opt_from(oa, &o); LIM = o.lim > 0 ? o.lim : 11;
    memset(out, 0, (size_t)cap);
    if (o.coder == 1 || o.coder == 2) {
        BW w; w.dec = 0; w.buf = out; w.cap = cap; w.len = 0; w.bitpos = 0;
        if (code_array(NULL, &w, v, R, C, &o, colcls, colterm, rowterm) < 0) return -1;
        int64_t n = (w.bitpos + 7) >> 3; return n <= cap ? n : -2;
    }
    RC r; memset(&r, 0, sizeof r); r.dec = 0; r.buf = out; r.cap = cap; r.range = 0xFFFFFFFFu; r.csize = 1;
    if ((o.coder == 4 ? code_pages(&r, v, R, C, &o, colcls, colterm, rowterm)
                      : code_array(&r, NULL, v, R, C, &o, colcls, colterm, rowterm)) < 0) return -1;
    for (int i = 0; i < 5; i++) enc_shift(&r);
    return r.pos <= cap ? r.pos : -2;
}
int ph_decode(uint8_t *in, int64_t len, int64_t R, int64_t C, const int32_t *oa, const int32_t *colcls,
              const int32_t *colterm, const int32_t *rowterm, int32_t *v) {
    Opt o; opt_from(oa, &o); LIM = o.lim > 0 ? o.lim : 11;
    if (o.coder == 1 || o.coder == 2) {
        BW w; w.dec = 1; w.buf = in; w.cap = 0; w.len = len; w.bitpos = 0;
        return code_array(NULL, &w, v, R, C, &o, colcls, colterm, rowterm);
    }
    RC r; memset(&r, 0, sizeof r); r.dec = 1; r.buf = in; r.len = len; r.range = 0xFFFFFFFFu;
    for (int i = 0; i < 5; i++) r.code = (r.code << 8) | in_byte(&r);
    if (o.coder == 4) return code_pages(&r, v, R, C, &o, colcls, colterm, rowterm);
    return code_array(&r, NULL, v, R, C, &o, colcls, colterm, rowterm);
}
