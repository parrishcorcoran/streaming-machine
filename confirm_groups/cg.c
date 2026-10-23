/* confirm_groups/cg.c -- the independent checker's own kernels for the Qwen3-0.6B group machine (written from scratch;
   shares no code with groups/, groups2/, base2_groups.py or qwen_groups.py). Whole numbers only (int64 sums).
   Layout: positions are the innermost index everywhere (T [G][H][P], acc [M][P], parts [M][nup][P]).

   half_tables : every group's half table of ready piles (the patterns with the group's last line up), built from the
                 group total by a Gray walk over the other lines; the walk's own photon count (the group total: the
                 signed digits of lines 1..gk-1; each Gray step: the signed digits of the light value turned), and the
                 signed digits of every entry.
   uh_reads    : each neuron, at each up/hole place, adds one ready pile (or takes away its flip) shifted to that place;
                 counted: the ready pile's signed digits.
   uh_parts    : the same, each place's part kept apart per word (the top read reads place by place).
   sd_reads    : each lit signed digit meets its own light value; counted: that value's signed digits.
   pile_write  : the 1s of every half-table entry the plate reads (the ready piles written in signed digits).
   mm16        : CHECK PATH ONLY: the direct product x . w, every line met one by one, int64.
*/
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
typedef int64_t i64;

static inline int naf1(i64 v) { uint64_t a = v < 0 ? (uint64_t)(-v) : (uint64_t)v; return __builtin_popcountll(a ^ (3 * a)); }
static inline int ones1(i64 v) { uint64_t a = v < 0 ? (uint64_t)(-v) : (uint64_t)v; return __builtin_popcountll(a); }
int naf_check(i64 v) { return naf1(v); }
int ones_check(i64 v) { return ones1(v); }

/* x [P][N]; groups: gl (line ids in group order), goff[G+1]; H slots a group (2^(gmax-1)).
   T [G][H][P] int64, NT [G][H][P] uint8, walk[P] += the walk's photons */
void half_tables(i64 P, i64 N, const i64 *x, i64 G, const i64 *gl, const i64 *goff, i64 H, i64 *T, uint8_t *NT, i64 *walk)
{
    for (i64 p = 0; p < P; p++) {
        const i64 *xp = x + p * N; i64 cost = 0;
        for (i64 k = 0; k < G; k++) {
            i64 a = goff[k], gk = goff[k + 1] - a; i64 h = (i64)1 << (gk - 1);
            i64 v = xp[gl[a]];                                   /* the group total: line 0, then lines 1..gk-1 added */
            for (i64 t = 1; t < gk; t++) { v += xp[gl[a + t]]; cost += naf1(xp[gl[a + t]]); }
            i64 idx = h - 1;                                     /* all lines up (bits over lines 0..gk-2) */
            T[(k * H + idx) * P + p] = v; NT[(k * H + idx) * P + p] = (uint8_t)naf1(v);
            for (i64 s = 1; s < h; s++) {
                int t = __builtin_ctzll((uint64_t)s); idx ^= ((i64)1 << t);
                i64 xv = xp[gl[a + t]];
                if ((idx >> t) & 1) v += 2 * xv; else v -= 2 * xv;     /* the turn: twice the line, a shift */
                cost += naf1(xv);
                T[(k * H + idx) * P + p] = v; NT[(k * H + idx) * P + p] = (uint8_t)naf1(v);
            }
        }
        walk[p] += cost;
    }
}

typedef struct {
    i64 P, G, H, M, nup, j0, j1; const int32_t *ups; const uint16_t *pat; const i64 *T; const uint8_t *NT; const i64 *gorder;
    i64 *acc; i64 *cnt; i64 *parts; int32_t *hparts;
} RArg;

static void *reads_w(void *vp)
{
    RArg *r = (RArg *)vp; i64 P = r->P, G = r->G, H = r->H, M = r->M, nup = r->nup;
    for (i64 kk = 0; kk < G; kk++) {
        i64 k = r->gorder ? r->gorder[kk] : kk;
        for (i64 j = r->j0; j < r->j1; j++) {
            const uint16_t *pc = r->pat + (k * M + j) * nup;
            for (i64 u = 0; u < nup; u++) {
                uint16_t c = pc[u]; i64 idx = c & 0x7FFF; int neg = c >> 15; int b = r->ups[u];
                const i64 *Tv = r->T + (k * H + idx) * P; const uint8_t *Nv = r->NT + (k * H + idx) * P;
                if (r->parts) {
                    i64 *o = r->parts + (j * nup + u) * P; int32_t *ho = r->hparts + (j * nup + u) * P;
                    if (neg) for (i64 p = 0; p < P; p++) { o[p] -= Tv[p]; ho[p] += Nv[p]; }
                    else     for (i64 p = 0; p < P; p++) { o[p] += Tv[p]; ho[p] += Nv[p]; }
                } else {
                    i64 *o = r->acc + j * P; i64 *cn = r->cnt;
                    if (neg) for (i64 p = 0; p < P; p++) { o[p] -= Tv[p] * ((i64)1 << b); cn[p] += Nv[p]; }
                    else     for (i64 p = 0; p < P; p++) { o[p] += Tv[p] * ((i64)1 << b); cn[p] += Nv[p]; }
                }
            }
        }
    }
    return NULL;
}

static void run_reads(i64 P, i64 G, i64 H, const i64 *T, const uint8_t *NT, i64 M, i64 nup, const int32_t *ups,
                      const uint16_t *pat, const i64 *gorder, i64 *acc, i64 *cnt, i64 *parts, int32_t *hparts, int nth)
{
    if (nth < 1) nth = 1; if (nth > 64) nth = 64;
    pthread_t th[64]; RArg a[64]; i64 *cn = calloc((size_t)(nth * P), sizeof(i64));
    for (int t = 0; t < nth; t++) {
        a[t] = (RArg){P, G, H, M, nup, M * t / nth, M * (t + 1) / nth, ups, pat, T, NT, gorder, acc, cn + t * P, parts, hparts};
        pthread_create(&th[t], NULL, reads_w, &a[t]);
    }
    for (int t = 0; t < nth; t++) pthread_join(th[t], NULL);
    if (cnt) for (int t = 0; t < nth; t++) for (i64 p = 0; p < P; p++) cnt[p] += cn[t * P + p];
    free(cn);
}

/* acc [M][P] += the shifted ready piles; cnt[P] += their signed digits */
void uh_reads(i64 P, i64 G, i64 H, const i64 *T, const uint8_t *NT, i64 M, i64 nup, const int32_t *ups, const uint16_t *pat,
              const i64 *gorder, i64 *acc, i64 *cnt, int nth)
{ run_reads(P, G, H, T, NT, M, nup, ups, pat, gorder, acc, cnt, NULL, NULL, nth); }

/* parts [M][nup][P] += the ready piles (unshifted), hparts [M][nup][P] += their signed digits */
void uh_parts(i64 P, i64 G, i64 H, const i64 *T, const uint8_t *NT, i64 M, i64 nup, const int32_t *ups, const uint16_t *pat,
              const i64 *gorder, i64 *parts, int32_t *hparts, int nth)
{ run_reads(P, G, H, T, NT, M, nup, ups, pat, gorder, NULL, NULL, parts, hparts, nth); }

/* each lit signed digit (line dl, neuron dj, place db, sign ds) meets its own light value: the doubled pile gets
   ds * 2^(db+1) * x; counted: naf(x). xT [N][P]. Either acc [M][P] and cnt[P], or (sdp, hsd) [M][nsp][P] by slot dsl. */
void sd_reads(i64 P, const i64 *xT, i64 nd, const int32_t *dl, const int32_t *dj, const int32_t *db, const int8_t *ds,
              i64 *acc, i64 *cnt, i64 nsp, const int32_t *dsl, i64 *sdp, int32_t *hsd)
{
    for (i64 d = 0; d < nd; d++) {
        const i64 *xv = xT + (i64)dl[d] * P; i64 j = dj[d]; int sh = db[d] + 1; int s = ds[d];
        if (sdp) {
            i64 *o = sdp + (j * nsp + dsl[d]) * P; int32_t *h = hsd + (j * nsp + dsl[d]) * P;
            for (i64 p = 0; p < P; p++) { i64 v = xv[p] * ((i64)1 << sh); o[p] += s > 0 ? v : -v; h[p] += naf1(xv[p]); }
        } else {
            i64 *o = acc + j * P;
            for (i64 p = 0; p < P; p++) { i64 v = xv[p] * ((i64)1 << sh); o[p] += s > 0 ? v : -v; cnt[p] += naf1(xv[p]); }
        }
    }
}

/* the 1s of every half-table entry the plate reads (used [G][H] = 1), per position */
void pile_write(i64 P, i64 G, i64 H, const i64 *T, const uint8_t *used, i64 *out)
{
    for (i64 k = 0; k < G; k++) for (i64 i = 0; i < H; i++) if (used[k * H + i]) {
        const i64 *Tv = T + (k * H + i) * P;
        for (i64 p = 0; p < P; p++) out[p] += ones1(Tv[p]);
    }
}

/* ---------------- CHECK PATH: direct product, int64 ---------------- */
typedef struct { i64 P, N, M, j0, j1; const i64 *x; const int16_t *w; i64 *out; } MArg;
static void *mm16_w(void *vp)
{
    MArg *a = (MArg *)vp; i64 P = a->P, N = a->N, M = a->M, J = a->j1 - a->j0;
    i64 *row = malloc(sizeof(i64) * (size_t)(J > 0 ? J : 1));
    for (i64 p = 0; p < P; p++) {
        memset(row, 0, sizeof(i64) * (size_t)J);
        for (i64 t = 0; t < N; t++) {
            i64 xv = a->x[p * N + t]; if (!xv) continue;
            const int16_t *wr = a->w + t * M + a->j0;
            for (i64 j = 0; j < J; j++) row[j] += xv * (i64)wr[j];
        }
        memcpy(a->out + p * M + a->j0, row, sizeof(i64) * (size_t)J);
    }
    free(row); return NULL;
}
/* x [P][N] int64, w [N][M] int16 (row-major), out [P][M] */
void mm16(i64 P, i64 N, i64 M, const i64 *x, const int16_t *w, i64 *out, int nth)
{
    if (nth < 1) nth = 1; if (nth > 64) nth = 64;
    pthread_t th[64]; MArg a[64];
    for (int t = 0; t < nth; t++) { a[t] = (MArg){P, N, M, M * t / nth, M * (t + 1) / nth, x, w, out}; pthread_create(&th[t], NULL, mm16_w, &a[t]); }
    for (int t = 0; t < nth; t++) pthread_join(th[t], NULL);
}
