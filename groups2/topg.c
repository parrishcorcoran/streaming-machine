/* groups2/topg.c -- group piles at the top tables (written for groups2/topg.py; shares no code with groups/gm.c,
 * groups/gm_uh.c or groups2/uhm.c). Whole numbers only: int64 inside, the place parts stored as int32 (their size is
 * bounded below 2^27; any overflow would break the pile check, which is made at every word and position).
 *
 * One position per task, positions spread over the cores. Per position, per random group of gk light lines (the group's
 * lines in the order given, bit t = the group's line t):
 *   UP/HOLE table (forms 0 and 2): V[p] = light on the lines up in p - light on the lines that are holes in p, every
 *     pattern p, made here by the lowest-bit rule V[0] = -(group total), V[p] = V[p less its lowest bit] + 2 x.
 *   SUBSET table (form 1): S[u] = the light on the lines in u, S[0] = 0, S[u] = S[u less its lowest bit] + x.
 *   (How the tables' values are made here does not matter to the count: their work is counted in topg.py by the
 *   Gray-walk relation, as every earlier group machine counted it.)
 *   Every word j at every place b reads one ready pile:
 *     form 0 (up/hole, 16 places): v = V[pattern]; the place part at doubled scale A[j][b] += v
 *     form 2 (per place): at the up/hole places (bits of ppmask) as form 0; at the signed-digit places each lit digit
 *       meets its own light value: A[j][b] += 2 s x(line)
 *     form 1 (signed digits, subsets): up-subset u, down-subset d; both lit: v = S[u] - S[d] (a combine: the down
 *       pile's digits, then the read: v's digits); one lit: v = S[u] or -S[d]; A[j][b] += 2 v (doubled scale)
 *   Photons of every read (the ready pile's signed digits, popcount(a ^ 3a); a shift adds none) and of every combine
 *   are summed per (word, place) into PH[j][b]; the adds per place are summed over the words into NA.
 * The doubled pile is then sum over b of 2^b A[j][b] + kcorr X (kcorr = -1 for form 0; K - 2C for form 2; 0 for form 1),
 * formed in topg.py.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <dispatch/dispatch.h>

static inline int sdc(int64_t v) { uint64_t a = v < 0 ? (uint64_t)0 - (uint64_t)v : (uint64_t)v; return __builtin_popcountll(a ^ (3 * a)); }
static inline uint32_t patv(const void *p, int64_t wide, int64_t i) {
    return wide ? ((const uint32_t *)p)[i] : (uint32_t)((const uint16_t *)p)[i];
}

#define NF 3
#define NPMAX 18
static const int64_t NPL[NF] = {16, 16, 18};

typedef struct {
    int64_t P, N, M, G, gmax, wide;
    const int32_t *x; const int32_t *glines; const int64_t *goff;
    int64_t on0, on1, on2;
    const void *pat_uh, *pat_u, *pat_d, *pat_pp;     /* G x M x 16 */
    int64_t ppmask;                                   /* the up/hole places of form 2 */
    const int64_t *toff; const int32_t *tline; const int8_t *tplace; const int8_t *tsign;   /* form 2's signed digits, by word */
    int32_t *A0, *A1, *A2; uint16_t *H0, *H1, *H2;   /* P x M x NPL[f] */
    int64_t *NA;                                      /* P x NF x NPMAX: adds by place, summed over the words */
} job_t;

static void one_position(void *ctx, size_t p) {
    const job_t *J = (const job_t *)ctx;
    const int64_t M = J->M, T = (int64_t)1 << J->gmax;
    int64_t *V = (J->on0 || J->on2) ? (int64_t *)malloc(sizeof(int64_t) * T) : NULL;
    int64_t *S = J->on1 ? (int64_t *)malloc(sizeof(int64_t) * T) : NULL;
    const int32_t *xp = J->x + p * J->N;
    int32_t *A0 = J->on0 ? J->A0 + p * M * 16 : NULL, *A1 = J->on1 ? J->A1 + p * M * 16 : NULL, *A2 = J->on2 ? J->A2 + p * M * 18 : NULL;
    uint16_t *H0 = J->on0 ? J->H0 + p * M * 16 : NULL, *H1 = J->on1 ? J->H1 + p * M * 16 : NULL, *H2 = J->on2 ? J->H2 + p * M * 18 : NULL;
    if (A0) { memset(A0, 0, sizeof(int32_t) * M * 16); memset(H0, 0, sizeof(uint16_t) * M * 16); }
    if (A1) { memset(A1, 0, sizeof(int32_t) * M * 16); memset(H1, 0, sizeof(uint16_t) * M * 16); }
    if (A2) { memset(A2, 0, sizeof(int32_t) * M * 18); memset(H2, 0, sizeof(uint16_t) * M * 18); }
    int64_t na[NF][NPMAX]; memset(na, 0, sizeof na);
    uint32_t ppm = (uint32_t)J->ppmask;
    for (int64_t k = 0; k < J->G; k++) {
        const int32_t *L = J->glines + J->goff[k]; int gk = (int)(J->goff[k + 1] - J->goff[k]); int64_t n = (int64_t)1 << gk;
        if (V) {
            int64_t TG = 0; for (int t = 0; t < gk; t++) TG += xp[L[t]];
            V[0] = -TG;
            for (int64_t q = 1; q < n; q++) V[q] = V[q & (q - 1)] + 2 * (int64_t)xp[L[__builtin_ctzll((uint64_t)q)]];
        }
        if (S) {
            S[0] = 0;
            for (int64_t q = 1; q < n; q++) S[q] = S[q & (q - 1)] + (int64_t)xp[L[__builtin_ctzll((uint64_t)q)]];
        }
        const int64_t base = k * M * 16;
        for (int64_t j = 0; j < M; j++) {
            const int64_t idx = base + j * 16;
            if (A0) {
                int32_t *a = A0 + j * 16; uint16_t *h = H0 + j * 16;
                for (int b = 0; b < 16; b++) { int64_t v = V[patv(J->pat_uh, J->wide, idx + b)]; a[b] += (int32_t)v; h[b] += (uint16_t)sdc(v); }
            }
            if (A2) {
                int32_t *a = A2 + j * 18; uint16_t *h = H2 + j * 18; uint32_t m = ppm;
                while (m) { int b = __builtin_ctz(m); m &= m - 1; int64_t v = V[patv(J->pat_pp, J->wide, idx + b)]; a[b] += (int32_t)v; h[b] += (uint16_t)sdc(v); }
            }
            if (A1) {
                int32_t *a = A1 + j * 16; uint16_t *h = H1 + j * 16;
                for (int b = 0; b < 16; b++) {
                    uint32_t u = patv(J->pat_u, J->wide, idx + b), d = patv(J->pat_d, J->wide, idx + b);
                    if (!(u | d)) continue;
                    int64_t v; int ph;
                    if (u && d) { v = S[u] - S[d]; ph = sdc(S[d]) + sdc(v); na[1][b] += 2; }
                    else if (u) { v = S[u]; ph = sdc(v); na[1][b] += 1; }
                    else { v = -S[d]; ph = sdc(v); na[1][b] += 1; }
                    a[b] += (int32_t)(2 * v); h[b] += (uint16_t)ph;
                }
            }
        }
    }
    if (A0) for (int b = 0; b < 16; b++) na[0][b] = J->G * M;
    if (A2) {
        for (int b = 0; b < 16; b++) if ((ppm >> b) & 1u) na[2][b] = J->G * M;
        for (int64_t j = 0; j < M; j++)
            for (int64_t q = J->toff[j]; q < J->toff[j + 1]; q++) {
                int64_t xv = xp[J->tline[q]]; int b = J->tplace[q];
                A2[j * 18 + b] += (int32_t)(2 * (int64_t)J->tsign[q] * xv); H2[j * 18 + b] += (uint16_t)sdc(xv); na[2][b] += 1;
            }
    }
    memcpy(J->NA + p * NF * NPMAX, na, sizeof na);
    free(V); free(S);
}

void top_groups(int64_t P, int64_t N, int64_t M, int64_t G, int64_t gmax, int64_t wide, const int32_t *x, const int32_t *glines,
                const int64_t *goff, int64_t on0, int64_t on1, int64_t on2, const void *pat_uh, const void *pat_u, const void *pat_d,
                const void *pat_pp, int64_t ppmask, const int64_t *toff, const int32_t *tline, const int8_t *tplace, const int8_t *tsign,
                int32_t *A0, int32_t *A1, int32_t *A2, uint16_t *H0, uint16_t *H1, uint16_t *H2, int64_t *NA) {
    job_t J = {P, N, M, G, gmax, wide, x, glines, goff, on0, on1, on2, pat_uh, pat_u, pat_d, pat_pp, ppmask, toff, tline, tplace, tsign,
               A0, A1, A2, H0, H1, H2, NA};
    dispatch_apply_f((size_t)P, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &J, one_position);
}

/* CHECK PATH ONLY: the direct product of the light and the table as stored, every line met one by one, int64 sums */
typedef struct { int64_t P, N, M; const int32_t *x; const int16_t *w; int64_t *out; } dp_t;
static void dp_row(void *ctx, size_t p) {
    const dp_t *D = (const dp_t *)ctx; int64_t *o = D->out + p * D->M; memset(o, 0, sizeof(int64_t) * D->M);
    for (int64_t i = 0; i < D->N; i++) {
        int64_t xv = D->x[p * D->N + i]; if (!xv) continue;
        const int16_t *wr = D->w + i * D->M;
        for (int64_t j = 0; j < D->M; j++) o[j] += xv * (int64_t)wr[j];
    }
}
void direct_product(int64_t P, int64_t N, int64_t M, const int32_t *x, const int16_t *w, int64_t *out) {
    dp_t D = {P, N, M, x, w, out};
    dispatch_apply_f((size_t)P, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &D, dp_row);
}

/* CHECK PATH ONLY: the same product with the table stored word by word (column order: w[j * N + i]) */
static void dpf_row(void *ctx, size_t p) {
    const dp_t *D = (const dp_t *)ctx; int64_t *o = D->out + p * D->M; const int32_t *xp = D->x + p * D->N;
    for (int64_t j = 0; j < D->M; j++) {
        const int16_t *wc = D->w + j * D->N; int64_t s = 0;
        for (int64_t i = 0; i < D->N; i++) s += (int64_t)xp[i] * (int64_t)wc[i];
        o[j] = s;
    }
}
void direct_product_f(int64_t P, int64_t N, int64_t M, const int32_t *x, const int16_t *w, int64_t *out) {
    dp_t D = {P, N, M, x, w, out};
    dispatch_apply_f((size_t)P, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &D, dpf_row);
}
