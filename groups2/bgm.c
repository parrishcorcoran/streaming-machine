/* groups2/bgm.c -- the bank kernel of base2_groups.py and qwen_groups.py (the whole machines with group piles).
 * Written for them; it shares no code with groups/gm.c, groups/gm_uh.c, groups2/uhm.c or groups2/topg.c.
 * Whole numbers only (int64 inside). One position per task, positions spread over the cores.
 *
 * A plate whose places are each read up/hole or in signed digits (the per-place form), one light (P x N):
 *   per position, per random group of gk light lines (the group's lines in the order given, bit t = the group's line t):
 *     V[p] = light on the lines up in p - light on the lines that are holes in p, every pattern p, made here by the
 *     lowest-bit rule V[0] = -(group total), V[p] = V[p less its lowest bit] + 2 x. (How the values are made here does
 *     not matter to the count: the table's adds are counted in the Python by the Gray walk's relation.)
 *     The half table holds the patterns with the group's last line up; the other half is each one's flip (Babinet),
 *     V[p ^ all] = -V[p]. A half-table entry that the plate reads (itself or its flip, plate only: readh) is written in
 *     signed digits once a position: its 1s are counted (wr).
 *     Neuron j, at each of its up/hole places b (bit b of up[j]): one ready pile V[pattern(group, j, b)] added into
 *     the place part A[j][b] (doubled scale, before the 2^b); its photons, the ready pile's signed digits, into H[j][b].
 *   every lit signed digit (line, place, sign) of neuron j: A[j][place] += 2 sign x(line) (the doubled scale), its
 *     photons, the light value's signed digits, into H[j][place].
 *   The doubled pile is sum over b of 2^b A[j][b] + kcorr[j] X (X = the light summed), formed in the Python.
 * plain_product: the check path's direct product (lines met one by one, int64).
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <dispatch/dispatch.h>

static inline int sdc(int64_t v) { uint64_t a = v < 0 ? (uint64_t)0 - (uint64_t)v : (uint64_t)v; return __builtin_popcountll(a ^ (3 * a)); }
static inline int onec(int64_t v) { uint64_t a = v < 0 ? (uint64_t)0 - (uint64_t)v : (uint64_t)v; return __builtin_popcountll(a); }

typedef struct {
    int64_t P, N, M, G, gmax, NPL, anyup;
    const int32_t *x;          /* P x N */
    const int32_t *glines;     /* N */
    const int64_t *goff;       /* G + 1 */
    const uint32_t *up;        /* M */
    const uint16_t *pat;       /* G x M x 16 */
    const uint8_t *readh;      /* G x 2^(gmax - 1) */
    const int64_t *toff;       /* M + 1 */
    const int32_t *tline; const int8_t *tplace; const int8_t *tsign;
    int64_t *A;                /* P x M x NPL */
    int32_t *H;                /* P x M x NPL */
    int64_t *wr;               /* P */
} job_t;

static void one_position(void *ctx, size_t p) {
    const job_t *J = (const job_t *)ctx;
    const int64_t M = J->M, NPL = J->NPL, T = (int64_t)1 << J->gmax, TH = T >> 1;
    const int32_t *xp = J->x + p * J->N;
    int64_t *Ap = J->A + p * M * NPL; int32_t *Hp = J->H + p * M * NPL;
    memset(Ap, 0, sizeof(int64_t) * M * NPL); memset(Hp, 0, sizeof(int32_t) * M * NPL);
    int64_t w = 0;
    if (J->anyup) {
        int64_t *V = (int64_t *)malloc(sizeof(int64_t) * T);
        for (int64_t k = 0; k < J->G; k++) {
            const int32_t *L = J->glines + J->goff[k]; int gk = (int)(J->goff[k + 1] - J->goff[k]);
            int64_t n = (int64_t)1 << gk, top = n >> 1, TG = 0;
            for (int t = 0; t < gk; t++) TG += xp[L[t]];
            V[0] = -TG;
            for (int64_t q = 1; q < n; q++) V[q] = V[q & (q - 1)] + 2 * (int64_t)xp[L[__builtin_ctzll((uint64_t)q)]];
            const uint8_t *rh = J->readh + k * TH;
            for (int64_t key = 0; key < top; key++) if (rh[key]) w += onec(V[top + key]);
            const uint16_t *pk = J->pat + k * M * 16;
            for (int64_t j = 0; j < M; j++) {
                uint32_t u = J->up[j]; if (!u) continue;
                const uint16_t *pj = pk + j * 16; int64_t *a = Ap + j * NPL; int32_t *h = Hp + j * NPL;
                while (u) { int b = __builtin_ctz(u); u &= u - 1; int64_t v = V[pj[b]]; a[b] += v; h[b] += sdc(v); }
            }
        }
        free(V);
    }
    for (int64_t j = 0; j < M; j++) {
        int64_t *a = Ap + j * NPL; int32_t *h = Hp + j * NPL;
        for (int64_t q = J->toff[j]; q < J->toff[j + 1]; q++) {
            int64_t xv = xp[J->tline[q]]; int b = J->tplace[q];
            a[b] += 2 * (int64_t)J->tsign[q] * xv; h[b] += sdc(xv);
        }
    }
    J->wr[p] = w;
}
void bank_groups(int64_t P, int64_t N, int64_t M, int64_t G, int64_t gmax, int64_t NPL, int64_t anyup, const int32_t *x,
                 const int32_t *glines, const int64_t *goff, const uint32_t *up, const uint16_t *pat, const uint8_t *readh,
                 const int64_t *toff, const int32_t *tline, const int8_t *tplace, const int8_t *tsign,
                 int64_t *A, int32_t *H, int64_t *wr) {
    job_t J = {P, N, M, G, gmax, NPL, anyup, x, glines, goff, up, pat, readh, toff, tline, tplace, tsign, A, H, wr};
    dispatch_apply_f((size_t)P, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &J, one_position);
}

/* the check path: out[p, j] = sum over i of x[p, i] w[i, j], lines met one by one, int64 */
typedef struct { int64_t P, N, M; const int32_t *x; const int16_t *w; int64_t *out; } pj_t;
static void prod_row(void *ctx, size_t p) {
    const pj_t *D = (const pj_t *)ctx; int64_t *o = D->out + p * D->M; memset(o, 0, sizeof(int64_t) * D->M);
    for (int64_t i = 0; i < D->N; i++) {
        int64_t xv = D->x[p * D->N + i]; if (xv == 0) continue;
        const int16_t *wr = D->w + i * D->M;
        for (int64_t j = 0; j < D->M; j++) o[j] += xv * (int64_t)wr[j];
    }
}
void plain_product(int64_t P, int64_t N, int64_t M, const int32_t *x, const int16_t *w, int64_t *out) {
    pj_t D = {P, N, M, x, w, out};
    dispatch_apply_f((size_t)P, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &D, prod_row);
}
