/* confirm_groups/audit_k.c -- the counting-convention audit's own kernels. Written for the audit; shares no code with
 * base2_groups.py, qwen_groups.py, groups/ or groups2/. Whole numbers only (int64 sums).
 *
 * prod_c: out[p, j] = sum_i x[p, i] w[i, j], w stored row by row (N x M, C order)      -- GPT-2's tables
 * prod_f: the same, w stored neuron by neuron (column j contiguous, Fortran order)     -- Qwen's tables
 * bank:   the up/hole part of one plate in the group form, one position per task:
 *           per random group of gk lines, every pattern's ready pile V[p] = (light on the up lines) - (light on the
 *           holes), made by the lowest-bit rule (V[0] = -total, V[p] = V[p less its lowest bit] + 2 x);
 *           each neuron j at each up/hole place b (bit b of up[j]) adds V[pattern] << b into UP[j];
 *           counted per position: the reads' signed digits (rn) and 1s (r1); the half-table entries the plate reads
 *           (readh, plate only), each once: their 1s (w1) and signed digits (wn).
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <dispatch/dispatch.h>

static inline int64_t nafc(int64_t v) { uint64_t a = v < 0 ? (uint64_t)0 - (uint64_t)v : (uint64_t)v; return __builtin_popcountll(a ^ (3 * a)); }
static inline int64_t onec(int64_t v) { uint64_t a = v < 0 ? (uint64_t)0 - (uint64_t)v : (uint64_t)v; return __builtin_popcountll(a); }

/* ---------- products ---------- */
typedef struct { int64_t P, N, M, PB, CB; const int32_t *x; const int16_t *w; int64_t *out; } prod_t;
static void tile_c(void *ctx, size_t t) {
    const prod_t *D = (const prod_t *)ctx; int64_t ncb = (D->M + D->CB - 1) / D->CB;
    int64_t p0 = (int64_t)(t / ncb) * D->PB, c0 = (int64_t)(t % ncb) * D->CB;
    int64_t p1 = p0 + D->PB < D->P ? p0 + D->PB : D->P, c1 = c0 + D->CB < D->M ? c0 + D->CB : D->M, nc = c1 - c0;
    int64_t *acc = (int64_t *)calloc((size_t)((p1 - p0) * nc), sizeof(int64_t));
    for (int64_t i = 0; i < D->N; i++) {
        const int16_t *wr = D->w + i * D->M + c0;
        for (int64_t p = p0; p < p1; p++) {
            int64_t xv = D->x[p * D->N + i]; if (!xv) continue;
            int64_t *o = acc + (p - p0) * nc;
            for (int64_t j = 0; j < nc; j++) o[j] += xv * (int64_t)wr[j];
        }
    }
    for (int64_t p = p0; p < p1; p++) memcpy(D->out + p * D->M + c0, acc + (p - p0) * nc, sizeof(int64_t) * nc);
    free(acc);
}
static void tile_f(void *ctx, size_t t) {
    const prod_t *D = (const prod_t *)ctx; int64_t ncb = (D->M + D->CB - 1) / D->CB;
    int64_t p0 = (int64_t)(t / ncb) * D->PB, c0 = (int64_t)(t % ncb) * D->CB;
    int64_t p1 = p0 + D->PB < D->P ? p0 + D->PB : D->P, c1 = c0 + D->CB < D->M ? c0 + D->CB : D->M;
    for (int64_t j = c0; j < c1; j++) {
        const int16_t *wc = D->w + j * D->N;
        for (int64_t p = p0; p < p1; p++) {
            const int32_t *xp = D->x + p * D->N; int64_t s = 0;
            for (int64_t i = 0; i < D->N; i++) s += (int64_t)xp[i] * (int64_t)wc[i];
            D->out[p * D->M + j] = s;
        }
    }
}
void prod_c(int64_t P, int64_t N, int64_t M, const int32_t *x, const int16_t *w, int64_t *out) {
    prod_t D = {P, N, M, 16, 2048, x, w, out}; size_t nt = (size_t)(((P + 15) / 16) * ((M + 2047) / 2048));
    dispatch_apply_f(nt, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &D, tile_c);
}
void prod_f(int64_t P, int64_t N, int64_t M, const int32_t *x, const int16_t *w, int64_t *out) {
    prod_t D = {P, N, M, 16, 256, x, w, out}; size_t nt = (size_t)(((P + 15) / 16) * ((M + 255) / 256));
    dispatch_apply_f(nt, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &D, tile_f);
}

/* ---------- the group bank, counted two ways ---------- */
typedef struct {
    int64_t P, N, M, G, gmax;
    const int32_t *x; const int32_t *glines; const int64_t *goff; const uint32_t *up;
    const uint16_t *pat;      /* G x M x 16 */
    const uint8_t *readh;     /* G x 2^(gmax-1) */
    int64_t *UP;              /* P x M */
    int64_t *cnt;             /* P x 4: rn, r1, w1, wn */
} bank_t;
static void bank_pos(void *ctx, size_t pp) {
    const bank_t *J = (const bank_t *)ctx; int64_t p = (int64_t)pp, M = J->M, TH = (int64_t)1 << (J->gmax - 1);
    const int32_t *xp = J->x + p * J->N; int64_t *U = J->UP + p * M; memset(U, 0, sizeof(int64_t) * M);
    int64_t *V = (int64_t *)malloc(sizeof(int64_t) * ((size_t)1 << J->gmax));
    int64_t rn = 0, r1 = 0, w1 = 0, wn = 0;
    for (int64_t k = 0; k < J->G; k++) {
        const int32_t *L = J->glines + J->goff[k]; int gk = (int)(J->goff[k + 1] - J->goff[k]);
        int64_t n = (int64_t)1 << gk, top = n >> 1, T = 0;
        for (int t = 0; t < gk; t++) T += xp[L[t]];
        V[0] = -T;
        for (int64_t q = 1; q < n; q++) V[q] = V[q & (q - 1)] + 2 * (int64_t)xp[L[__builtin_ctzll((uint64_t)q)]];
        const uint8_t *rh = J->readh + k * TH;
        for (int64_t key = 0; key < top; key++) if (rh[key]) { int64_t v = V[top + key]; w1 += onec(v); wn += nafc(v); }
        const uint16_t *pk = J->pat + k * M * 16;
        for (int64_t j = 0; j < M; j++) {
            uint32_t u = J->up[j]; const uint16_t *pj = pk + j * 16; int64_t acc = 0;
            while (u) { int b = __builtin_ctz(u); u &= u - 1; int64_t v = V[pj[b]]; acc += v << b; rn += nafc(v); r1 += onec(v); }
            U[j] += acc;
        }
    }
    free(V);
    J->cnt[p * 4 + 0] = rn; J->cnt[p * 4 + 1] = r1; J->cnt[p * 4 + 2] = w1; J->cnt[p * 4 + 3] = wn;
}
void bank(int64_t P, int64_t N, int64_t M, int64_t G, int64_t gmax, const int32_t *x, const int32_t *glines,
          const int64_t *goff, const uint32_t *up, const uint16_t *pat, const uint8_t *readh, int64_t *UP, int64_t *cnt) {
    bank_t J = {P, N, M, G, gmax, x, glines, goff, up, pat, readh, UP, cnt};
    dispatch_apply_f((size_t)P, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &J, bank_pos);
}
