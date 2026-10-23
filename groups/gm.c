/* The group machine's inner loop, whole numbers only (int64), one position per task, positions spread over the cores.
 *
 * For one plate (lines x neurons) and one grouping of its light lines:
 *   per position, per group of gk lines:
 *     the table: every subset's pile of the light, T[S] = sum of x over the lines in S, built in Gray-code order, each
 *       entry from its neighbour by one add (or take-away) of one light value; counted in the light value's signed
 *       digits. Also counted: the half table (subsets without the group's last line, Gray order over the other gk-1
 *       lines, then the group total by one more add), and making the other half by the flip (T[S] = T[G] - T[G\S],
 *       one take-away of a whole pile per entry).
 *     every neuron, at every place b of its plate: up-subset u, down-subset d (as g-bit masks over the group's lines);
 *       both lit: combine c = T[u] - T[d] (one take-away, the down pile's signed digits), then one read of c;
 *       one lit: one read of T[u] (or a take-away of T[d]). The read adds the ready pile shifted to place b into the
 *       neuron's pile; counted in the ready pile's signed digits (a shift moves digits, it adds none).
 *     the half table: a lookup of a subset holding the last line (and not the whole group) is made by the flip,
 *       T[S] = T[G] - T[G\S]: one more take-away, counted in T[G\S]'s signed digits.
 * Signed digits of a whole number a: popcount(a ^ 3a) (the same rule as table_bits.naf).
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <dispatch/dispatch.h>

enum { BF_PH, BF_N, BH_PH, BH_N, FB_PH, FB_N, RD_PH, RD_N, CB_PH, CB_N, FL_PH, FL_N, NSZ0 };   /* then 17 sums of nafT by |S| */
#define NC (NSZ0 + 17)

static inline int64_t sd(int64_t v) { uint64_t a = v < 0 ? (uint64_t)(-v) : (uint64_t)v; return __builtin_popcountll(a ^ (3 * a)); }

typedef struct {
    int64_t P, N, M, G, gmax;
    const int32_t *x;          /* P x N light */
    const int32_t *glines;     /* the groups' lines, group after group, each group's lines in order (bit t = line t) */
    const int64_t *goff;       /* G + 1 */
    const uint64_t *cells;     /* u | d << 16 | b << 32, by (group, neuron, place) */
    const int64_t *coff;       /* G * M + 1 */
    int64_t *pile;             /* P x M */
    int64_t *cnt;              /* P x NC */
} job_t;

static void one_position(void *ctx, size_t p) {
    const job_t *J = (const job_t *)ctx;
    int64_t *T = (int64_t *)malloc(sizeof(int64_t) << J->gmax);
    uint8_t *nT = (uint8_t *)malloc((size_t)1 << J->gmax);
    const int32_t *xp = J->x + p * J->N;
    int64_t *pr = J->pile + p * J->M; memset(pr, 0, sizeof(int64_t) * J->M);
    int64_t c[NC]; memset(c, 0, sizeof c);
    for (int64_t k = 0; k < J->G; k++) {
        const int32_t *L = J->glines + J->goff[k]; int gk = (int)(J->goff[k + 1] - J->goff[k]);
        uint32_t mask = (1u << gk) - 1, top = 1u << (gk - 1);
        /* the table, Gray-code order */
        T[0] = 0; uint32_t prev = 0;
        for (uint32_t i = 1; i <= mask; i++) {
            int t = __builtin_ctz(i); uint32_t gi = i ^ (i >> 1); int64_t xv = xp[L[t]], ph = sd(xv);
            T[gi] = ((gi >> t) & 1) ? T[prev] + xv : T[prev] - xv;
            c[BF_PH] += ph; c[BF_N] += 1;
            if (i < top) { c[BH_PH] += ph; c[BH_N] += 1; }
            prev = gi;
        }
        c[BH_PH] += sd(xp[L[gk - 1]]); c[BH_N] += 1;                 /* the half table's group total */
        for (uint32_t S = 0; S <= mask; S++) { nT[S] = (uint8_t)sd(T[S]); c[NSZ0 + __builtin_popcount(S)] += nT[S]; }
        for (uint32_t S = 1; S < top; S++) { c[FB_PH] += nT[S]; c[FB_N] += 1; }   /* the other half by the flip */
        /* the reads */
        const int64_t *co = J->coff + k * J->M;
        for (int64_t j = 0; j < J->M; j++) {
            int64_t acc = 0;
            for (int64_t q = co[j]; q < co[j + 1]; q++) {
                uint64_t cell = J->cells[q]; uint32_t u = cell & 0xFFFF, d = (cell >> 16) & 0xFFFF; int b = (int)(cell >> 32);
                int64_t v;
                if (u && d) { v = T[u] - T[d]; c[CB_PH] += nT[d]; c[CB_N] += 1; c[RD_PH] += sd(v); }
                else if (u) { v = T[u]; c[RD_PH] += nT[u]; }
                else { v = -T[d]; c[RD_PH] += nT[d]; }
                c[RD_N] += 1;
                acc += v * ((int64_t)1 << b);
                if ((u & top) && u != mask) { c[FL_PH] += nT[u ^ mask]; c[FL_N] += 1; }
                if ((d & top) && d != mask) { c[FL_PH] += nT[d ^ mask]; c[FL_N] += 1; }
            }
            pr[j] += acc;
        }
    }
    memcpy(J->cnt + p * NC, c, sizeof c);
    free(T); free(nT);
}

int nc(void) { return NC; }

void group_machine(int64_t P, int64_t N, int64_t M, int64_t G, int64_t gmax, const int32_t *x, const int32_t *glines,
                   const int64_t *goff, const uint64_t *cells, const int64_t *coff, int64_t *pile, int64_t *cnt) {
    job_t J = {P, N, M, G, gmax, x, glines, goff, cells, coff, pile, cnt};
    dispatch_apply_f((size_t)P, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &J, one_position);
}

/* the direct product, whole numbers: out[p, j] = sum_i x[p, i] w[i, j] (int64) */
typedef struct { int64_t P, N, M; const int32_t *x; const int16_t *w; int64_t *out; } dj_t;
static void direct_row(void *ctx, size_t p) {
    const dj_t *D = (const dj_t *)ctx; int64_t *o = D->out + p * D->M; memset(o, 0, sizeof(int64_t) * D->M);
    for (int64_t i = 0; i < D->N; i++) {
        int64_t xv = D->x[p * D->N + i]; if (!xv) continue;
        const int16_t *wr = D->w + i * D->M;
        for (int64_t j = 0; j < D->M; j++) o[j] += xv * (int64_t)wr[j];
    }
}
void direct(int64_t P, int64_t N, int64_t M, const int32_t *x, const int16_t *w, int64_t *out) {
    dj_t D = {P, N, M, x, w, out};
    dispatch_apply_f((size_t)P, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &D, direct_row);
}
