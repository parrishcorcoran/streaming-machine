/* The group machine with the plate in its up/hole form (updown_machine.py's form): every place of every weight is a
 * photon up (+1) or a hole (-1): D = 2w + 1 = sum over 16 places of 2^b s_b, s_b = +1 where w + 2^15 has a 1, -1 where
 * it has a 0. So at one place, a neuron's g lines form one up/hole pattern p (bit t = 1: line t up), never an empty one,
 * and the pattern's pile V[p] = sum of up light - sum of hole light. The flip: the complementary pattern (ups and holes
 * exchanged) gives exactly the opposite pile, V[p ^ mask] = -V[p], so only half the patterns need building and the other
 * half is read as a take-away of the same pile (no extra work).
 * The pile: x . w = (sum over groups and places of 2^b V[p] - X) / 2, X = the sum of all the position's light values.
 * Counted (signed digits of what is added): the table by Gray steps (each step turns one line from hole to up or back:
 * 2 x, the value's signed digits); the half table (patterns with the group's last line up: its first entry, all lines
 * up, is the group total, g - 1 adds, then Gray steps over the other g - 1 lines); every read (the ready pile's signed
 * digits); X once a position (N - 1 adds of light values); X taken from each neuron's pile (X's signed digits); the
 * halving is a shift. Whole numbers only (int64). */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <dispatch/dispatch.h>

enum { BF_PH, BF_N, BH_PH, BH_N, RD_PH, RD_N, X_PH, X_N, OFF_PH, OFF_N, ODD };
#define NCU 11
static inline int64_t sd(int64_t v) { uint64_t a = v < 0 ? (uint64_t)(-v) : (uint64_t)v; return __builtin_popcountll(a ^ (3 * a)); }

typedef struct {
    int64_t P, N, M, G, gmax;
    const int32_t *x; const int32_t *glines; const int64_t *goff;
    const uint16_t *pat;       /* G x M x 16: the up/hole pattern of each (group, neuron, place) */
    int64_t *pile; int64_t *cnt;
} job_t;

static void one_position(void *ctx, size_t p) {
    const job_t *J = (const job_t *)ctx;
    int64_t *V = (int64_t *)malloc(sizeof(int64_t) << J->gmax);
    const int32_t *xp = J->x + p * J->N;
    int64_t *pr = J->pile + p * J->M; memset(pr, 0, sizeof(int64_t) * J->M);
    int64_t c[NCU]; memset(c, 0, sizeof c);
    int64_t X = 0;
    for (int64_t i = 0; i < J->N; i++) { X += xp[i]; if (i) { c[X_PH] += sd(xp[i]); c[X_N] += 1; } }
    for (int64_t k = 0; k < J->G; k++) {
        const int32_t *L = J->glines + J->goff[k]; int gk = (int)(J->goff[k + 1] - J->goff[k]);
        uint32_t mask = (1u << gk) - 1, top = 1u << (gk - 1);
        int64_t TG = 0; for (int t = 0; t < gk; t++) TG += xp[L[t]];
        /* full table: start at all holes (-TG; its g - 1 adds counted with the half table's start), Gray steps */
        V[0] = -TG; uint32_t prev = 0;
        for (uint32_t i = 1; i <= mask; i++) {
            int t = __builtin_ctz(i); uint32_t gi = i ^ (i >> 1); int64_t xv = xp[L[t]];
            V[gi] = ((gi >> t) & 1) ? V[prev] + 2 * xv : V[prev] - 2 * xv;
            c[BF_PH] += sd(xv); c[BF_N] += 1; prev = gi;
        }
        for (int t = 1; t < gk; t++) { c[BF_PH] += sd(xp[L[t]]); c[BF_N] += 1; }        /* the start, -TG */
        /* half table: the patterns with the last line up; first entry all up = TG (g - 1 adds), then Gray steps over
           the other g - 1 lines: 2^(g-1) - 1 steps */
        for (int t = 1; t < gk; t++) { c[BH_PH] += sd(xp[L[t]]); c[BH_N] += 1; }
        for (uint32_t i = 1; i < top; i++) { int t = __builtin_ctz(i); c[BH_PH] += sd(xp[L[t]]); c[BH_N] += 1; }
        const uint16_t *pk = J->pat + k * J->M * 16;
        for (int64_t j = 0; j < J->M; j++) {
            const uint16_t *pj = pk + j * 16; int64_t acc = 0;
            for (int b = 0; b < 16; b++) { int64_t v = V[pj[b]]; acc += v * ((int64_t)1 << b); c[RD_PH] += sd(v); c[RD_N] += 1; }
            pr[j] += acc;
        }
    }
    int64_t xs = sd(X);
    for (int64_t j = 0; j < J->M; j++) {
        int64_t D = pr[j] - X; c[OFF_PH] += xs; c[OFF_N] += 1;
        if (D & 1) c[ODD] += 1;
        pr[j] = D / 2;                     /* exact: D is even (checked by ODD) */
    }
    memcpy(J->cnt + p * NCU, c, sizeof c);
    free(V);
}
int ncu(void) { return NCU; }
void group_machine_uh(int64_t P, int64_t N, int64_t M, int64_t G, int64_t gmax, const int32_t *x, const int32_t *glines,
                      const int64_t *goff, const uint16_t *pat, int64_t *pile, int64_t *cnt) {
    job_t J = {P, N, M, G, gmax, x, glines, goff, pat, pile, cnt};
    dispatch_apply_f((size_t)P, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &J, one_position);
}
