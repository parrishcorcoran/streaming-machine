/* flat_build/2/runm.c -- the bank kernel of pick 2: the plate's sign run read once in every (gate, group) pair.
 * Written for flat_build/2/machine.py. Whole numbers only (int64 inside). One position per task, spread over the cores.
 *
 * The plate: every weight w as E = w + 2^15 in 16 binary places (a 1 is up, a 0 is a hole); every place up/hole.
 * Per (group k, gate j) cell: c = the smallest c with every weight of the gate in that group in [-2^c, 2^c).
 *   Places 0..c-1 are read as they are; places c..14 hold the sign sheet (place 15) flipped, so
 *   sum over b = c..15 of 2^b (2 bit_b - 1) = 2^c (2 bit_15 - 1): the sign sheet is read ONCE, at place c.
 *
 * Per position, per random group of gk light lines (bit t of a pattern = the group's line t):
 *   the half table (the patterns with the group's last line up) is WALKED in Gray order:
 *     start: key 0 = last line up, all others holes: V = x_last - (x_0 + ... + x_{gk-2}), counted as gk - 1 adds
 *            of light values (lines 1..gk-1 added to line 0, as the relation counts the group total);
 *     step i = 1 .. 2^(gk-1) - 1: line t = ctz(i) turns (up: + 2 x_t, hole: - 2 x_t), photons = x_t's signed digits.
 *   the other half by the flip (Babinet): V[p] = -V[p ^ all].
 *   every half-table entry the plate reads (readh, plate only) is written in signed digits once: its 1s (wr).
 *   gate j: at each place b < c(k, j) one ready pile V[pat(k, j, b)] into A[j][b]; at place c one ready pile
 *   V[pat(k, j, 15)] (the sign sheet) into A[j][c]; each read's photons (the ready pile's signed digits) into H[j][b].
 * The doubled pile = sum over b of 2^b A[j][b] - X, formed in the Python; the halving is a shift.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <dispatch/dispatch.h>

static inline int sdc(int64_t v) { uint64_t a = v < 0 ? (uint64_t)0 - (uint64_t)v : (uint64_t)v; return __builtin_popcountll(a ^ (3 * a)); }
static inline int onec(int64_t v) { uint64_t a = v < 0 ? (uint64_t)0 - (uint64_t)v : (uint64_t)v; return __builtin_popcountll(a); }

typedef struct {
    int64_t P, N, M, G, gmax;
    const int32_t *x;          /* P x N */
    const int32_t *glines;     /* N */
    const int64_t *goff;       /* G + 1 */
    const uint16_t *pat;       /* G x M x 16 */
    const int8_t *cc;          /* G x M: the cell's run place c */
    const uint8_t *readh;      /* G x 2^(gmax - 1) */
    int64_t *A;                /* P x M x 16 */
    int32_t *H;                /* P x M x 16 */
    int64_t *wr;               /* P: entries read, written in signed digits (their 1s) */
    int64_t *tw;               /* P: the walk's photons */
} job_t;

static void one_position(void *ctx, size_t p) {
    const job_t *J = (const job_t *)ctx;
    const int64_t M = J->M, TH = (int64_t)1 << (J->gmax - 1);
    const int32_t *xp = J->x + p * J->N;
    int64_t *Ap = J->A + p * M * 16; int32_t *Hp = J->H + p * M * 16;
    memset(Ap, 0, sizeof(int64_t) * M * 16); memset(Hp, 0, sizeof(int32_t) * M * 16);
    int64_t w = 0, walk = 0;
    int64_t *Vh = (int64_t *)malloc(sizeof(int64_t) * TH);
    for (int64_t k = 0; k < J->G; k++) {
        const int32_t *L = J->glines + J->goff[k]; int gk = (int)(J->goff[k + 1] - J->goff[k]);
        int64_t top = (int64_t)1 << (gk - 1), full = ((int64_t)1 << gk) - 1;
        /* the walk: key = Gray code of i; the value lives at Vh[key] */
        int64_t v = xp[L[gk - 1]];
        for (int t = 0; t < gk - 1; t++) v -= xp[L[t]];
        for (int t = 1; t < gk; t++) walk += sdc(xp[L[t]]);
        int64_t key = 0; Vh[0] = v;
        for (int64_t i = 1; i < top; i++) {
            int t = __builtin_ctzll((uint64_t)i); int64_t bit = (int64_t)1 << t;
            int64_t xv = xp[L[t]];
            if (key & bit) v -= 2 * xv; else v += 2 * xv;
            key ^= bit; Vh[key] = v; walk += sdc(xv);
        }
        const uint8_t *rh = J->readh + k * TH;
        for (int64_t q = 0; q < top; q++) if (rh[q]) w += onec(Vh[q]);
        const uint16_t *pk = J->pat + k * M * 16; const int8_t *ck = J->cc + k * M;
        for (int64_t j = 0; j < M; j++) {
            const uint16_t *pj = pk + j * 16; int c = ck[j];
            int64_t *a = Ap + j * 16; int32_t *h = Hp + j * 16;
            for (int b = 0; b <= c; b++) {
                int64_t pt = (b < c) ? pj[b] : pj[15];
                int64_t val = (pt & top) ? Vh[pt - top] : -Vh[(pt ^ full) - top];
                a[b] += val; h[b] += sdc(val);
            }
        }
    }
    free(Vh);
    J->wr[p] = w; J->tw[p] = walk;
}

void bank_run(int64_t P, int64_t N, int64_t M, int64_t G, int64_t gmax, const int32_t *x, const int32_t *glines,
              const int64_t *goff, const uint16_t *pat, const int8_t *cc, const uint8_t *readh,
              int64_t *A, int32_t *H, int64_t *wr, int64_t *tw) {
    job_t J = {P, N, M, G, gmax, x, glines, goff, pat, cc, readh, A, H, wr, tw};
    dispatch_apply_f((size_t)P, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &J, one_position);
}
