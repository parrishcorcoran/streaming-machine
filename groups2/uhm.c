/* groups2/uhm.c -- an independent group machine for the plate in up/hole form, and for a plate whose places are each
 * either up/hole or signed digits (written for groups2/, sharing no code with groups/gm_uh.c or groups/gm.c).
 *
 * Whole numbers only (int64). One position per task, positions spread over the cores.
 *
 * Per position, per group of gk light lines (the group's lines in the given order, bit t = the group's line t):
 *   the table of up/hole piles, V[p] = (light on the lines up in p) - (light on the lines that are holes in p), for
 *   every pattern p, made here by the lowest-bit rule V[0] = -(group total), V[p] = V[p with its lowest bit cleared]
 *   + 2 x(line of that bit). (This is only how the values are made here; the work of the table is counted outside, in
 *   the Python, by its own relation.)
 *   Every neuron j, at every up/hole place b of j (bit b of upplaces[j]), reads V[pattern(group, j, b)] shifted to b
 *   into its doubled pile.
 * Then every lit signed-digit place of j (a list per neuron: line, place, sign) adds +-x(line) shifted to place + 1
 * into the doubled pile (the doubling is a shift), and the doubled pile takes kcorr[j] x X (X = the position's light
 * summed). The pile is the doubled pile halved; an odd doubled pile is counted (must stay 0).
 *
 * Counted here, per position and per segment of neurons (a segment = one plate of a slab):
 *   C_RDH  read photons by the table rule: for each group, sum over patterns of (reads of p) x signed digits of V[p],
 *          with the signed digits counted digit by digit (odd: digit 2 - (a mod 4), take it away; halve);
 *   C_RDP  the same read photons counted read by read, by popcount(a ^ 3a) (a second count of the same thing);
 *   C_RDN  the number of reads;  C_ODD  odd doubled piles.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <dispatch/dispatch.h>

enum { C_RDH, C_RDP, C_RDN, C_ODD, NCNT };

static int sd_digit_by_digit(int64_t v) {
    uint64_t a = v < 0 ? (uint64_t)0 - (uint64_t)v : (uint64_t)v; int c = 0;
    while (a) { if (a & 1u) { c++; if (a & 2u) a += 1; else a -= 1; } a >>= 1; }
    return c;
}
static inline int sd_pop(int64_t v) { uint64_t a = v < 0 ? (uint64_t)0 - (uint64_t)v : (uint64_t)v; return __builtin_popcountll(a ^ (3 * a)); }

typedef struct {
    int64_t P, N, M, G, gmax, nseg;
    const int32_t *x;          /* P x N */
    const int32_t *glines;     /* N: the groups' lines, group after group */
    const int64_t *goff;       /* G + 1 */
    const uint32_t *upplaces;  /* M: bit b set = place b of neuron j is read up/hole */
    const uint16_t *pat;       /* G x M x 16 */
    const int64_t *hist;       /* nseg x G x 2^gmax: how many reads of each pattern (plate only) */
    const int8_t *seg;         /* M */
    const int64_t *toff;       /* M + 1: the signed-digit places of each neuron */
    const int32_t *tline; const int8_t *tplace; const int8_t *tsign;
    const int64_t *kcorr;      /* M */
    int64_t *pile;             /* P x M */
    int64_t *cnt;              /* P x nseg x NCNT */
} job_t;

static void one_position(void *ctx, size_t p) {
    const job_t *J = (const job_t *)ctx;
    int64_t T = (int64_t)1 << J->gmax;
    int64_t *V = (int64_t *)malloc(sizeof(int64_t) * T);
    int32_t *sV = (int32_t *)malloc(sizeof(int32_t) * T);
    int64_t *acc = (int64_t *)calloc((size_t)J->M, sizeof(int64_t));
    int64_t *c = J->cnt + p * J->nseg * NCNT; memset(c, 0, sizeof(int64_t) * J->nseg * NCNT);
    const int32_t *xp = J->x + p * J->N;
    int64_t X = 0; for (int64_t i = 0; i < J->N; i++) X += xp[i];
    for (int64_t k = 0; k < J->G; k++) {
        const int32_t *L = J->glines + J->goff[k]; int gk = (int)(J->goff[k + 1] - J->goff[k]);
        int64_t n = (int64_t)1 << gk, TG = 0;
        for (int t = 0; t < gk; t++) TG += xp[L[t]];
        V[0] = -TG;
        for (int64_t q = 1; q < n; q++) V[q] = V[q & (q - 1)] + 2 * (int64_t)xp[L[__builtin_ctzll((uint64_t)q)]];
        for (int64_t q = 0; q < n; q++) sV[q] = sd_digit_by_digit(V[q]);
        for (int64_t s = 0; s < J->nseg; s++) {
            const int64_t *h = J->hist + (s * J->G + k) * T; int64_t ph = 0;
            for (int64_t q = 0; q < n; q++) ph += h[q] * sV[q];
            c[s * NCNT + C_RDH] += ph;
        }
        const uint16_t *pk = J->pat + k * J->M * 16;
        for (int64_t j = 0; j < J->M; j++) {
            uint32_t up = J->upplaces[j]; if (!up) continue;
            const uint16_t *pj = pk + j * 16; int64_t a = 0, ph = 0, nr = 0;
            while (up) {
                int b = __builtin_ctz(up); up &= up - 1;
                int64_t v = V[pj[b]]; a += v * ((int64_t)1 << b); ph += sd_pop(v); nr++;
            }
            acc[j] += a; int s = J->seg[j]; c[s * NCNT + C_RDP] += ph; c[s * NCNT + C_RDN] += nr;
        }
    }
    int64_t *pr = J->pile + p * J->M;
    for (int64_t j = 0; j < J->M; j++) {
        int64_t a = acc[j];
        for (int64_t q = J->toff[j]; q < J->toff[j + 1]; q++)
            a += (int64_t)J->tsign[q] * (int64_t)xp[J->tline[q]] * ((int64_t)1 << (J->tplace[q] + 1));
        a += J->kcorr[j] * X;
        if (a & 1) c[J->seg[j] * NCNT + C_ODD] += 1;
        pr[j] = a >> 1;
    }
    free(V); free(sV); free(acc);
}
int ncnt(void) { return NCNT; }
void uh_machine(int64_t P, int64_t N, int64_t M, int64_t G, int64_t gmax, int64_t nseg, const int32_t *x,
                const int32_t *glines, const int64_t *goff, const uint32_t *upplaces, const uint16_t *pat,
                const int64_t *hist, const int8_t *seg, const int64_t *toff, const int32_t *tline, const int8_t *tplace,
                const int8_t *tsign, const int64_t *kcorr, int64_t *pile, int64_t *cnt) {
    job_t J = {P, N, M, G, gmax, nseg, x, glines, goff, upplaces, pat, hist, seg, toff, tline, tplace, tsign, kcorr, pile, cnt};
    dispatch_apply_f((size_t)P, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &J, one_position);
}

/* the plain product, for the shuffled light: out[p, j] = sum_i x[p, i] w[i, j] (int64), lines met one by one */
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
