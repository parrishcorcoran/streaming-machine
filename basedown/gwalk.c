/* basedown/gwalk.c -- the group piles walked along each group's own line-up (basedown/combine.py; whole numbers).
 * Group k (lines glines[goff[k]..goff[k+1])) has its own token order in every soup: row r of the soup is rank r, and
 * tokmap[k*P + r] is the token there. D (P x N) holds every group's gaps at their ranks (row r, the group's lines).
 * For every neuron n, the group's doubled part before the offset correction,
 *     Q'[k,n] at rank r = Q'[k,n] at rank r-1 + (2 D[r,G].w[G,n] - kcorr[n] x (D[r,G] summed))
 * (the reads of the gaps' ready piles land straight into Q'), and Q' lands in its token's pile2 (pile2 = sum over
 * groups; the machine adds kcorr x X(t) once a token and halves). The step's part is formed here by the direct
 * product of the gaps (the check path standing in for the group reads, whose sum over groups is gbank's own pile
 * of D, checked in the Python). Counted: out[0] the 1s of every Q' landed; out[1] the 1s of the true group part
 * x[t,G].w[G,n] (count only); out[2] CHECK: Q' + kcorr X_G(t) != 2 x[t,G].w[G,n] (must be 0). */
#include <stdint.h>
#include <string.h>
static inline int64_t pc(uint64_t a) { return (int64_t)__builtin_popcountll(a); }
static inline uint64_t ab(int64_t v) { return (uint64_t)(v < 0 ? -v : v); }

void gwalk(int64_t P, int64_t N, int64_t M, int64_t G, int64_t S, const int32_t *glines, const int64_t *goff,
           const int64_t *soff, const int32_t *tokmap, const int32_t *D, const int32_t *x, const int16_t *w,
           const int64_t *kcorr, int64_t *pile2, int64_t *Q, int64_t *y, int64_t *z, int64_t *out)
{
    int64_t land = 0, landq = 0, bad = 0;
    for (int64_t k = 0; k < G; k++) {
        const int32_t *L = glines + goff[k]; int64_t gk = goff[k + 1] - goff[k];
        for (int64_t s = 0; s < S; s++) {
            memset(Q, 0, (size_t)M * sizeof(int64_t));
            for (int64_t r = soff[s]; r < soff[s + 1]; r++) {
                int64_t t = tokmap[k * P + r], XD = 0, XG = 0;
                memset(y, 0, (size_t)M * sizeof(int64_t)); memset(z, 0, (size_t)M * sizeof(int64_t));
                for (int64_t i = 0; i < gk; i++) {
                    int64_t j = L[i], d = D[r * N + j], xv = x[t * N + j]; const int16_t *wr = w + j * M;
                    XD += d; XG += xv;
                    if (d) for (int64_t n = 0; n < M; n++) y[n] += 2 * d * (int64_t)wr[n];
                    if (xv) for (int64_t n = 0; n < M; n++) z[n] += xv * (int64_t)wr[n];
                }
                int64_t *pr = pile2 + t * M, l1 = 0, lq = 0, b = 0;
                for (int64_t n = 0; n < M; n++) {
                    int64_t v = Q[n] + y[n] - kcorr[n] * XD; Q[n] = v; pr[n] += v;
                    l1 += pc(ab(v)); lq += pc(ab(z[n])); b += (v + kcorr[n] * XG != 2 * z[n]);
                }
                land += l1; landq += lq; bad += b;
            }
        }
    }
    out[0] = land; out[1] = landq; out[2] = bad;
}
