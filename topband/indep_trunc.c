/* INDEPENDENT CHECKER'S KERNEL (never reads band_walk.c). No walk: each exposure is formed directly as x * w.

   For every position p and word w, over the lit lines i (x_i != 0): r = x_i * w_(i,w), a = |r| (< 2^30 by format).
     TR[p][w][k] = sum over lines of sign(r) * (a >> Q[k])     Q = {26, 22, 18, 14, 10, 6, 2, 0}
                   -> TR << Q[k] is the pile with every exposure cut toward 0 at place Q[k]: the counted pile after
                      band k. TR[..][7] is the whole pile.
     PH[p][w][k] = sum over lines of the 1s of band k's places of a, band k's places taken as
                   (a >> Q[k]) - ((a >> Q[k-1]) << (Q[k-1] - Q[k]))   (Q[-1] = 30).
   BIG[p] counts any a >= 2^30 (never expected). Words are worked in blocks, positions side by side. */
#include <stdint.h>
#include <dispatch/dispatch.h>

static const int Q[8] = {26, 22, 18, 14, 10, 6, 2, 0};

void trunc_sums(int64_t P, int64_t L, int64_t N, const int64_t *X, const int16_t *TW,
                int64_t *TR, int32_t *PH, int64_t *BIG) {
    const int64_t BLK = 1024;
    const int64_t NBLK = (N + BLK - 1) / BLK;
    dispatch_apply((size_t)(P * NBLK), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t job) {
        int64_t p = (int64_t)job / NBLK, blk = (int64_t)job % NBLK;
        int64_t w0 = blk * BLK, w1 = w0 + BLK < N ? w0 + BLK : N;
        const int64_t *x = X + p * L;
        int64_t lit[4096]; int64_t xv[4096]; int64_t nl = 0;
        for (int64_t i = 0; i < L; i++) if (x[i] != 0) { lit[nl] = i; xv[nl] = x[i]; nl++; }
        int64_t big = 0;
        for (int64_t w = w0; w < w1; w++) {
            const int16_t *t = TW + w * L;
            int64_t acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            int32_t ph[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (int64_t j = 0; j < nl; j++) {
                int64_t r = xv[j] * (int64_t)t[lit[j]];
                int64_t a = r < 0 ? -r : r;
                if (a >= (1LL << 30)) big++;
                int64_t prev = a >> 30;
                int prevq = 30;
                for (int k = 0; k < 8; k++) {
                    int64_t c = a >> Q[k];
                    acc[k] += r < 0 ? -c : c;
                    ph[k] += __builtin_popcountll((uint64_t)(c - (prev << (prevq - Q[k]))));
                    prev = c; prevq = Q[k];
                }
            }
            for (int k = 0; k < 8; k++) { TR[(p * N + w) * 8 + k] = acc[k]; PH[(p * N + w) * 8 + k] = ph[k]; }
        }
        if (big) __atomic_fetch_add(&BIG[p], big, __ATOMIC_RELAXED);
    });
}
