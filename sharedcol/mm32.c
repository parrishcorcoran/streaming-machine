/* sharedcol/cut.py's whole-number product: the light at the top (int64, |x| <= 2^15) meeting a table stored as int32
   whole numbers (the fringes, some words wider than 16 places), every sum an int64 sum, no float.
       OUT[p][n] = sum over lanes i of X[p][i] * W[i][n]
   With count != 0 it also counts, per position, the photons of every exposure x * w (ones of |x w|): way 2's pours,
   what a line-up would pour into every word's pile (count only; the product itself does not use it).
   Bounds (asserted by the caller): |x| <= 2^15, |w| < 2^24, at most 1,024 lanes: every pile below 2^50. */
#include <stdint.h>
#include <string.h>
#include <dispatch/dispatch.h>

static inline int64_t ones64(int64_t v) { uint64_t a = v < 0 ? (uint64_t)(-v) : (uint64_t)v; return __builtin_popcountll(a); }

void mm32(int64_t P, int64_t L, int64_t N, const int64_t *X, const int32_t *W, int64_t *OUT, int64_t *POUR, int count) {
    dispatch_apply((size_t)P, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t p) {
        int64_t *o = OUT + (int64_t)p * N; memset(o, 0, (size_t)N * sizeof(int64_t)); int64_t s1 = 0;
        for (int64_t i = 0; i < L; i++) {
            int64_t x = X[(int64_t)p * L + i]; if (x == 0) continue;     /* a dark lane: nothing met, nothing poured */
            const int32_t *w = W + i * N;
            if (count) for (int64_t n = 0; n < N; n++) { int64_t r = x * (int64_t)w[n]; o[n] += r; s1 += ones64(r); }
            else       for (int64_t n = 0; n < N; n++) o[n] += x * (int64_t)w[n];
        }
        if (count) POUR[p] = s1;
    });
}
