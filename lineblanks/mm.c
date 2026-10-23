/* The CHECK PATH for lineblanks/way2.py: the direct product of the light (int64, |x| <= 2^15) and a plate as stored
   (int16), every sum an int64 sum, no float. OUT[p][n] = sum over lines i of X[p][i] * W[i][n].
   Bounds: |x w| < 2^30, at most 3,072 lines: every pile below 2^42. Positions run side by side. */
#include <stdint.h>
#include <string.h>
#include <dispatch/dispatch.h>

void mm16(int64_t P, int64_t L, int64_t N, const int64_t *X, const int16_t *W, int64_t *OUT) {
    dispatch_apply((size_t)P, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t p) {
        int64_t *o = OUT + (int64_t)p * N; memset(o, 0, (size_t)N * sizeof(int64_t));
        for (int64_t i = 0; i < L; i++) {
            int64_t x = X[(int64_t)p * L + i]; if (x == 0) continue;
            const int16_t *w = W + i * N;
            for (int64_t n = 0; n < N; n++) o[n] += x * (int64_t)w[n];
        }
    });
}
