/* Checker's own counter (whole numbers only). Nothing shared with qwen_fringes/walk2.c or check.py.
   Signed-digit (non-adjacent) weight by Prodinger's form: h = n >> 1, c = h ^ (n + h), weight = popcount(c).

   plate_counts: for every position p, every line i with x != 0 and every neuron j, the product r = x w[i][j] formed
     directly (no walk):  POUR1[p] += ones(|r|), POURSD[p] += naf(|r|).
     For W1 (each distinct running product written once): DV holds each line's distinct weight values (offsets DO):
     WRITE1[p] += ones(|x v|) over the distinct values v of the line.
   band_parts: for one position, every word j: PILE[j] = sum_i x_i T[i][j]; for each band b (places p_b .. p_b + width-1
     of |r|): DN[j][b] = sum_i sign(r) part_b(|r|), HP[j][b] = sum_i ones(part_b(|r|)). Words taken in blocks of 512. */
#include <stdint.h>
#include <dispatch/dispatch.h>
static inline uint64_t mg(int64_t v) { return v < 0 ? (uint64_t)(-v) : (uint64_t)v; }
static inline int64_t nafw(uint64_t n) { uint64_t h = n >> 1; return __builtin_popcountll(h ^ (n + h)); }

void plate_counts(int64_t P, int64_t L, int64_t N, const int64_t *X, const int16_t *W, const int64_t *DV, const int64_t *DO,
                  int64_t *POUR1, int64_t *POURSD, int64_t *WRITE1) {
    dispatch_apply((size_t)P, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t pp) {
        int64_t p = (int64_t)pp, a = 0, s = 0, w1 = 0;
        for (int64_t i = 0; i < L; i++) {
            int64_t x = X[p * L + i]; if (!x) continue;
            const int16_t *w = W + i * N;
            for (int64_t j = 0; j < N; j++) { uint64_t r = mg(x * (int64_t)w[j]); a += __builtin_popcountll(r); s += nafw(r); }
            for (int64_t k = DO[i]; k < DO[i + 1]; k++) w1 += __builtin_popcountll(mg(x * DV[k]));
        }
        POUR1[p] = a; POURSD[p] = s; WRITE1[p] = w1;
    });
}

void band_parts(int64_t L, int64_t N, const int64_t *x, const int16_t *T, const int64_t *PB, const int64_t *MASK, int64_t NB,
                int64_t *PILE, int32_t *DN, int32_t *HP) {
    int64_t nblk = (N + 511) / 512;
    dispatch_apply((size_t)nblk, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t bb) {
        int64_t j0 = (int64_t)bb * 512, j1 = j0 + 512 < N ? j0 + 512 : N;
        for (int64_t j = j0; j < j1; j++) { PILE[j] = 0; for (int64_t b = 0; b < NB; b++) { DN[j * NB + b] = 0; HP[j * NB + b] = 0; } }
        for (int64_t i = 0; i < L; i++) {
            int64_t xi = x[i]; if (!xi) continue;
            const int16_t *t = T + i * N;
            for (int64_t j = j0; j < j1; j++) {
                int64_t r = xi * (int64_t)t[j]; uint64_t m = mg(r); PILE[j] += r;
                for (int64_t b = 0; b < NB; b++) {
                    uint64_t part = (m >> PB[b]) & (uint64_t)MASK[b];
                    DN[j * NB + b] += r < 0 ? -(int32_t)part : (int32_t)part; HP[j * NB + b] += __builtin_popcountll(part);
                }
            }
        }
    });
}
