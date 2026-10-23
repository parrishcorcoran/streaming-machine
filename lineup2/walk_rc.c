/* WAY 2, THE LINE-UP ACROSS THE NEURONS ON EACH LINE: the walk, in whole numbers only.

   For one plate: every line i has its weights over all neurons lined up once (G: each line's gaps in rank order, the
   first entry the lowest weight itself; NR: the neuron standing at each rank). Per token the light on line i is one
   whole number x. Walking up the ranks:
       r_0 = x * w_(0);   r_k = r_(k-1) + x * gap_k;   r_k is poured into the pile of the neuron at rank k.
   Walking down (the order-free check): r = x * w_(top); pour r at the top rank; r = r - x * gap_k going down; after
   rank 0 the walk must close at exactly 0 (checked, not counted).
   Every product here is a whole-number product (|x| < 2^16, |gap| < 2^17: below 2^33), every pile below 2^47.
   Counted per position (only when count != 0): the photons poured, ones(|r|) (each running product added into its
   pile), and as a side count the lit places of r in single waves, ones(a ^ 3a).
   The lines are walked in the order LO gives (any order: the order-free check passes a random one).
   Positions are split into blocks; the blocks run side by side (each block writes only its own pile rows). */
#include <stdint.h>
#include <dispatch/dispatch.h>

static inline int64_t ones64(int64_t v) { uint64_t a = v < 0 ? (uint64_t)(-v) : (uint64_t)v; return __builtin_popcountll(a); }
static inline int64_t naf64(int64_t v) { uint64_t a = v < 0 ? (uint64_t)(-v) : (uint64_t)v; return __builtin_popcountll(a ^ (3 * a)); }

void walk(int64_t P, int64_t L, int64_t N, const int64_t *X, const int32_t *G, const uint16_t *NR, const int64_t *TOP,
          const int64_t *LO, int dir, int count, int64_t blk, int64_t *PILE, int64_t *POUR1, int64_t *POURN, int64_t *OPEN) {
    int64_t nblk = (P + blk - 1) / blk;
    dispatch_apply((size_t)nblk, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t b) {
        int64_t p0 = (int64_t)b * blk, p1 = p0 + blk < P ? p0 + blk : P;
        for (int64_t li = 0; li < L; li++) {
            int64_t i = LO[li]; const int32_t *g = G + i * N; const uint16_t *nr = NR + i * N;
            for (int64_t p = p0; p < p1; p++) {
                int64_t x = X[p * L + i];
                if (x == 0) continue;                                  /* a dark line: nothing meets, nothing poured */
                int64_t *pile = PILE + p * N, s1 = 0, sn = 0;
                if (dir > 0) {
                    int64_t r = 0;
                    if (count) for (int64_t k = 0; k < N; k++) { r += x * (int64_t)g[k]; pile[nr[k]] += r; s1 += ones64(r); sn += naf64(r); }
                    else       for (int64_t k = 0; k < N; k++) { r += x * (int64_t)g[k]; pile[nr[k]] += r; }
                } else {
                    int64_t r = x * TOP[i];
                    for (int64_t k = N - 1; k >= 0; k--) {
                        pile[nr[k]] += r;
                        if (count) { s1 += ones64(r); sn += naf64(r); }
                        r -= x * (int64_t)g[k];
                    }
                    if (r != 0) OPEN[p] += 1;                          /* the walk down did not close: never expected */
                }
                if (count) { POUR1[p] += s1; POURN[p] += sn; }
            }
        }
    });
}
