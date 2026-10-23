/* WAY 2 for Qwen3-0.6B's plates and word table: the line-up across the neurons on each line, the walk, whole numbers only.
   (lineup2/walk_rc.c's walk, with ranks in 32 bits for the 151,936-word table and the gaps kept as uint16.)

   For one plate: every line i has its weights over all neurons lined up once: LOW[i] the lowest weight, G[i][k] the gap
   from rank k-1 to rank k (k >= 1, never negative, below 2^16), NR[i][k] the neuron standing at rank k. Per token the
   light on line i is one whole number x. Walking up the ranks:
       r_0 = x * LOW[i];   r_k = r_(k-1) + x * G[i][k];   r_k is poured into the pile of the neuron at rank k.
   Walking down (the order-free check): r = x * TOP[i]; pour r at the top rank; r -= x * gap going down; after rank 0
   (taking off x * LOW) the walk must close at exactly 0 (checked, not counted).
   Products: |x| <= 2^15, |weight| < 2^15, gaps < 2^16: every running product below 2^31, every pile below 2^43.
   Counted per position (count != 0): the photons poured, ones(|r_k|) for every running product added into its pile.
   Lines are walked in the order LO gives. Position blocks run side by side, each writing only its own pile rows. */
#include <stdint.h>
#include <dispatch/dispatch.h>

static inline int64_t ones64(int64_t v) { uint64_t a = v < 0 ? (uint64_t)(-v) : (uint64_t)v; return __builtin_popcountll(a); }

void walk(int64_t P, int64_t L, int64_t N, const int64_t *X, const int64_t *LOW, const uint16_t *G, const uint32_t *NR,
          const int64_t *TOP, const int64_t *LO, int dir, int count, int64_t blk, int64_t *PILE, int64_t *POUR1, int64_t *OPEN) {
    int64_t nblk = (P + blk - 1) / blk;
    dispatch_apply((size_t)nblk, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t b) {
        int64_t p0 = (int64_t)b * blk, p1 = p0 + blk < P ? p0 + blk : P;
        for (int64_t li = 0; li < L; li++) {
            int64_t i = LO[li]; const uint16_t *g = G + i * N; const uint32_t *nr = NR + i * N;
            for (int64_t p = p0; p < p1; p++) {
                int64_t x = X[p * L + i];
                if (x == 0) continue;                                  /* a dark line: nothing met, nothing poured */
                int64_t *pile = PILE + p * N, s1 = 0;
                if (dir > 0) {
                    int64_t r = x * LOW[i];
                    pile[nr[0]] += r; if (count) s1 += ones64(r);
                    if (count) for (int64_t k = 1; k < N; k++) { r += x * (int64_t)g[k]; pile[nr[k]] += r; s1 += ones64(r); }
                    else       for (int64_t k = 1; k < N; k++) { r += x * (int64_t)g[k]; pile[nr[k]] += r; }
                } else {
                    int64_t r = x * TOP[i];
                    for (int64_t k = N - 1; k >= 1; k--) {
                        pile[nr[k]] += r; if (count) s1 += ones64(r);
                        r -= x * (int64_t)g[k];
                    }
                    pile[nr[0]] += r; if (count) s1 += ones64(r);
                    r -= x * LOW[i];
                    if (r != 0) OPEN[p] += 1;                          /* the walk down did not close: never expected */
                }
                if (count) POUR1[p] += s1;
            }
        }
    });
}
