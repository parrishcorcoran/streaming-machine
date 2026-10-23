/* qwen_fringes.py's two walks, whole numbers only.

   walk2: WAY 2 AT ONE PLATE, WITH THE POURS IN SIGNED DIGITS (additions a and b).
   Every line i has its weights over the plate's N neurons lined up once: LOW[i] the lowest weight, G[i][k] the gap from
   rank k-1 to rank k (k >= 1, never negative, below 2^16), NR[i][k] the neuron standing at rank k, TOP[i] the top weight.
   Per token the light on line i is one whole number x, and 3x (the light's own signed-digit writing, 3x = x + 2x, made
   once a bank and counted there). Walking up the ranks, TWO running products on the same gaps:
       r_0 = x LOW,  r_k = r_(k-1) + x g_k          (a: way 2's walk; r_k = x w of the neuron at rank k)
       t_0 = 3x LOW, t_k = t_(k-1) + 3x g_k         (b: the 3x walk; t_k = 3 r_k, CHECKED at every rank, never assumed)
   r_k is poured into the pile of the neuron at rank k. Its signed digits are read off r and t as the light's are off x
   and 3x: up = (t & ~r) >> 1, down = (r & ~t) >> 1 on the sizes, so the lit digits poured are ones(|r| ^ |t|).
   Counted per position:
     POUR1  ones(|r_k|) for every pour             (a: the pour in plain 1s)
     POURS  ones(|r_k| ^ |t_k|) for every pour     (b: the same pour in signed digits)
     WRITE1 ones(|r_k|) once for each new value of r on the line (rank 0 and every nonzero gap)
            (SIDE COUNT only: the other way to write each running product, 3r = r + 2r, once a value)
   Walking down (the order-free check): r = x TOP, t = 3x TOP; pour at the top rank; going down r -= x g_k, t -= 3x g_k;
   after rank 0 (taking off x LOW and 3x LOW) both must close at exactly 0 (OPEN counts a walk that does not).
   Sizes: |x| <= 2^15, |w| < 2^15, gaps < 2^16: |r| < 2^31, |t| < 2^33, every pile below 2^43.
   Lines are walked in the order LO gives. Position blocks run side by side, each writing only its own pile rows.

   band2: THE WORD TABLE'S WALK ONCE, EACH EXPOSURE SPLIT INTO ITS BANDS (addition c, topband/band_walk.c's kernel with
   ranks in 32 bits for 151,936 words and the gaps kept as uint16 beside LOW).
   Each exposure |r| < 2^30 has 30 places, split into 8 bands fixed by that format (topband/): places 29-26, 25-22, 21-18,
   17-14, 13-10, 9-6, 5-2, 1-0. For each word and band:
       DN[word][b] = sum over lines of sign(r) (|r| >> p_b & mask_b)      (the band's part is DN << p_b)
       HN[word][b] = sum over lines of ones(|r| >> p_b & mask_b)          (the photons that part pours)
   |DN| <= 1024 x 15 = 15,360 and HN <= 1024 x 4 = 4,096: both fit 16 bits.
   THE SIMULATION'S SHORTCUT (as topband's): every band of every word is formed so the run can check; the machine pours
   band b only into the words still in at band b, and only those photons are counted (qwen_fringes.py). */
#include <stdint.h>
#include <dispatch/dispatch.h>

static inline uint64_t mag(int64_t v) { return v < 0 ? (uint64_t)(-v) : (uint64_t)v; }
static inline int64_t pc(uint64_t a) { return __builtin_popcountll(a); }

void walk2(int64_t P, int64_t L, int64_t N, const int64_t *X, const int64_t *LOW, const uint16_t *G, const uint16_t *NR,
           const int64_t *TOP, const int64_t *LO, int dir, int64_t blk,
           int64_t *PILE, int64_t *POUR1, int64_t *POURS, int64_t *WRITE1, int64_t *OPEN, int64_t *BAD3) {
    int64_t nblk = (P + blk - 1) / blk;
    dispatch_apply((size_t)nblk, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t b) {
        int64_t p0 = (int64_t)b * blk, p1 = p0 + blk < P ? p0 + blk : P;
        for (int64_t li = 0; li < L; li++) {
            int64_t i = LO[li]; const uint16_t *g = G + i * N; const uint16_t *nr = NR + i * N;
            for (int64_t p = p0; p < p1; p++) {
                int64_t x = X[p * L + i];
                if (x == 0) continue;                                      /* a dark line: nothing met, nothing poured */
                int64_t x3 = 3 * x, *pile = PILE + p * N, s1 = 0, ss = 0, w1 = 0, bad = 0;
                if (dir > 0) {
                    int64_t r = x * LOW[i], t = x3 * LOW[i];
                    w1 += pc(mag(r));
                    pile[nr[0]] += r; s1 += pc(mag(r)); ss += pc(mag(r) ^ mag(t)); bad += (t != 3 * r);
                    for (int64_t k = 1; k < N; k++) {
                        int64_t gk = g[k];
                        if (gk) { r += x * gk; t += x3 * gk; w1 += pc(mag(r)); }
                        pile[nr[k]] += r; s1 += pc(mag(r)); ss += pc(mag(r) ^ mag(t)); bad += (t != 3 * r);
                    }
                } else {
                    int64_t r = x * TOP[i], t = x3 * TOP[i];
                    w1 += pc(mag(r));
                    for (int64_t k = N - 1; k >= 1; k--) {
                        pile[nr[k]] += r; s1 += pc(mag(r)); ss += pc(mag(r) ^ mag(t)); bad += (t != 3 * r);
                        int64_t gk = g[k];
                        if (gk) { r -= x * gk; t -= x3 * gk; w1 += pc(mag(r)); }
                    }
                    pile[nr[0]] += r; s1 += pc(mag(r)); ss += pc(mag(r) ^ mag(t)); bad += (t != 3 * r);
                    r -= x * LOW[i]; t -= x3 * LOW[i];
                    if (r != 0 || t != 0) OPEN[p] += 1;                    /* the walk down did not close: never expected */
                }
                POUR1[p] += s1; POURS[p] += ss; WRITE1[p] += w1; BAD3[p] += bad;
            }
        }
    });
}

#define NB 8
static const int PB[NB] = {26, 22, 18, 14, 10, 6, 2, 0};
static const uint64_t MB[NB] = {15, 15, 15, 15, 15, 15, 15, 3};
static inline void bpour(int64_t r, int16_t *dn, uint16_t *hn) {
    uint64_t a = mag(r); int neg = r < 0;
    for (int b = 0; b < NB; b++) {
        uint64_t nib = (a >> PB[b]) & MB[b];
        dn[b] += neg ? -(int16_t)nib : (int16_t)nib;
        hn[b] += (uint16_t)__builtin_popcountll(nib);
    }
}

void band2(int64_t P, int64_t L, int64_t N, const int64_t *X, const int64_t *LOW, const uint16_t *G, const uint32_t *NR,
           const int64_t *TOP, const int64_t *LO, int dir, int64_t blk, int16_t *DN, uint16_t *HN, int64_t *OPEN, int64_t *BIG) {
    int64_t nblk = (P + blk - 1) / blk;
    dispatch_apply((size_t)nblk, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t b) {
        int64_t p0 = (int64_t)b * blk, p1 = p0 + blk < P ? p0 + blk : P;
        for (int64_t li = 0; li < L; li++) {
            int64_t i = LO[li]; const uint16_t *g = G + i * N; const uint32_t *nr = NR + i * N;
            for (int64_t p = p0; p < p1; p++) {
                int64_t x = X[p * L + i];
                if (x == 0) continue;
                int16_t *dn = DN + p * N * NB; uint16_t *hn = HN + p * N * NB;
                if (dir > 0) {
                    int64_t r = x * LOW[i];
                    for (int64_t k = 0; k < N; k++) {
                        if (k) r += x * (int64_t)g[k];
                        if (r >= (1LL << 30) || r <= -(1LL << 30)) BIG[p] += 1;   /* never expected */
                        bpour(r, dn + (int64_t)nr[k] * NB, hn + (int64_t)nr[k] * NB);
                    }
                } else {
                    int64_t r = x * TOP[i];
                    for (int64_t k = N - 1; k >= 0; k--) {
                        if (r >= (1LL << 30) || r <= -(1LL << 30)) BIG[p] += 1;
                        bpour(r, dn + (int64_t)nr[k] * NB, hn + (int64_t)nr[k] * NB);
                        r -= k ? x * (int64_t)g[k] : x * LOW[i];
                    }
                    if (r != 0) OPEN[p] += 1;                              /* the walk down did not close */
                }
            }
        }
    });
}
