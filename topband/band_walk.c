/* THE TOP TABLE, WAY 2'S WALK, EACH EXPOSURE SPLIT INTO ITS BANDS. Whole numbers only.

   For one position: every line i of the top table has its 50,257 weights lined up once (G: the gaps in rank order, the
   first entry the lowest weight itself; NR: the word standing at each rank). The light on line i is one whole number x.
   Walking up the ranks: r_0 = x * w_(0), r_k = r_(k-1) + x * gap_k. r_k is the exposure x * w of the word at rank k.
   Walking down (the order-free check): r = x * w_(top); r = r - x * gap_k going down; the walk must close at 0.

   Each exposure |r| < 2^30 (|x| < 2^15 and |w| < 2^15, asserted by the caller) has 30 places, split into 8 bands fixed
   from that format before anything ran: places 29-26, 25-22, 21-18, 17-14, 13-10, 9-6, 5-2, 1-0 (bands 0..7).
   Band b's part of r is sign(r) * (the places of |r| in band b). For each word and band this kernel keeps
       DN[word][b] = sum over lines of sign(r) * (|r| >> p_b & mask_b)      (the band's part is DN << p_b)
       HN[word][b] = sum over lines of ones(|r| >> p_b & mask_b)            (the photons that part pours)
   |DN| <= 768 * 15 = 11,520 and HN <= 768 * 4 = 3,072: both fit 16 bits.
   THIS IS THE SIMULATION'S SHORTCUT: every band of every word is formed here so the run can check; the machine pours
   band b only into the words still in at band b, and only those photons are counted (topband.py).
   A dark line (x = 0) pours nothing. Lines are walked in the order LO gives. Positions run side by side, each writing
   only its own rows. */
#include <stdint.h>
#include <dispatch/dispatch.h>

#define NB 8
static const int PB[NB] = {26, 22, 18, 14, 10, 6, 2, 0};
static const uint64_t MB[NB] = {15, 15, 15, 15, 15, 15, 15, 3};

static inline void pour(int64_t r, int16_t *dn, uint16_t *hn) {
    uint64_t a = r < 0 ? (uint64_t)(-r) : (uint64_t)r;
    int neg = r < 0;
    for (int b = 0; b < NB; b++) {
        uint64_t nib = (a >> PB[b]) & MB[b];
        dn[b] += neg ? -(int16_t)nib : (int16_t)nib;
        hn[b] += (uint16_t)__builtin_popcountll(nib);
    }
}

void band_walk(int64_t P, int64_t L, int64_t N, const int64_t *X, const int32_t *G, const uint16_t *NR,
               const int64_t *TOP, const int64_t *LO, int dir, int16_t *DN, uint16_t *HN, int64_t *OPEN, int64_t *BIG) {
    dispatch_apply((size_t)P, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t pp) {
        int64_t p = (int64_t)pp;
        int16_t *dn = DN + p * N * NB; uint16_t *hn = HN + p * N * NB;
        for (int64_t li = 0; li < L; li++) {
            int64_t i = LO[li]; const int32_t *g = G + i * N; const uint16_t *nr = NR + i * N;
            int64_t x = X[p * L + i];
            if (x == 0) continue;
            if (dir > 0) {
                int64_t r = 0;
                for (int64_t k = 0; k < N; k++) {
                    r += x * (int64_t)g[k];
                    if (r >= (1LL << 30) || r <= -(1LL << 30)) BIG[p] += 1;   /* never expected */
                    int64_t j = nr[k];
                    pour(r, dn + j * NB, hn + j * NB);
                }
            } else {
                int64_t r = x * TOP[i];
                for (int64_t k = N - 1; k >= 0; k--) {
                    if (r >= (1LL << 30) || r <= -(1LL << 30)) BIG[p] += 1;
                    int64_t j = nr[k];
                    pour(r, dn + j * NB, hn + j * NB);
                    r -= x * (int64_t)g[k];
                }
                if (r != 0) OPEN[p] += 1;                              /* the walk down did not close */
            }
        }
    });
}
