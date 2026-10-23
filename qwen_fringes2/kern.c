/* qwen_fringes2.py's kernels, whole numbers only (no float anywhere in this file).

   walk2, band2: qwen_fringes/walk2.c's two walks, copied unchanged (see that file for the full description).
     walk2: way 2 at one line-up (a plate, or a whole slab lined up across all its neurons): per line, the running
            product r_k = r_(k-1) + x g_k poured into the neuron at rank k; the 3x walk t_k beside it (side count, checked
            t = 3r at every rank); counts POUR1 (ones of every pour), POURS (signed digits of every pour, side), WRITE1
            (side), OPEN (a walk down not closed at 0), BAD3.
     band2: the word table's walk once, each exposure split into its 8 bands.

   pp8: THE ARRIVAL'S PLACE-PAIR PRODUCTS FROM SIGNED-DIGIT PLANES, AS WHOLE NUMBERS.
     A and B are planes of -1 / 0 / +1 held in int8; C[i][j] = sum over k < K of A[i][k] B[j][k], summed in int64.
     Row i of A starts at A + i lda, row j of B at B + j ldb (lda, ldb in elements), so a slice of a wider plane can be
     read in place. Every term is -1, 0 or +1; every sum is a whole number of at most K in size.

   mm16: THE CHECK PATH'S EXACT PRODUCT, AS WHOLE NUMBERS (never used by the machine).
     OUT[p][j] = sum over lines i of X[p][i] W[i][j]: X int64 (the light, |x| <= 2^15), W int16 (the plate as stored),
     summed in int64 (every product below 2^30, every sum below 2^41 for 2,048 lines, 2^42 for 3,072).
     Tiles of positions x columns run side by side, each writing only its own part of OUT. */
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

/* the arrival: int8 planes, int64 sums */
#define TI 8
#define TJ 64
void pp8(int64_t M, int64_t N, int64_t K, const int8_t *A, int64_t lda, const int8_t *B, int64_t ldb, int64_t *C) {
    int64_t ni = (M + TI - 1) / TI, nj = (N + TJ - 1) / TJ;
    dispatch_apply((size_t)(ni * nj), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t t) {
        int64_t i0 = (int64_t)(t / nj) * TI, j0 = (int64_t)(t % nj) * TJ;
        int64_t i1 = i0 + TI < M ? i0 + TI : M, j1 = j0 + TJ < N ? j0 + TJ : N;
        for (int64_t i = i0; i < i1; i++) {
            const int8_t *a = A + i * lda;
            for (int64_t j = j0; j < j1; j++) {
                const int8_t *b = B + j * ldb; int64_t s = 0;
                for (int64_t k = 0; k < K; k++) s += (int64_t)a[k] * (int64_t)b[k];
                C[i * N + j] = s;
            }
        }
    });
}

/* the check path: int64 light x int16 plate, int64 sums */
#define MP 16
#define MC 1024
void mm16(int64_t P, int64_t L, int64_t N, const int64_t *X, const int16_t *W, int64_t *OUT) {
    int64_t np_ = (P + MP - 1) / MP, nc = (N + MC - 1) / MC;
    dispatch_apply((size_t)(np_ * nc), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t t) {
        int64_t p0 = (int64_t)(t / nc) * MP, c0 = (int64_t)(t % nc) * MC;
        int64_t p1 = p0 + MP < P ? p0 + MP : P, c1 = c0 + MC < N ? c0 + MC : N, w = c1 - c0;
        for (int64_t p = p0; p < p1; p++) { int64_t *o = OUT + p * N + c0; for (int64_t j = 0; j < w; j++) o[j] = 0; }
        for (int64_t i = 0; i < L; i++) {
            const int16_t *row = W + i * N + c0;
            for (int64_t p = p0; p < p1; p++) {
                int64_t x = X[p * L + i];
                if (x == 0) continue;
                int64_t *o = OUT + p * N + c0;
                for (int64_t j = 0; j < w; j++) o[j] += x * (int64_t)row[j];
            }
        }
    });
}
