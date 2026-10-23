/* TABLEDIFFUSE family: developing the entry table E with each value's rounding error carried on (error diffusion,
   halftoning). Whole numbers only: adds, subtracts, shifts, ANDs, compares.
   Every value v (E at 2^16) is developed on its line's grid, step 2^e[i] (round one's lens-matched exposure):
       Q(t) = sign(t) x ((|t| + 2^(e-1)) AND NOT (2^e - 1))        -- nearest, as round one
   The error is carried in one of three units (what is kept equal is the running deficit in that unit):
       unit 0: E's own units (the entry's frame: every line read with gain 1)
       unit 1: fractions of each line's own step, at 2^16 (the error measured in steps)
       unit 2: the top's units: the error times 2^p[i], p[i] = the place of line i's lens gain (what the top sees)
   Carry from unit u back to a line's E units is a shift with rounding to nearest.
   mode 0: no carry (nearest; must equal round one)
   mode 1: first order along each word's lines, in the order lord[] (the chain restarts at every word)
   mode 2: second order along each word's lines: the next value gets 2u1 - u2 (noise shaped by (1 - z^-1)^2)
   mode 3: first order along each line, word after word in the order word[] (the chain runs through all words)
   mode 4: two-dimensional, Floyd-Steinberg weights: 7/16 to the next line of the same word, 3/16, 5/16, 1/16 to
           the previous, same and next line of the next word (words in the order word[], lines in natural order)
   mode 5: first order along each word's lines with a leak: carry u - (u >> k) (k = leak)
   cc -O2 -shared -o tablediffuse_ed.dylib tablediffuse_ed.c                                                    */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define L 768

static inline int64_t qnear(int64_t t, int e) {
    int64_t a = t < 0 ? -t : t;
    int64_t q = (int64_t)1 << e;
    int64_t a2 = (a + (q >> 1)) & ~(q - 1);
    return t < 0 ? -a2 : a2;
}
static inline int64_t sh(int u, int i, const int32_t *e, const int32_t *p) {  /* places of unit u on line i */
    return u == 0 ? 0 : (u == 1 ? 16 - e[i] : p[i]);
}
static inline int64_t from_unit(int64_t x, int64_t k) {           /* x x 2^-k, nearest (k >= 0) */
    if (k <= 0) return x;
    return (x + ((int64_t)1 << (k - 1))) >> k;
}

/* V: n x 768 int32 (E at 2^16 fits in 19 bits), out: n x 768 int32. Returns 0. */
int64_t ed_develop(const int32_t *V, int64_t n, const int32_t *e, const int32_t *p, int mode, int unit, int leak,
                   const int32_t *lord, const int64_t *word, int32_t *out) {
    if (mode == 0) {
        for (int64_t r = 0; r < n * L; r++) out[r] = (int32_t)qnear(V[r], e[r % L]);
        return 0;
    }
    if (mode == 1 || mode == 2 || mode == 5) {
        for (int64_t w = 0; w < n; w++) {
            const int32_t *v = V + w * L; int32_t *o = out + w * L;
            int64_t u1 = 0, u2 = 0;                        /* last two quantizer errors, in the carry unit */
            for (int j = 0; j < L; j++) {
                int i = lord[j]; int64_t k = sh(unit, i, e, p);
                int64_t f = (mode == 2) ? 2 * u1 - u2 : u1;
                if (mode == 5) f = f - (f >> leak);
                int64_t d = qnear(v[i] + from_unit(f, k), e[i]);
                int64_t u = ((v[i] - d) << k) + f;         /* what is still owed after this value, exactly */
                u2 = u1; u1 = u; o[i] = (int32_t)d;
            }
        }
        return 0;
    }
    if (mode == 3) {
        int64_t *acc = calloc(L, sizeof(int64_t));
        for (int64_t k = 0; k < n; k++) {
            int64_t w = word[k]; const int32_t *v = V + w * L; int32_t *o = out + w * L;
            for (int i = 0; i < L; i++) {
                int64_t s = sh(unit, i, e, p);
                int64_t d = qnear(v[i] + from_unit(acc[i], s), e[i]);
                acc[i] = ((v[i] - d) << s) + acc[i]; o[i] = (int32_t)d;
            }
        }
        free(acc); return 0;
    }
    if (mode == 4) {
        int64_t *cur = calloc(L + 2, sizeof(int64_t)), *nxt = calloc(L + 2, sizeof(int64_t));
        for (int64_t k = 0; k < n; k++) {
            int64_t w = word[k]; const int32_t *v = V + w * L; int32_t *o = out + w * L;
            for (int i = 0; i < L; i++) {
                int64_t s = sh(unit, i, e, p);
                int64_t f = cur[i + 1];
                int64_t d = qnear(v[i] + from_unit(f, s), e[i]);
                int64_t u = ((v[i] - d) << s) + f;
                int64_t a = (7 * u) >> 4, b = (3 * u) >> 4, c = (5 * u) >> 4, dd = u - a - b - c;   /* sums to u exactly */
                if (i + 1 < L) cur[i + 2] += a; else c += a;  /* the last line passes its share down */
                nxt[i] += (i > 0) ? b : 0; if (i == 0) c += b;
                nxt[i + 1] += c; if (i + 1 < L) nxt[i + 2] += dd; else nxt[i + 1] += dd;
                o[i] = (int32_t)d;
            }
            int64_t *t = cur; cur = nxt; nxt = t; memset(nxt, 0, (L + 2) * sizeof(int64_t));
        }
        free(cur); free(nxt); return 0;
    }
    if (mode == 6) {                                       /* the owed leftover spread evenly over the word's
                                                              remaining lines: the k-th from last line repays 1/k of
                                                              it (reciprocal table, whole numbers: R[k] = 2^32 / k) */
        int64_t R[L + 1]; R[0] = 0;
        for (int k = 1; k <= L; k++) R[k] = ((int64_t)1 << 32) / k;
        for (int64_t w = 0; w < n; w++) {
            const int32_t *v = V + w * L; int32_t *o = out + w * L;
            int64_t acc = 0;                               /* owed, in the carry unit */
            for (int j = 0; j < L; j++) {
                int i = lord[j]; int64_t k = sh(unit, i, e, p); int left = L - j;
                int64_t f = (left == 1) ? acc : ((acc * R[left]) >> 32);   /* this line's share */
                int64_t d = qnear(v[i] + from_unit(f, k), e[i]);
                acc = acc - f + (((v[i] - d) << k) + f);      /* = acc + (v - d) in the unit */
                o[i] = (int32_t)d;
            }
        }
        return 0;
    }
    return -1;
}
