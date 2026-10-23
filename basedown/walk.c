/* basedown/walk.c -- THE LINE-UP WALK at one plate (basedown/lineup.py lays the entries; whole numbers only).
 * For each line j (in the order lorder gives), the entries off[j]..off[j+1]-1 are the line's lit tokens in their
 * lined-up order: tok (the token's row), gap (its value minus the previous value on its side; the first on a side
 * against 0), start (1: the running product starts at 0 here). For every neuron n of the plate:
 *     running[n] += w[j,n] x gap         (the gap's signed digits meet the weight's signed digits)
 *     pile[tok, n] += running[n]         (the product lands in its own token's pile)
 * Counted (photons): out[0] the walk, naf(gap) x naf(w[j,n]) summed over the neurons; out[1] the landings, the 1s of
 * each product landed; out[2] the same products' signed digits (count only); out[3] CHECK: products that differ
 * from w[j,n] x x[tok,j] (must be 0); out[4] entries; out[5] entries with a gap other than 0. */
#include <stdint.h>
#include <string.h>
static inline int64_t pc(uint64_t a) { return (int64_t)__builtin_popcountll(a); }
static inline uint64_t ab(int64_t v) { return (uint64_t)(v < 0 ? -v : v); }

void walk(int64_t P, int64_t N, int64_t M, const int32_t *x, const int16_t *w, const int64_t *lorder,
          const int64_t *off, const int32_t *tok, const int32_t *gap, const uint8_t *start,
          int64_t *pile, int64_t *run, int64_t *out)
{
    int64_t walkp = 0, land1 = 0, landn = 0, bad = 0, ent = 0, nzg = 0;
    for (int64_t q = 0; q < N; q++) {
        int64_t j = lorder[q]; const int16_t *wr = w + j * M;
        int64_t nr = 0;
        for (int64_t n = 0; n < M; n++) { uint64_t a = ab(wr[n]); nr += pc(a ^ (3 * a)); }
        for (int64_t e = off[j]; e < off[j + 1]; e++) {
            int64_t t = tok[e], d = gap[e];
            if (start[e]) memset(run, 0, (size_t)M * sizeof(int64_t));
            if (d) {
                uint64_t ad = ab(d); walkp += pc(ad ^ (3 * ad)) * nr; nzg++;
                for (int64_t n = 0; n < M; n++) run[n] += (int64_t)wr[n] * d;
            }
            int64_t *pr = pile + t * M; int64_t xv = x[t * N + j];
            int64_t l1 = 0, ln = 0, b = 0;
            for (int64_t n = 0; n < M; n++) {
                int64_t v = run[n]; pr[n] += v; uint64_t a = ab(v);
                l1 += pc(a); ln += pc(a ^ (3 * a)); b += (v != (int64_t)wr[n] * xv);
            }
            land1 += l1; landn += ln; bad += b; ent++;
        }
    }
    out[0] = walkp; out[1] = land1; out[2] = landn; out[3] = bad; out[4] = ent; out[5] = nzg;
}

/* CHECK PATH: the direct product, every line met one by one, int64 */
void direct(int64_t P, int64_t N, int64_t M, const int32_t *x, const int16_t *w, int64_t *out)
{
    for (int64_t t = 0; t < P; t++) {
        int64_t *o = out + t * M; memset(o, 0, (size_t)M * sizeof(int64_t));
        for (int64_t j = 0; j < N; j++) {
            int64_t xv = x[t * N + j]; if (!xv) continue; const int16_t *wr = w + j * M;
            for (int64_t n = 0; n < M; n++) o[n] += xv * (int64_t)wr[n];
        }
    }
}
