/* TABLEDIFFUSE family: the parity screen -- whole-number lattices for the developed entry table.
   Every value keeps round one's grid (line i: multiples of 2^e[i]); its stored whole number is k = D >> e[i].
   The 768 lines are dealt into 96 groups of 8 at random (a fixed-seed shuffle, the same for every word), and the 8
   stored numbers of a group must obey a parity rule, so fewer combinations exist and fewer bits are needed:
     kind 1 (D8, the checkerboard): the sum of the group's 8 numbers is even. Every value is rounded to nearest; if
            the sum comes out odd, the ONE value whose leftover is closest to half a step goes to its other neighbour
            (the cheapest dot to move -- the press nudges one dot of eight instead of carrying grain along).
            Saves 1 bit per 8 values; adds 1 - 2|u_max| step^2 to one value in half the groups.
     kind 2 (E8, the [8,4,4] code): the 8 low places (k AND 1) must spell one of the 16 words of the extended Hamming
            code. For each code word the nearest numbers with those low places are found (each value's nearest even or
            nearest odd number); the code word with the least total leftover^2 is kept.
            Saves 4 bits per 8 values (the low place of each of 8 values, minus the 4-bit code word).
   Leftovers are measured in fractions of each line's own step at 2^16 (the lens-matched frame: for the lines the lens
   governs, a step is the same top light on every line). Whole numbers only: shifts, adds, ANDs, compares.
   cc -O2 -shared -o tablediffuse_lattice.dylib tablediffuse_lattice.c                                           */
#include <stdint.h>
#include <stdlib.h>

#define L 768
#define G 96

static inline int64_t qnear(int64_t t, int e) {
    int64_t a = t < 0 ? -t : t, q = (int64_t)1 << e, a2 = (a + (q >> 1)) & ~(q - 1);
    return t < 0 ? -a2 : a2;
}

static int CW[16];                                   /* the 16 words of the [8,4,4] code: affine maps on F_2^3 */
static void code_words(void) {
    for (int a = 0; a < 8; a++) for (int b = 0; b < 2; b++) {
        int w = 0;
        for (int x = 0; x < 8; x++) { int bit = (__builtin_popcount(a & x) + b) & 1; w |= bit << x; }
        CW[a * 2 + b] = w;
    }
}

/* V n x 768 int32; e[768]; grp[768] = the lines of group 0 (8), then group 1, ...; out n x 768 int32 (developed);
   cw n x 96 uint8 (E8: the code word index of each group). */
int64_t lat_develop(const int32_t *V, int64_t n, const int32_t *e, const int32_t *grp, int kind, int32_t *out, uint8_t *cw) {
    code_words();
    for (int64_t w = 0; w < n; w++) {
        const int32_t *v = V + w * L; int32_t *o = out + w * L;
        for (int g = 0; g < G; g++) {
            const int32_t *ln = grp + g * 8;
            if (kind == 1) {
                int64_t k[8], u[8]; int par = 0, jmax = 0; int64_t umax = -1;
                for (int j = 0; j < 8; j++) {
                    int i = ln[j]; int64_t d = qnear(v[i], e[i]); k[j] = d >> e[i];
                    u[j] = ((int64_t)v[i] - d) << (16 - e[i]);          /* leftover, steps at 2^16 */
                    par ^= (int)(k[j] & 1);
                    int64_t au = u[j] < 0 ? -u[j] : u[j];
                    if (au > umax) { umax = au; jmax = j; }
                }
                if (par) k[jmax] += (u[jmax] >= 0) ? 1 : -1;            /* the dot nearest halfway moves over */
                for (int j = 0; j < 8; j++) o[ln[j]] = (int32_t)(k[j] << e[ln[j]]);
            } else {
                int64_t ke[8], ko[8], ce[8], co[8];
                for (int j = 0; j < 8; j++) {
                    int i = ln[j]; int64_t x = (int64_t)v[i] << (16 - e[i]);   /* the value in steps at 2^16 */
                    ke[j] = ((x + 65536) >> 17) << 1;                       /* nearest even number of steps */
                    ko[j] = ((x >> 17) << 1) + 1;                           /* nearest odd number of steps */
                    int64_t de = x - (ke[j] << 16), dd = x - (ko[j] << 16);
                    ce[j] = de * de; co[j] = dd * dd;
                }
                int best = 0; int64_t bc = -1;
                for (int c = 0; c < 16; c++) {
                    int64_t s = 0;
                    for (int j = 0; j < 8; j++) s += ((CW[c] >> j) & 1) ? co[j] : ce[j];
                    if (bc < 0 || s < bc) { bc = s; best = c; }
                }
                for (int j = 0; j < 8; j++) {
                    int64_t k = ((CW[best] >> j) & 1) ? ko[j] : ke[j];
                    o[ln[j]] = (int32_t)(k << e[ln[j]]);
                }
                cw[w * G + g] = (uint8_t)best;
            }
        }
    }
    return 0;
}

int64_t lat_code_word(int c) { code_words(); return CW[c]; }
