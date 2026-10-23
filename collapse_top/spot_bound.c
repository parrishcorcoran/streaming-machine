/* THE SPOT BOUND, word by word (called from collapse_top/spot_top.py; whole numbers only).

For one position, one group count G and one schedule: every word still in, after each stage, gets the exact bound on
its unread part from the stored counts, and is set aside when five words are surely above it.

  light counts  LA[a][g]  lines of group g where the light is lit at place a (signed digits: its ups; LB: its downs)
  word counts   WA[w][b][g]  lines of group g where word w is lit at place b (signed digits: its ups; WB: its downs)
  tables        TA_k[b][g][m] = sum over unread a (pair (a,b) not yet read at stage k) of min(LA[a][g], m) x 2^(a+b)
                (TB_k from LB). m runs to J = the light's largest count in the group; a larger stored count reads entry J.
  plain photons (one side):  |rest| <= R = sum over (g, b) TA_k[b][g][WA]
  signed digits (two sides): rest <= Rh = sum TA[WA] + TB[WB]   (up pairs: up meets up, down meets down)
                             rest >= -Rl, Rl = sum TA[WB] + TB[WA]   (down pairs: up meets down)
  lo = power(counted - Rl, F) + bias <= score <= hi = power(counted + Rh, F) + bias; set aside when hi < 5th best lo.

Work counted per stage, in photons (the 1s of |value| that an add carries):
  0 stored counts read   1s of every stored count read (each touched (g, b) of every word still in)
  1 table values added   1s of every table value added into a bound
  2 tables built         1s of min(LA[a][g], m) for every entry m = 1..J and unread a of every touched (g, b)
  3 bound adds           1s of Rh and Rl (pile + Rh, pile - Rl)
  4 divides              (1s of (pile - Rl) + 1s of (pile + Rh)) x 1s of the gain
  5 standing amounts     2 x 1s of the word's standing amount
  6 compares             a count, not photons
*/
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <dispatch/dispatch.h>

#define NPARTS 7
static inline int pc(int64_t v) { return __builtin_popcountll((uint64_t)(v < 0 ? -v : v)); }
static inline int64_t power_(int64_t X, int64_t F, const int64_t *EXP2) {   /* base2.power: X x 2^(F/65536) */
    int64_t I = F >> 16, f = F & 0xFFFF, v = X * EXP2[f], s = 16 - I;
    if (s > 0) return (v + ((int64_t)1 << (s - 1))) >> s;
    if (s < 0) return v << (-s);
    return v;
}

typedef struct {
    int P, NW, PL, G, K, two;
    const uint8_t *unread;                  /* K x PL(a) x PL(b) */
    const int64_t *CNT;                     /* P x K x NW: counted pile after stage k */
    const int32_t *MEET;                    /* P x (K+1) x NW: photon meetings of read stage k */
    const int64_t *S, *Fv, *GE, *BIAS, *EXP2;
    const int32_t *BONES;
    const uint16_t *WA, *WB;                /* NW x PL(b) x G */
    const int32_t *LA, *LB;                 /* P x PL(a) x G */
    int8_t *EXIT; int64_t *ENTER, *SURV, *MEETS, *PARTS, *BAD;
    int nsamp; const int32_t *SAMP; int64_t *SAMPOUT;   /* P x nsamp words; out P x nsamp x K x 4 (Rh, Rl, adds, reads) */
} Job;

static void build(const Job *J, int p, int k, const int32_t *L, int *Jg, int64_t *T, uint8_t *O, int Jmax, int64_t *built,
                  const uint8_t *touched) {
    int PL = J->PL, G = J->G;
    for (int g = 0; g < G; g++) { int m = 0; for (int a = 0; a < PL; a++) { int v = L[((size_t)p * PL + a) * G + g]; if (v > m) m = v; } Jg[g] = m; }
    for (int b = 0; b < PL; b++) for (int g = 0; g < G; g++) {
        int64_t *t = T + ((size_t)b * G + g) * (Jmax + 1); uint8_t *o = O + ((size_t)b * G + g) * (Jmax + 1);
        t[0] = 0; o[0] = 0;
        for (int m = 1; m <= Jg[g]; m++) {
            int64_t s = 0;
            for (int a = 0; a < PL; a++) if (J->unread[((size_t)k * PL + a) * PL + b]) {
                int v = L[((size_t)p * PL + a) * G + g]; int mn = v < m ? v : m;
                s += (int64_t)mn << (a + b);
                if (touched[b]) *built += pc(mn);
            }
            t[m] = s; o[m] = (uint8_t)pc(s);
        }
    }
}

static inline void bound_word(const Job *J, int w, const uint8_t *touched, const int *JA, const int *JB, int Jmax,
                              const int64_t *TA, const uint8_t *OA, const int64_t *TB, const uint8_t *OB,
                              int64_t *Rh, int64_t *Rl, int64_t *adds, int64_t *reads) {
    int PL = J->PL, G = J->G; int64_t rh = 0, rl = 0, ad = 0, rd = 0;
    const uint16_t *wa = J->WA + (size_t)w * PL * G, *wb = J->two ? J->WB + (size_t)w * PL * G : 0;
    for (int b = 0; b < PL; b++) {
        if (!touched[b]) continue;
        const int64_t *ta = TA + (size_t)b * G * (Jmax + 1); const uint8_t *oa = OA + (size_t)b * G * (Jmax + 1);
        if (J->two) {
            const int64_t *tb = TB + (size_t)b * G * (Jmax + 1); const uint8_t *ob = OB + (size_t)b * G * (Jmax + 1);
            for (int g = 0; g < G; g++) {
                int u = wa[b * G + g], d = wb[b * G + g], o = g * (Jmax + 1);
                int uA = u < JA[g] ? u : JA[g], dA = d < JA[g] ? d : JA[g], uB = u < JB[g] ? u : JB[g], dB = d < JB[g] ? d : JB[g];
                rh += ta[o + uA] + tb[o + dB];  rl += ta[o + dA] + tb[o + uB];
                ad += oa[o + uA] + ob[o + dB] + oa[o + dA] + ob[o + uB];
                rd += __builtin_popcount(u) + __builtin_popcount(d);
            }
        } else {
            for (int g = 0; g < G; g++) {
                int u = wa[b * G + g], o = g * (Jmax + 1); int uA = u < JA[g] ? u : JA[g];
                rh += ta[o + uA]; ad += oa[o + uA]; rd += __builtin_popcount(u);
            }
        }
    }
    if (!J->two) rl = rh;
    *Rh = rh; *Rl = rl; *adds = ad; *reads = rd;
}

static void job(const Job *J, int p) {
    int NW = J->NW, PL = J->PL, G = J->G, K = J->K;
    int Jmax = 0;
    for (size_t i = 0; i < (size_t)PL * G; i++) {
        int v = J->LA[(size_t)p * PL * G + i]; if (v > Jmax) Jmax = v;
        if (J->two) { v = J->LB[(size_t)p * PL * G + i]; if (v > Jmax) Jmax = v; }
    }
    size_t tsz = (size_t)PL * G * (Jmax + 1);
    int64_t *TA = calloc(tsz, 8), *TB = calloc(tsz, 8); uint8_t *OA = calloc(tsz, 1), *OB = calloc(tsz, 1);
    int *JA = calloc(G, sizeof(int)), *JB = calloc(G, sizeof(int));
    int32_t *alive = malloc((size_t)NW * 4); int64_t *lo = malloc((size_t)NW * 8), *hi = malloc((size_t)NW * 8);
    for (int w = 0; w < NW; w++) { alive[w] = w; J->EXIT[(size_t)p * NW + w] = (int8_t)K; }
    int n = NW; int64_t F = J->Fv[p], ge = J->GE[p], bad = 0;
    const int64_t *S = J->S + (size_t)p * NW;
    uint8_t touched[32];
    for (int k = 0; k < K; k++) {
        for (int b = 0; b < PL; b++) { touched[b] = 0; for (int a = 0; a < PL; a++) touched[b] |= J->unread[((size_t)k * PL + a) * PL + b]; }
        int64_t *part = J->PARTS + ((size_t)p * K + k) * NPARTS; int64_t built = 0;
        build(J, p, k, J->LA, JA, TA, OA, Jmax, &built, touched);
        if (J->two) build(J, p, k, J->LB, JB, TB, OB, Jmax, &built, touched);
        part[2] += built;
        const int32_t *meet = J->MEET + ((size_t)p * (K + 1) + k) * NW; const int64_t *cnt = J->CNT + ((size_t)p * K + k) * NW;
        int64_t ms = 0, rd_s = 0, ad_s = 0, ba_s = 0, dv_s = 0, st_s = 0;
        int64_t top[5]; int nt = 0;                 /* the five best lo so far, top[0] the smallest of them */
        for (int i = 0; i < n; i++) {
            int w = alive[i]; ms += meet[w];
            int64_t Rh, Rl, ad, rd; bound_word(J, w, touched, JA, JB, Jmax, TA, OA, TB, OB, &Rh, &Rl, &ad, &rd);
            int64_t c = cnt[w], lp = c - Rl, hp = c + Rh;
            int64_t l = power_(lp, F, J->EXP2) + J->BIAS[w], h = power_(hp, F, J->EXP2) + J->BIAS[w];
            if (l > S[w] || S[w] > h) bad++;
            rd_s += rd; ad_s += ad; ba_s += pc(Rh) + pc(Rl); dv_s += (int64_t)(pc(lp) + pc(hp)) * ge; st_s += 2 * J->BONES[w];
            lo[i] = l; hi[i] = h;
            if (nt < 5) { int j = nt++; while (j > 0 && top[j - 1] > l) { top[j] = top[j - 1]; j--; } top[j] = l; }
            else if (l > top[0]) { int j = 0; while (j < 4 && top[j + 1] < l) { top[j] = top[j + 1]; j++; } top[j] = l; }
        }
        J->MEETS[(size_t)p * (K + 1) + k] += ms;
        part[0] += rd_s; part[1] += ad_s; part[3] += ba_s; part[4] += dv_s; part[5] += st_s; part[6] += n;
        J->ENTER[(size_t)p * K + k] = n;
        int64_t L5 = top[0]; int m = 0;             /* n >= 5 always: the true top 5 are never set aside */
        for (int i = 0; i < n; i++) { if (hi[i] >= L5) alive[m++] = alive[i]; else J->EXIT[(size_t)p * NW + alive[i]] = (int8_t)k; }
        n = m; J->SURV[(size_t)p * K + k] = n;
        if (J->nsamp) for (int s = 0; s < J->nsamp; s++) {        /* the pieces, for the slow check in python */
            int w = J->SAMP[(size_t)p * J->nsamp + s]; int64_t *o = J->SAMPOUT + (((size_t)p * J->nsamp + s) * K + k) * 4;
            bound_word(J, w, touched, JA, JB, Jmax, TA, OA, TB, OB, &o[0], &o[1], &o[2], &o[3]);
        }
    }
    const int32_t *meet = J->MEET + ((size_t)p * (K + 1) + K) * NW; int64_t ms = 0;
    for (int i = 0; i < n; i++) ms += meet[alive[i]];
    J->MEETS[(size_t)p * (K + 1) + K] += ms;
    J->BAD[p] += bad;
    free(TA); free(TB); free(OA); free(OB); free(JA); free(JB); free(alive); free(lo); free(hi);
}

void collapse(int P, int NW, int PL, int G, int K, int two, const uint8_t *unread, const int64_t *CNT, const int32_t *MEET,
              const int64_t *S, const int64_t *Fv, const int64_t *GE, const int64_t *BIAS, const int32_t *BONES,
              const int64_t *EXP2, const uint16_t *WA, const uint16_t *WB, const int32_t *LA, const int32_t *LB,
              int8_t *EXIT, int64_t *ENTER, int64_t *SURV, int64_t *MEETS, int64_t *PARTS, int64_t *BAD,
              int nsamp, const int32_t *SAMP, int64_t *SAMPOUT, int threads) {
    Job J = {P, NW, PL, G, K, two, unread, CNT, MEET, S, Fv, GE, BIAS, EXP2, BONES, WA, WB, LA, LB,
             EXIT, ENTER, SURV, MEETS, PARTS, BAD, nsamp, SAMP, SAMPOUT};
    Job *Jp = &J;
    if (threads <= 1) { for (int p = 0; p < P; p++) job(Jp, p); return; }
    dispatch_apply((size_t)P, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t p) { job(Jp, (int)p); });
}
