/* confirm_groups/gk.c -- the checker's own kernels for the independent rebuild of base2_groups.py (GPT-2).
   Written from the stated rules only; no code from base2_groups.py, groups/ or groups2/.
   Whole numbers only.

   bank_uh: one plate, every position. For each group of g light lines every pattern's ready pile is made once a
     position, V[p] = sum over the group's lines t of (+x if bit t of p is up, -x if a hole), by the lowest-bit rule
     V[p] = V[p less its lowest line] + 2 x(that line), V[0] = -(the group's light summed). Each neuron at each up/hole
     place adds one ready pile shifted to its place into its doubled pile. Counted per position (photons):
       reads  = sum over every read of the ready pile's signed digits (by the plate's multiplicity of each pattern)
       pwrite = every pattern pair {p, complement} the plate reads at all, written once: its 1s
   top_uh: the top table. For every position, every up/hole place u and every word w:
       contrib[u][w] = sum over the groups of the ready pile the word reads there (not shifted)
       cost[u][w]    = sum over the groups of that ready pile's signed digits
     and pwrite as above. The set-aside is made by the caller from these.
   Signed digits of a whole number a: popcount(|a| xor 3|a|) (checked by the caller against the digit-by-digit rule). */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

static inline uint64_t uabs(int64_t v) { return v < 0 ? (uint64_t)(-v) : (uint64_t)v; }
static inline int64_t sdig(int64_t v) { uint64_t a = uabs(v); return __builtin_popcountll(a ^ (3 * a)); }
static inline int64_t ones(int64_t v) { return __builtin_popcountll(uabs(v)); }

int64_t sd_count(int64_t v) { return sdig(v); }

static void make_table(int64_t *V, int g, const int32_t *lines, const int64_t *x) {
    int64_t s = 0;
    for (int t = 0; t < g; t++) s += x[lines[t]];
    V[0] = -s;
    int n = 1 << g;
    for (int p = 1; p < n; p++) {
        int t = __builtin_ctz(p);
        V[p] = V[p & (p - 1)] + 2 * x[lines[t]];
    }
}

typedef struct {
    int P, N, M, g, G, nU, B;
    const int32_t *gl, *ub, *gorder;
    const uint16_t *pat;
    const uint32_t *cnt;
    const int64_t *x;
    int64_t *pile2, *reads, *pwrite;
    int p0, p1;
} BankJob;

static void *bank_worker(void *arg) {
    BankJob *J = (BankJob *)arg;
    int g = J->g, n = 1 << g, M = J->M, nU = J->nU, B = J->B;
    int64_t *V = (int64_t *)malloc(sizeof(int64_t) * (size_t)n * B);
    int64_t mult[32];
    for (int u = 0; u < nU; u++) mult[u] = (int64_t)1 << J->ub[u];
    for (int q0 = J->p0; q0 < J->p1; q0 += B) {
        int nb = J->p1 - q0 < B ? J->p1 - q0 : B;
        for (int gi = 0; gi < J->G; gi++) {
            int k = J->gorder[gi];
            const int32_t *lines = J->gl + (size_t)k * g;
            const uint32_t *ck = J->cnt + (size_t)k * n;
            for (int b = 0; b < nb; b++) {
                int64_t *Vb = V + (size_t)b * n;
                make_table(Vb, g, lines, J->x + (size_t)(q0 + b) * J->N);
                int64_t r = 0, w = 0;
                for (int p = 0; p < n; p++) if (ck[p]) r += (int64_t)ck[p] * sdig(Vb[p]);
                int half = n >> 1;
                for (int p = half; p < n; p++) if (ck[p] || ck[p ^ (n - 1)]) w += ones(Vb[p]);
                J->reads[q0 + b] += r; J->pwrite[q0 + b] += w;
            }
            const uint16_t *pk = J->pat + (size_t)k * M * nU;
            for (int j = 0; j < M; j++) {
                const uint16_t *pj = pk + (size_t)j * nU;
                for (int b = 0; b < nb; b++) {
                    const int64_t *Vb = V + (size_t)b * n;
                    int64_t acc = 0;
                    for (int u = 0; u < nU; u++) acc += Vb[pj[u]] * mult[u];
                    J->pile2[(size_t)(q0 + b) * M + j] += acc;
                }
            }
        }
    }
    free(V);
    return NULL;
}

/* pile2 (P x M), reads (P), pwrite (P) are ADDED INTO (the caller zeroes them). */
void bank_uh(int P, int N, int M, int g, int G, const int32_t *gl, const int32_t *gorder, int nU, const int32_t *ub,
             const uint16_t *pat, const uint32_t *cnt, const int64_t *x, int64_t *pile2, int64_t *reads,
             int64_t *pwrite, int nthreads) {
    if (nthreads < 1) nthreads = 1;
    if (nthreads > P) nthreads = P > 0 ? P : 1;
    pthread_t th[64]; BankJob jobs[64];
    int per = (P + nthreads - 1) / nthreads;
    for (int i = 0; i < nthreads; i++) {
        BankJob *J = &jobs[i];
        J->P = P; J->N = N; J->M = M; J->g = g; J->G = G; J->nU = nU; J->B = 4;
        J->gl = gl; J->ub = ub; J->gorder = gorder; J->pat = pat; J->cnt = cnt; J->x = x;
        J->pile2 = pile2; J->reads = reads; J->pwrite = pwrite;
        J->p0 = i * per; J->p1 = (i + 1) * per < P ? (i + 1) * per : P;
        if (J->p0 >= J->p1) { J->p0 = J->p1 = 0; }
        pthread_create(&th[i], NULL, bank_worker, J);
    }
    for (int i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
}

typedef struct {
    int P, N, NW, g, G, nU;
    const int32_t *gl;
    const uint16_t *pat;   /* G x NW x nU */
    const uint32_t *cnt;
    const int64_t *x;
    int64_t *contrib;      /* P x nU x NW */
    int32_t *cost;         /* P x nU x NW */
    int64_t *pwrite;
    int p0, p1;
} TopJob;

static void *top_worker(void *arg) {
    TopJob *J = (TopJob *)arg;
    int g = J->g, n = 1 << g, NW = J->NW, nU = J->nU;
    int64_t *V = (int64_t *)malloc(sizeof(int64_t) * (size_t)n);
    uint8_t *S = (uint8_t *)malloc((size_t)n);
    for (int q = J->p0; q < J->p1; q++) {
        int64_t *C = J->contrib + (size_t)q * nU * NW;
        int32_t *K = J->cost + (size_t)q * nU * NW;
        memset(C, 0, sizeof(int64_t) * (size_t)nU * NW);
        memset(K, 0, sizeof(int32_t) * (size_t)nU * NW);
        int64_t w = 0;
        for (int k = 0; k < J->G; k++) {
            make_table(V, g, J->gl + (size_t)k * g, J->x + (size_t)q * J->N);
            for (int p = 0; p < n; p++) S[p] = (uint8_t)sdig(V[p]);
            const uint32_t *ck = J->cnt + (size_t)k * n;
            for (int p = n >> 1; p < n; p++) if (ck[p] || ck[p ^ (n - 1)]) w += ones(V[p]);
            const uint16_t *pk = J->pat + (size_t)k * NW * nU;
            for (int wd = 0; wd < NW; wd++) {
                const uint16_t *pw = pk + (size_t)wd * nU;
                for (int u = 0; u < nU; u++) {
                    uint16_t p = pw[u];
                    C[(size_t)u * NW + wd] += V[p];
                    K[(size_t)u * NW + wd] += S[p];
                }
            }
        }
        J->pwrite[q] = w;
    }
    free(V); free(S);
    return NULL;
}

void top_uh(int P, int N, int NW, int g, int G, const int32_t *gl, int nU, const uint16_t *pat, const uint32_t *cnt,
            const int64_t *x, int64_t *contrib, int32_t *cost, int64_t *pwrite, int nthreads) {
    if (nthreads < 1) nthreads = 1;
    if (nthreads > P) nthreads = P > 0 ? P : 1;
    pthread_t th[64]; TopJob jobs[64];
    int per = (P + nthreads - 1) / nthreads;
    for (int i = 0; i < nthreads; i++) {
        TopJob *J = &jobs[i];
        J->P = P; J->N = N; J->NW = NW; J->g = g; J->G = G; J->nU = nU;
        J->gl = gl; J->pat = pat; J->cnt = cnt; J->x = x; J->contrib = contrib; J->cost = cost; J->pwrite = pwrite;
        J->p0 = i * per; J->p1 = (i + 1) * per < P ? (i + 1) * per : P;
        if (J->p0 >= J->p1) { J->p0 = J->p1 = 0; }
        pthread_create(&th[i], NULL, top_worker, J);
    }
    for (int i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
}

/* top_sd: the top table's signed-digit places. For each signed place s, the lit digits by word (CSR: wstart[s*(NW+1)
   + w] .. wstart[s*(NW+1) + w + 1] index lines[] and signs[], offsets into one shared list). For every position:
     contrib[s][w] = sum over the word's lit digits at that place of sign x x(line)   (not shifted, not doubled)
     cost[s][w]    = sum over those digits of the light value's signed digits                                        */
typedef struct {
    int P, N, NW, nS;
    const int64_t *wstart; const int32_t *lines; const int8_t *signs;
    const int64_t *x; int64_t *contrib; int32_t *cost; int p0, p1;
} SdJob;
static void *sd_worker(void *arg) {
    SdJob *J = (SdJob *)arg;
    int NW = J->NW, nS = J->nS;
    int32_t *nx = (int32_t *)malloc(sizeof(int32_t) * J->N);
    for (int q = J->p0; q < J->p1; q++) {
        const int64_t *x = J->x + (size_t)q * J->N;
        for (int t = 0; t < J->N; t++) nx[t] = (int32_t)sdig(x[t]);
        for (int s = 0; s < nS; s++) {
            const int64_t *ws = J->wstart + (size_t)s * (NW + 1);
            int64_t *C = J->contrib + ((size_t)q * nS + s) * NW;
            int32_t *K = J->cost + ((size_t)q * nS + s) * NW;
            for (int w = 0; w < NW; w++) {
                int64_t c = 0; int32_t k = 0;
                for (int64_t i = ws[w]; i < ws[w + 1]; i++) { int t = J->lines[i]; c += J->signs[i] * x[t]; k += nx[t]; }
                C[w] = c; K[w] = k;
            }
        }
    }
    free(nx);
    return NULL;
}
void top_sd(int P, int N, int NW, int nS, const int64_t *wstart, const int32_t *lines, const int8_t *signs,
            const int64_t *x, int64_t *contrib, int32_t *cost, int nthreads) {
    if (nthreads < 1) nthreads = 1;
    if (nthreads > P) nthreads = P > 0 ? P : 1;
    pthread_t th[64]; SdJob jobs[64];
    int per = (P + nthreads - 1) / nthreads;
    for (int i = 0; i < nthreads; i++) {
        SdJob *J = &jobs[i];
        J->P = P; J->N = N; J->NW = NW; J->nS = nS; J->wstart = wstart; J->lines = lines; J->signs = signs; J->x = x;
        J->contrib = contrib; J->cost = cost;
        J->p0 = i * per; J->p1 = (i + 1) * per < P ? (i + 1) * per : P;
        if (J->p0 >= J->p1) { J->p0 = J->p1 = 0; }
        pthread_create(&th[i], NULL, sd_worker, J);
    }
    for (int i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
}
