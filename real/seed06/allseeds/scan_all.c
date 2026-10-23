// All-seeds sign scan for Qwen3-0.6B's blank.
// For every 32-bit seed (torch's CPU generator uses only seed & 0xffffffff): mt19937 from torch.manual_seed(seed),
// the signs of the first draws under five CPU draws, against the first 2,048 values of every candidate plate.
//   A bf16 (8-bit uniforms, angle bits 7,6)    B fp16 (11-bit, bits 10,9)    C fp32 (24-bit, bits 23,22)
//     all three: one word a value, blocks of 16, value j and j+8 share the angle of word j+8
//   D fp64 (two words a value, high word first, bits 20,19 of the high word), blocks of 16 as above
//   E one at a time (non-contiguous or < 16 values): 4 words per 2 values, angle from word 4i, bits 20,19
//   sign of cos = top XOR next bit; sign of sin = top bit.  A, B, C: 2,048 values; D, E: 1,248 values.
// Stage 1: first 512 signs, |z| >= 4 passes.  Stage 2: all signs, written out if |z| >= 6.5.
// usage: scan_all <start> <end> <threads> <outprefix>      (seeds start <= s < end)
//        scan_all dump <seed>                              (words and sign vectors, for checking)
#include <arm_neon.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NW 2496
#define NCONV 5
#define NDE 1248
#define MAXC 512
static const char CONVN[NCONV] = {'A', 'B', 'C', 'D', 'E'};
static int NC;
static uint64_t CS[MAXC][32], CM[MAXC][32], CMDE[MAXC][32];
static int N2[NCONV][MAXC];
static uint64_t g_start, g_end, CHUNK = 1u << 20;
static atomic_uint_fast64_t next_chunk;
static atomic_uint_fast64_t seeds_done;
static pthread_mutex_t out_mu = PTHREAD_MUTEX_INITIALIZER;
static FILE *hits;

static inline void mt_init(uint32_t *mt, uint32_t s) {
  mt[0] = s;
  for (int i = 1; i < 624; i++) mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + (uint32_t)i;
}
static inline uint32_t mix(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t y = (a & 0x80000000u) | (b & 0x7fffffffu);
  return c ^ (y >> 1) ^ ((0u - (y & 1u)) & 0x9908b0dfu);
}
static inline void mt_twist(uint32_t *mt) {
  int i;
  for (i = 0; i < 227; i++) mt[i] = mix(mt[i], mt[i + 1], mt[i + 397]);
  for (; i < 623; i++) mt[i] = mix(mt[i], mt[i + 1], mt[i - 227]);
  mt[623] = mix(mt[623], mt[0], mt[396]);
}
static inline uint32_t temper(uint32_t y) {
  y ^= y >> 11; y ^= (y << 7) & 0x9d2c5680u; y ^= (y << 15) & 0xefc60000u; y ^= y >> 18;
  return y;
}
static void words(uint32_t seed, uint32_t *w) {
  uint32_t mt[624];
  mt_init(mt, seed);
  for (int t = 0; t < 4; t++) {
    mt_twist(mt);
    for (int i = 0; i < 624; i++) w[t * 624 + i] = temper(mt[i]);
  }
}
// sign vectors: bit v = 1 when value v is drawn negative
static void signs(const uint32_t *w, uint64_t sv[NCONV][32]) {
  memset(sv, 0, sizeof(uint64_t) * NCONV * 32);
  static const int HI[3] = {7, 10, 23};
  for (int c = 0; c < 3; c++) {
    int hi = HI[c];
    for (int b = 0; b < 128; b++) {
      uint64_t chunk = 0;
      for (int j = 0; j < 8; j++) {
        uint32_t a = w[16 * b + 8 + j], t = (a >> hi) & 1u, u = (a >> (hi - 1)) & 1u;
        chunk |= (uint64_t)(t ^ u) << j;
        chunk |= (uint64_t)t << (8 + j);
      }
      sv[c][b >> 2] |= chunk << (16 * (b & 3));
    }
  }
  for (int b = 0; b < NDE / 16; b++) {  // D
    uint64_t chunk = 0;
    for (int j = 0; j < 8; j++) {
      uint32_t a = w[2 * (16 * b + 8 + j)], t = (a >> 20) & 1u, u = (a >> 19) & 1u;
      chunk |= (uint64_t)(t ^ u) << j;
      chunk |= (uint64_t)t << (8 + j);
    }
    sv[3][b >> 2] |= chunk << (16 * (b & 3));
  }
  for (int i = 0; i < NDE / 2; i++) {  // E
    uint32_t a = w[4 * i], t = (a >> 20) & 1u, u = (a >> 19) & 1u;
    int v = 2 * i;
    sv[4][v >> 6] |= (uint64_t)(t ^ u) << (v & 63);
    sv[4][(v + 1) >> 6] |= (uint64_t)t << ((v + 1) & 63);
  }
}
static inline int diff512(const uint64_t *s, const uint64_t *c) {
  uint8x16_t x0 = veorq_u8(vld1q_u8((const uint8_t *)s), vld1q_u8((const uint8_t *)c));
  uint8x16_t x1 = veorq_u8(vld1q_u8((const uint8_t *)(s + 2)), vld1q_u8((const uint8_t *)(c + 2)));
  uint8x16_t x2 = veorq_u8(vld1q_u8((const uint8_t *)(s + 4)), vld1q_u8((const uint8_t *)(c + 4)));
  uint8x16_t x3 = veorq_u8(vld1q_u8((const uint8_t *)(s + 6)), vld1q_u8((const uint8_t *)(c + 6)));
  uint8x16_t n = vaddq_u8(vaddq_u8(vcntq_u8(x0), vcntq_u8(x1)), vaddq_u8(vcntq_u8(x2), vcntq_u8(x3)));
  return (int)vaddlvq_u8(n);
}
static inline int diff_full(const uint64_t *s, const uint64_t *c, const uint64_t *m) {
  int d = 0;
  for (int k = 0; k < 32; k++) d += __builtin_popcountll((s[k] ^ c[k]) & m[k]);
  return d;
}

typedef struct { double best[NCONV][MAXC]; uint32_t best_seed[NCONV][MAXC]; uint64_t stage2; } tstate;

static void *worker(void *arg) {
  tstate *ts = (tstate *)arg;
  uint32_t w[NW];
  uint64_t sv[NCONV][32];
  char buf[1 << 16]; int blen = 0;
  for (;;) {
    uint64_t ch = atomic_fetch_add(&next_chunk, 1);
    uint64_t s0 = g_start + ch * CHUNK;
    if (s0 >= g_end) break;
    uint64_t s1 = s0 + CHUNK < g_end ? s0 + CHUNK : g_end;
    for (uint64_t s = s0; s < s1; s++) {
      words((uint32_t)s, w);
      signs(w, sv);
      for (int c = 0; c < NCONV; c++) {
        for (int k = 0; k < NC; k++) {
          int d1 = diff512(sv[c], CS[k]);
          if (d1 > 210 && d1 < 302) continue;  // |z| on the first 512 signs below 4
          ts->stage2++;
          int d2 = diff_full(sv[c], CS[k], c < 3 ? CM[k] : CMDE[k]);
          double z2 = (N2[c][k] - 2.0 * d2) / sqrt((double)N2[c][k]);
          if (fabs(z2) > fabs(ts->best[c][k])) { ts->best[c][k] = z2; ts->best_seed[c][k] = (uint32_t)s; }
          if (fabs(z2) >= 6.5) {
            blen += snprintf(buf + blen, sizeof(buf) - blen, "%u\t%c\t%d\t%+.3f\t%+.3f\n", (uint32_t)s, CONVN[c], k,
                             z2, (512 - 2.0 * d1) / sqrt(512.0));
            if (blen > (int)sizeof(buf) - 256) {
              pthread_mutex_lock(&out_mu); fwrite(buf, 1, blen, hits); fflush(hits); pthread_mutex_unlock(&out_mu);
              blen = 0;
            }
          }
        }
      }
    }
    if (blen) { pthread_mutex_lock(&out_mu); fwrite(buf, 1, blen, hits); fflush(hits); pthread_mutex_unlock(&out_mu); blen = 0; }
    atomic_fetch_add(&seeds_done, s1 - s0);
  }
  return NULL;
}

static void load(void) {
  FILE *f = fopen("cands.bin", "rb");
  if (!f) { perror("cands.bin"); exit(1); }
  uint64_t row[64];
  NC = 0;
  while (fread(row, 8, 64, f) == 64) {
    if (NC >= MAXC) { fprintf(stderr, "too many candidates\n"); exit(1); }
    memcpy(CS[NC], row, 256); memcpy(CM[NC], row + 32, 256);
    for (int q = 0; q < 32; q++) {
      int lo = q * 64;
      uint64_t keep = lo + 64 <= NDE ? ~0ull : (lo >= NDE ? 0ull : ((1ull << (NDE - lo)) - 1));
      CMDE[NC][q] = CM[NC][q] & keep;
    }
    for (int q = 0; q < 8; q++)
      if (CM[NC][q] != ~0ull) { fprintf(stderr, "candidate %d has a zero in its first 512 values\n", NC); exit(1); }
    int a = 0, b = 0;
    for (int q = 0; q < 32; q++) { a += __builtin_popcountll(CM[NC][q]); b += __builtin_popcountll(CMDE[NC][q]); }
    for (int c = 0; c < NCONV; c++) N2[c][NC] = c < 3 ? a : b;
    NC++;
  }
  fclose(f);
}

int main(int argc, char **argv) {
  if (argc == 3 && !strcmp(argv[1], "dump")) {
    uint32_t w[NW]; uint64_t sv[NCONV][32];
    words((uint32_t)strtoull(argv[2], 0, 10), w); signs(w, sv);
    for (int i = 0; i < NW; i++) printf("%u%c", w[i], i + 1 < NW ? ' ' : '\n');
    for (int c = 0; c < NCONV; c++) { for (int k = 0; k < 32; k++) printf("%016llx", (unsigned long long)sv[c][k]); printf("\n"); }
    return 0;
  }
  if (argc != 5) { fprintf(stderr, "usage: scan_all <start> <end> <threads> <outprefix>\n"); return 1; }
  load();
  g_start = strtoull(argv[1], 0, 0); g_end = strtoull(argv[2], 0, 0);
  int nt = atoi(argv[3]);
  char path[512];
  snprintf(path, sizeof path, "%s_hits.tsv", argv[4]); hits = fopen(path, "w");
  fprintf(hits, "seed\tdraw\tcand\tz\tz_first512\n"); fflush(hits);
  if (g_end - g_start < CHUNK * (uint64_t)nt * 4) CHUNK = (g_end - g_start) / ((uint64_t)nt * 4) + 1;
  tstate *ts = calloc(nt, sizeof(tstate));
  pthread_t th[64];
  struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
  for (int i = 0; i < nt; i++) pthread_create(&th[i], NULL, worker, &ts[i]);
  uint64_t total = g_end - g_start, last = 0;
  for (;;) {
    struct timespec rq = {30, 0}; nanosleep(&rq, NULL);
    uint64_t d = atomic_load(&seeds_done);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double el = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
    if (d != last || d >= total) {
      fprintf(stderr, "%6.2f%% of %llu seeds, %.0f s, %.2f M seeds/s, about %.0f s left\n", 100.0 * d / total,
              (unsigned long long)total, el, d / el / 1e6, d ? el * (total - d) / d : 0.0);
      last = d;
    }
    if (d >= total) break;
  }
  for (int i = 0; i < nt; i++) pthread_join(th[i], NULL);
  fclose(hits);
  // largest |z| for each draw and candidate, over all seeds, merged over threads
  snprintf(path, sizeof path, "%s_best.tsv", argv[4]);
  FILE *b = fopen(path, "w");
  uint64_t st2 = 0;
  for (int i = 0; i < nt; i++) st2 += ts[i].stage2;
  fprintf(b, "# seeds %llu..%llu, stage-2 checks %llu\ndraw\tcand\tbest_z\tseed\n", (unsigned long long)g_start,
          (unsigned long long)g_end, (unsigned long long)st2);
  for (int c = 0; c < NCONV; c++)
    for (int k = 0; k < NC; k++) {
      double bz = 0; uint32_t bs = 0;
      for (int i = 0; i < nt; i++)
        if (fabs(ts[i].best[c][k]) > fabs(bz)) { bz = ts[i].best[c][k]; bs = ts[i].best_seed[c][k]; }
      fprintf(b, "%c\t%d\t%+.3f\t%u\n", CONVN[c], k, bz, bs);
    }
  fclose(b);
  return 0;
}
