// THE STREAMING MACHINE, ALL BASE 2, WITH TWO CUTS THAT REDUCE THE BYTES READ A TOKEN. base2_stream.swift's machine,
// unchanged in every part (its kernels are copied here line for line), plus:
//
//  1. THE TOP READ, COLLAPSING, EXACT (top=fr, the default; top=plain is base2_stream.swift's own read).
//     The word table is stored as two planes: each weight w = hi x 256 + lo, hi = w >> 8 (a signed byte), lo = w & 255
//     (an unsigned byte). The high plane is laid line by line (like the plain table), the low plane word by word (each
//     word's 768 low bytes side by side, so a word that stays in is read in one piece).
//       (a) every word's coarse pile from the high bytes only: C = 256 x sum(hi x light)
//       (b) what the low bytes can add is held exactly by the light alone: 255 x (sum of the light's negative values)
//           <= sum(lo x light) <= 255 x (sum of the light's positive values)
//       (c) each end goes through the word's own gauge (power is never decreasing) and gets the standing amount:
//           every word gets a low and a high score. T5 = the 5th-best low. A word whose high is below T5 is below five
//           other words for sure, and is set aside. The rest stay in.
//       (d) only the words still in read their low bytes; their scores are finished exactly (the machine's own pile,
//           gauge and standing amount); the race runs among them (largest, ties to the lowest slot, as the plain race).
//  2. LAYER 0'S FIRST PLATE FROM STORAGE (entry=1), for each new token: the word's and the position's stored rows
//     (entry/, memory-mapped on the CPU side; the two rows copied once a token) plus only the -1/0/+1 carry met with
//     the plate (only the plate rows whose carry is lit are read). The prompt keeps the plain plate read: there the
//     plate is read once for all its positions together.
//
//   check=1  the plain read runs beside the collapsing one at every read position: top 5 (slots and scores), race
//            winner, every survivor's score, the bound on every word, every set-aside word below the plain 5th best;
//            with entry=1 also every qkv.0 pile against the plain plate read. Never used for timing.
//   out=FILE the words of every prompt (ids) as JSON
//
//   swiftc -O -o gpu_fr/base2_stream_fr base2_stream_fr.swift && ./gpu_fr/base2_stream_fr all=1 [top=plain] [entry=1] [check=1]
import Foundation
import Metal

var ARG = [String: String]()
for a in CommandLine.arguments.dropFirst() { let p = a.split(separator: "=", maxSplits: 1).map(String.init); ARG[p[0]] = p.count > 1 ? p[1] : "1" }
func argInt(_ k: String, _ d: Int) -> Int { return Int(ARG[k] ?? "") ?? d }
let CASE = argInt("case", 0), WORDS = argInt("words", 16), MAXSTEPS = argInt("steps", 200), QUIET = ARG["quiet"] != nil
let FR = (ARG["top"] ?? "fr") == "fr", ENTRY = ARG["entry"] == "1", CHECK = ARG["check"] != nil
let dir = (ARG["dir"] ?? "base2_gpu") + "/"
let man = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: dir + "manifest.json"))) as! [String: Any]
let LOG768 = man["log2_768"] as! Int, EPS40 = man["eps40"] as! Int, ALEN = man["arrival_len"] as! Int
let L = 12, D = 768, V = 50257, PMAX = 64

var ORDERL = Array(0..<12); if let sw = ARG["swap"] { let ab = sw.split(separator: ",").map { Int($0)! }; ORDERL.swapAt(ab[0], ab[1]) }
let LG = ARG["groups"] != nil
let INV = ARG["inv"] != nil, DEV = argInt("digits", 16), UP = ARG["fromBottom"] != nil
let ROT = ARG["seed"] != nil ? (argInt("seed", 0) * 7919 + 13) % 1000 : 0
// the two cuts are built for the machine as it is: the positive picture, whole digits, its own wiring
precondition(!(FR || ENTRY) || (!INV && DEV == 16 && ARG["seed"] == nil && ARG["swap"] == nil && !LG), "top=fr / entry=1 run on the machine's own wiring only")
let src = """
#include <metal_stdlib>
using namespace metal;
constant long LOG768 = \(LOG768);
constant long EPS40 = \(EPS40);
constant long ALEN = \(ALEN);
#define RC 8
constant bool INV = \(INV);
constant bool LG = \(LG);         // the light entering every slab grouped by brightness: 4,096 up / 4,096 down
constant int DEV = \(DEV);       // developed digits: 16 = the whole picture
constant bool UP = \(UP);        // the control: develop from the bottom digits up
constant uint ROT = \(ROT);
constant int IMAX = 0x7fffffff;
inline int develop(int v) { if (DEV >= 16) return v; return UP ? (v & ((1 << DEV) - 1)) : (v >> (16 - DEV)) << (16 - DEV); }   // top digits kept (the rest cut), or for the control the bottom ones
inline long rsh(long x, long s) { if (s <= 0) return x << (-s); return (x + (1L << (s - 1))) >> s; }
inline long log2fix(ulong v, device const long* LOG2) {
    int t = 63 - clz(v); ulong m = t >= 16 ? (v >> (t - 16)) & 0xFFFF : (v << (16 - t)) & 0xFFFF; return (long)t * 65536 + LOG2[m]; }
inline long power(long x, long F, device const long* EXP2) { long I = F >> 16, f = F & 0xFFFF; return rsh(x * EXP2[f], 16 - I); }
inline long shfl_down(long v, ushort o) { uint lo = simd_shuffle_down((uint)(ulong)v, o), hi = simd_shuffle_down((uint)((ulong)v >> 32), o); return (long)(((ulong)hi << 32) | lo); }
inline long bcast0(long v) { uint lo = simd_broadcast((uint)(ulong)v, 0), hi = simd_broadcast((uint)((ulong)v >> 32), 0); return (long)(((ulong)hi << 32) | lo); }
inline long sum64(long v) { for (ushort o = 16; o > 0; o >>= 1) v += shfl_down(v, o); return bcast0(v); }

kernel void to16(device const long* X [[buffer(0)]], device int* x16 [[buffer(1)]], device int* ex [[buffer(2)]], device long* sx [[buffer(3)]],
                 constant uint& n [[buffer(4)]], constant uint& r0 [[buffer(5)]], constant uint& sinv [[buffer(6)]], constant int& gsh [[buffer(7)]],
                 uint t [[thread_position_in_threadgroup]], uint g [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint orh[8], orl[8]; threadgroup int sums[8]; uint p = r0 + g; device const long* x = X + p * n;
    ulong o = 0; for (uint i = t; i < n; i += 256) { long v = sinv ? ~x[i] : x[i]; o |= (ulong)(v < 0 ? -v : v); }
    uint hi = simd_or((uint)(o >> 32)), lo = simd_or((uint)o);
    if (lane == 0) { orh[sg] = hi; orl[sg] = lo; } threadgroup_barrier(mem_flags::mem_threadgroup);
    ulong all = 0; for (int k = 0; k < 8; k++) all |= ((ulong)orh[k] << 32) | orl[k];
    int top = all ? 64 - clz(all) : 0; int sh = max(top - 15, 0); int s = 0;
    for (uint i = t; i < n; i += 256) { int v = develop((int)rsh(sinv ? ~x[i] : x[i], sh)); if (gsh > 0) { int hb = 1 << (gsh - 1); v = v < 0 ? -(((-v + hb) >> gsh) << gsh) : (((v + hb) >> gsh) << gsh); } s += v; x16[p * n + i] = INV ? ~v : v; }   // the light, flipped on the negative
    s = simd_sum(s); if (lane == 0) sums[sg] = s; threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) { long S = 0; for (int k = 0; k < 8; k++) S += sums[k]; sx[p] = S; ex[p] = -16 + sh; } }
kernel void gate1(device const short* W [[buffer(0)]], device const int* x16 [[buffer(1)]], device long* part [[buffer(2)]],
                  constant uint& nin [[buffer(3)]], constant uint& nout [[buffer(4)]], constant uint& p [[buffer(5)]], constant uint& span [[buffer(6)]],
                  uint2 id [[thread_position_in_grid]]) {
    // one new position: thread (pile, slice of inputs); neighbouring threads read neighbouring piles' weights; two narrow cells, read once
    uint j = id.x; if (j >= nout) return; uint i0 = id.y * span;
    device const int* x = x16 + p * nin; int lo = 0, hi = 0;
    for (uint i = i0; i < i0 + span; i++) { int pr = (int)W[i * nout + j] * x[i]; lo += pr & 0x7FFF; hi += pr >> 15; }
    part[id.y * nout + j] = ((long)hi << 15) + lo; }
kernel void addparts1(device const long* part [[buffer(0)]], device long* pile [[buffer(1)]], device const long* sx [[buffer(2)]], device const long* sw [[buffer(3)]],
                      constant uint& nout [[buffer(4)]], constant uint& p [[buffer(5)]], constant uint& ks [[buffer(6)]], constant uint& nin [[buffer(7)]], uint j [[thread_position_in_grid]]) {
    if (j >= nout) return; long s = 0; for (uint k = 0; k < ks; k++) s += part[((k + ROT) % ks) * nout + j];
    if (INV) s -= sx[p] + sw[j] + (long)nin;                                 // the negative's pile = the pile + the light's sum + the plate's column sum + the count
    pile[p * nout + j] = s; }
kernel void gateN(device const short* W [[buffer(0)]], device const int* x16 [[buffer(1)]], device long* part [[buffer(2)]],
                  constant uint& nin [[buffer(3)]], constant uint& nout [[buffer(4)]], constant uint& r0 [[buffer(5)]], constant uint& nr [[buffer(6)]],
                  constant uint& span [[buffer(7)]], uint3 id [[thread_position_in_grid]]) {
    // only the moving positions: thread (pile, slice of inputs, group of 8 positions); each weight read once meets all 8
    uint j = id.x; if (j >= nout) return; uint i0 = id.y * span, rb = r0 + id.z * 8, rn = min(8u, r0 + nr - rb);
    int lo[8], hi[8]; for (int r = 0; r < 8; r++) { lo[r] = 0; hi[r] = 0; }
    device const int* x = x16 + rb * nin;                                  // always 8 positions (past the moving ones: rows of nothing, never written out)
    for (uint i = i0; i < i0 + span; i++) { int w = W[i * nout + j];
        #pragma unroll
        for (int r = 0; r < 8; r++) { int pr = w * x[r * nin + i]; lo[r] += pr & 0x7FFF; hi[r] += pr >> 15; } }
    for (uint r = 0; r < rn; r++) part[(id.y * 64 + rb + r) * nout + j] = ((long)hi[r] << 15) + lo[r]; }
kernel void addpartsN(device const long* part [[buffer(0)]], device long* pile [[buffer(1)]], device const long* sx [[buffer(2)]], device const long* sw [[buffer(3)]],
                      constant uint& nout [[buffer(4)]], constant uint& r0 [[buffer(5)]], constant uint& ks [[buffer(6)]], constant uint& n [[buffer(7)]], constant uint& nin [[buffer(8)]],
                      uint t [[thread_position_in_grid]]) {
    if (t >= n) return; uint p = r0 + t / nout, j = t % nout; long s = 0;
    for (uint k = 0; k < ks; k++) s += part[(((k + ROT) % ks) * 64 + p) * nout + j];
    if (INV) s -= sx[p] + sw[j] + (long)nin;
    pile[p * nout + j] = s; }
kernel void gauge_log(device const int* x16 [[buffer(0)]], device const int* ex [[buffer(1)]], device long* LS [[buffer(2)]],
                      device const long* LOG2 [[buffer(3)]], constant uint& r0 [[buffer(4)]], uint lane [[thread_index_in_simdgroup]], uint g [[threadgroup_position_in_grid]]) {
    uint p = r0 + g; long s1 = 0, s2 = 0;
    for (uint i = lane; i < 768; i += 32) { long v = INV ? ~x16[p * 768 + i] : x16[p * 768 + i]; s1 += v; s2 += v * v; }
    s1 = sum64(s1); s2 = sum64(s2);
    if (lane == 0) { long V = 768 * s2 - s1 * s1; int sh = 40 + 2 * ex[p]; V += sh >= 0 ? (EPS40 >> sh) : (EPS40 << -sh);
        LS[p] = (log2fix((ulong)V, LOG2) >> 1) - LOG768 + (long)ex[p] * 65536; } }
kernel void gauged(device const long* pile [[buffer(0)]], device const int* we [[buffer(1)]], device const int* ex [[buffer(2)]], device const long* LS [[buffer(3)]],
                   device const long* bias [[buffer(4)]], device long* out [[buffer(5)]], device const long* EXP2 [[buffer(6)]],
                   constant uint& nout [[buffer(7)]], constant uint& r0 [[buffer(8)]], constant uint& n [[buffer(9)]], uint k [[thread_position_in_grid]]) {
    if (k >= n) return; uint i = r0 * nout + k, p = i / nout, c = i % nout;
    out[i] = power(pile[i], ((long)we[c] + ex[p] + 16) * 65536 - LS[p], EXP2) + bias[c]; }
kernel void plain(device const long* pile [[buffer(0)]], device const int* we [[buffer(1)]], device const int* ex [[buffer(2)]],
                  device const long* bias [[buffer(3)]], device long* R [[buffer(4)]], constant uint& nout [[buffer(5)]], constant uint& r0 [[buffer(6)]],
                  constant uint& n [[buffer(7)]], uint k [[thread_position_in_grid]]) {
    if (k >= n) return; uint i = r0 * nout + k, p = i / nout, c = i % nout;
    long a = rsh(pile[i], -((long)we[c] + ex[p] + 16)) + bias[c]; if (INV) R[i] -= a; else R[i] += a; }   // on the negative stream, adding is subtracting
kernel void edge(device long* H [[buffer(0)]], device const long* PHI [[buffer(1)]], constant uint& off [[buffer(2)]], constant uint& n [[buffer(3)]], uint k [[thread_position_in_grid]]) {
    if (k >= n) return; uint i = off + k; long h = H[i]; long idx = rsh(h, 4) + 32768;
    long share = idx < 0 ? 0 : (idx > 65535 ? 65536 : PHI[idx]); H[i] = rsh(h * share, 16); }
kernel void heads16(device const long* QKV [[buffer(0)]], device int* q16 [[buffer(1)]], device int* eq [[buffer(2)]], constant uint& r0 [[buffer(3)]], constant uint& n [[buffer(4)]],
                    uint k [[thread_position_in_grid]]) {
    if (k >= n) return; uint id = r0 * 36 + k, p = id / 36, part = id % 36;
    device const long* x = QKV + p * 2304 + part * 64; ulong o = 0;
    for (uint d = 0; d < 64; d++) { long v = x[d]; o |= (ulong)(v < 0 ? -v : v); }
    int top = o ? 64 - clz(o) : 0; int sh = max(top - 15, 0);
    for (uint d = 0; d < 64; d++) q16[p * 2304 + part * 64 + d] = (int)rsh(x[d], sh);
    eq[p * 36 + part] = -16 + sh; }
kernel void attend(device const long* QKV [[buffer(0)]], device const int* q16 [[buffer(1)]], device const int* eq [[buffer(2)]], device long* out [[buffer(3)]],
                   device const long* ARR [[buffer(4)]], device const long* LOG2 [[buffer(5)]], device const long* EXP2 [[buffer(6)]], constant uint& r0 [[buffer(7)]],
                   uint lane [[thread_index_in_simdgroup]], uint g [[threadgroup_position_in_grid]]) {
    uint p = r0 + g / 12, h = g % 12; long sc[64];
    for (uint s = 0; s <= p; s++) {
        long dot = (long)q16[p * 2304 + h * 64 + lane] * q16[s * 2304 + 768 + h * 64 + lane]
                 + (long)q16[p * 2304 + h * 64 + lane + 32] * q16[s * 2304 + 768 + h * 64 + lane + 32];
        sc[s] = rsh(sum64(dot), -((long)eq[p * 36 + h] + eq[s * 36 + 12 + h] - 3 + 10)); }
    uint best = 0; for (uint s = 1; s <= p; s++) if (sc[s] > sc[best]) best = s;
    long start = sc[best], num0 = 0, num1 = 0, Z = 0;
    for (uint s = 0; s <= p; s++) { long dist = start - sc[s]; long a = dist < ALEN ? ARR[dist] : 0; if (a == 0) continue; Z += a;
        num0 += a * QKV[s * 2304 + 1536 + h * 64 + lane]; num1 += a * QKV[s * 2304 + 1536 + h * 64 + lane + 32]; }
    long F = -log2fix((ulong)Z, LOG2); long I = F >> 16, f = F & 0xFFFF;
    out[p * 768 + h * 64 + lane] = rsh(num0 * EXP2[f], 16 - I); out[p * 768 + h * 64 + lane + 32] = rsh(num1 * EXP2[f], 16 - I); }
#define BETTER(a, b) (INV ? (a) < (b) : (a) > (b))
kernel void race(device const long* lg [[buffer(0)]], device int* word [[buffer(1)]], constant uint& V [[buffer(2)]], constant uint& r0 [[buffer(3)]],
                 uint t [[thread_position_in_threadgroup]], uint g [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup long bv[32]; threadgroup int bi[32]; uint p = r0 + g; device const long* v = lg + p * V;
    long b = INV ? LONG_MAX : LONG_MIN; int bw = 0;                    // on the negative the race reads every pile flipped: the smallest wins
    for (uint i = t; i < V; i += 1024) { long u = INV ? ~v[i] : v[i]; if (BETTER(u, b)) { b = u; bw = (int)i; } }
    for (ushort o = 16; o > 0; o >>= 1) { long ob = shfl_down(b, o); int ow = simd_shuffle_down(bw, o); if (BETTER(ob, b) || (ob == b && ow < bw)) { b = ob; bw = ow; } }
    if (lane == 0) { bv[sg] = b; bi[sg] = bw; } threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) { long B = bv[0]; int W = bi[0]; for (int k = 1; k < 32; k++) if (BETTER(bv[k], B) || (bv[k] == B && bi[k] < W)) { B = bv[k]; W = bi[k]; } word[p] = W; } }
kernel void nextin(device const int* word [[buffer(0)]], device const int* wordP [[buffer(1)]], device const long* wte [[buffer(2)]], device const long* wpe [[buffer(3)]],
                   device long* R0 [[buffer(4)]], device int* seq [[buffer(5)]], constant uint& p [[buffer(6)]], uint i [[thread_position_in_grid]]) {
    // the word the race just gave at position p goes straight in at p+1: no stop, nothing leaves the GPU
    if (i >= 768) return; int w = wordP[word[p]]; long e = wte[w * 768 + i] + wpe[(p + 1) * 768 + i]; R0[(p + 1) * 768 + i] = INV ? ~e : e; if (i == 0) seq[p + 1] = w; }
kernel void copyl(device const long* a [[buffer(0)]], device long* b [[buffer(1)]], constant uint& off [[buffer(2)]], constant uint& n [[buffer(3)]], uint k [[thread_position_in_grid]]) {
    if (k < n) b[off + k] = a[off + k]; }

// ======================= CUT 1: THE TOP READ, COLLAPSING (high-byte plane for every word, low bytes for the words still in) =======================
// the water level of the light at the top, as gauge_log, and the light's sums of its positive and of its negative values
kernel void gauge_log_fr(device const int* x16 [[buffer(0)]], device const int* ex [[buffer(1)]], device long* LS [[buffer(2)]],
                         device const long* LOG2 [[buffer(3)]], device long* SP [[buffer(4)]], device long* SN [[buffer(5)]], constant uint& r0 [[buffer(6)]],
                         uint lane [[thread_index_in_simdgroup]], uint g [[threadgroup_position_in_grid]]) {
    uint p = r0 + g; long s1 = 0, s2 = 0, sp = 0, sn = 0;
    for (uint i = lane; i < 768; i += 32) { long v = x16[p * 768 + i]; s1 += v; s2 += v * v; if (v > 0) sp += v; else sn += v; }
    s1 = sum64(s1); s2 = sum64(s2); sp = sum64(sp); sn = sum64(sn);
    if (lane == 0) { long V = 768 * s2 - s1 * s1; int sh = 40 + 2 * ex[p]; V += sh >= 0 ? (EPS40 >> sh) : (EPS40 << -sh);
        LS[p] = (log2fix((ulong)V, LOG2) >> 1) - LOG768 + (long)ex[p] * 65536; SP[p] = sp; SN[p] = sn; } }
// (a) the high bytes met with the light: thread (word, slice of 64 lines); |hi x light| < 2^22, 64 of them < 2^28: one 32-bit cell
kernel void coarse1(device const char* H [[buffer(0)]], device const int* x16 [[buffer(1)]], device int* part [[buffer(2)]],
                    constant uint& nin [[buffer(3)]], constant uint& nout [[buffer(4)]], constant uint& p [[buffer(5)]], constant uint& span [[buffer(6)]],
                    uint2 id [[thread_position_in_grid]]) {
    uint j = id.x; if (j >= nout) return; uint i0 = id.y * span;
    device const int* x = x16 + p * nin; int s = 0;
    for (uint i = i0; i < i0 + span; i++) s += (int)H[i * nout + j] * x[i];
    part[id.y * nout + j] = s; }
// (b, c) every word's coarse pile, and its low and high score: the low bytes add between 255 x SN and 255 x SP
kernel void bounds(device const int* part [[buffer(0)]], device const int* we [[buffer(1)]], device const int* ex [[buffer(2)]], device const long* LS [[buffer(3)]],
                   device const long* bias [[buffer(4)]], device const long* SP [[buffer(5)]], device const long* SN [[buffer(6)]], device const long* EXP2 [[buffer(7)]],
                   device long* CP [[buffer(8)]], device long* LOS [[buffer(9)]], device long* HIS [[buffer(10)]], device atomic_uint* cnt [[buffer(11)]],
                   constant uint& nout [[buffer(12)]], constant uint& p [[buffer(13)]], constant uint& ks [[buffer(14)]], uint j [[thread_position_in_grid]]) {
    if (j >= nout) return; if (j == 0) atomic_store_explicit(cnt, 0u, memory_order_relaxed);
    long c = 0; for (uint k = 0; k < ks; k++) c += part[k * nout + j];
    long C = c * 256, F = ((long)we[j] + ex[p] + 16) * 65536 - LS[p];
    CP[j] = C; LOS[j] = power(C + 255 * SN[p], F, EXP2) + bias[j]; HIS[j] = power(C + 255 * SP[p], F, EXP2) + bias[j]; }
// the five best (score, slot) of a list: larger score first, equal scores by the lower slot (the race's own order)
inline bool better(long a, int ia, long b, int ib) { return a > b || (a == b && ia < ib); }
inline void put5(thread long* v, thread int* ix, long u, int iu) {
    if (!better(u, iu, v[4], ix[4])) return;
    v[4] = u; ix[4] = iu;
    for (int k = 4; k > 0; k--) { if (better(v[k], ix[k], v[k - 1], ix[k - 1])) { long tv = v[k]; v[k] = v[k - 1]; v[k - 1] = tv; int ti = ix[k]; ix[k] = ix[k - 1]; ix[k - 1] = ti; } else break; } }
inline void merge5(thread long* v, thread int* ix, thread long* ov, thread int* oi) {
    for (int r = 0; r < 5; r++) {
        long bv = v[0]; int bi = ix[0];
        for (ushort o = 16; o > 0; o >>= 1) { long cv = shfl_down(bv, o); int ci = simd_shuffle_down(bi, o); if (better(cv, ci, bv, bi)) { bv = cv; bi = ci; } }
        bv = bcast0(bv); bi = simd_broadcast(bi, 0);
        if (v[0] == bv && ix[0] == bi) { for (int k = 0; k < 4; k++) { v[k] = v[k + 1]; ix[k] = ix[k + 1]; } v[4] = LONG_MIN; ix[4] = IMAX; }
        ov[r] = bv; oi[r] = bi; } }
kernel void top5(device const long* val [[buffer(0)]], device const uint* list [[buffer(1)]], device atomic_uint* cnt [[buffer(2)]],
                 device int* outS [[buffer(3)]], device long* outV [[buffer(4)]],
                 constant uint& n0 [[buffer(5)]], constant uint& off [[buffer(6)]], constant uint& useList [[buffer(7)]], constant uint& p [[buffer(8)]],
                 uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup long tv[160]; threadgroup int ti[160];
    uint n = useList ? atomic_load_explicit(cnt, memory_order_relaxed) : n0;
    long v[5]; int ix[5]; for (int k = 0; k < 5; k++) { v[k] = LONG_MIN; ix[k] = IMAX; }
    for (uint k = t; k < n; k += 1024) put5(v, ix, val[off + k], useList ? (int)list[k] : (int)k);
    long ov[5]; int oi[5]; merge5(v, ix, ov, oi);
    if (lane == 0) for (int r = 0; r < 5; r++) { tv[sg * 5 + r] = ov[r]; ti[sg * 5 + r] = oi[r]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) { for (int r = 0; r < 5; r++) { v[r] = tv[lane * 5 + r]; ix[r] = ti[lane * 5 + r]; }
        merge5(v, ix, ov, oi);
        if (lane == 0) for (int r = 0; r < 5; r++) { outS[p * 5 + r] = oi[r]; outV[p * 5 + r] = ov[r]; } } }
// the words whose high reaches T5 (the 5th-best low) stay in; one count a simdgroup
kernel void survive(device const long* HIS [[buffer(0)]], device const long* T5 [[buffer(1)]], device atomic_uint* cnt [[buffer(2)]], device uint* list [[buffer(3)]],
                    constant uint& nout [[buffer(4)]], uint j [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    bool keep = j < nout && HIS[j] >= T5[4];
    uint c = simd_sum(keep ? 1u : 0u), pre = simd_prefix_exclusive_sum(keep ? 1u : 0u); uint base = 0;
    if (lane == 0 && c > 0) base = atomic_fetch_add_explicit(cnt, c, memory_order_relaxed);
    base = simd_broadcast(base, 0); if (keep) list[base + pre] = j; }
kernel void args(device atomic_uint* cnt [[buffer(0)]], device uint* ia [[buffer(1)]], device uint* survN [[buffer(2)]], constant uint& p [[buffer(3)]], uint t [[thread_position_in_grid]]) {
    if (t != 0) return; uint n = atomic_load_explicit(cnt, memory_order_relaxed); ia[0] = (n + 7) / 8; ia[1] = 1; ia[2] = 1; survN[p] = n; }
// (d) a word still in: its 768 low bytes (side by side) met with the light, its exact pile, gauge and standing amount; one simdgroup a word
kernel void finish(device const uchar* LOT [[buffer(0)]], device const int* x16 [[buffer(1)]], device const long* CP [[buffer(2)]], device const int* we [[buffer(3)]],
                   device const int* ex [[buffer(4)]], device const long* LS [[buffer(5)]], device const long* bias [[buffer(6)]], device const long* EXP2 [[buffer(7)]],
                   device const uint* list [[buffer(8)]], device atomic_uint* cnt [[buffer(9)]], device long* SC [[buffer(10)]], constant uint& p [[buffer(11)]],
                   uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint g [[threadgroup_position_in_grid]]) {
    uint k = g * 8 + sg; if (k >= atomic_load_explicit(cnt, memory_order_relaxed)) return;
    uint j = list[k]; device const uchar4* lo = (device const uchar4*)(LOT + (ulong)j * 768); device const int4* x = (device const int4*)(x16 + p * 768);
    int s = 0;                                                                  // |lo x light| < 2^23, 24 a lane < 2^28
    for (uint m = 0; m < 6; m++) { int4 l = int4(lo[m * 32 + lane]); int4 xv = x[m * 32 + lane]; s += l.x * xv.x + l.y * xv.y + l.z * xv.z + l.w * xv.w; }
    long Lw = sum64((long)s);
    if (lane == 0) { long F = ((long)we[j] + ex[p] + 16) * 65536 - LS[p]; SC[k] = power(CP[j] + Lw, F, EXP2) + bias[j]; } }
// the race among the words still in: the largest score, equal scores to the lower slot (the plain race's order)
kernel void race_fr(device const long* SC [[buffer(0)]], device const uint* list [[buffer(1)]], device atomic_uint* cnt [[buffer(2)]], device int* word [[buffer(3)]],
                    constant uint& p [[buffer(4)]], uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup long bv[32]; threadgroup int bi[32]; uint n = atomic_load_explicit(cnt, memory_order_relaxed);
    long b = LONG_MIN; int bw = IMAX;
    for (uint k = t; k < n; k += 1024) { long u = SC[k]; int w = (int)list[k]; if (better(u, w, b, bw)) { b = u; bw = w; } }
    for (ushort o = 16; o > 0; o >>= 1) { long ob = shfl_down(b, o); int ow = simd_shuffle_down(bw, o); if (better(ob, ow, b, bw)) { b = ob; bw = ow; } }
    if (lane == 0) { bv[sg] = b; bi[sg] = bw; } threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) { long B = bv[0]; int W = bi[0]; for (int k = 1; k < 32; k++) if (better(bv[k], bi[k], B, W)) { B = bv[k]; W = bi[k]; } word[p] = W; } }
// planes=3: the low byte read as its two nibbles, each plane word by word (384 bytes a word, line 2t in the low half of
// byte t, line 2t+1 in the high half); one simdgroup a word: sum over the 768 lines of nibble x light (< 2^19 each, 24 a lane)
inline int nib384(device const uchar* plane, ulong j, device const int* x16, uint p, uint lane) {
    device const uchar4* c4 = (device const uchar4*)(plane + j * 384); device const int4* x = (device const int4*)(x16 + p * 768); int s = 0;
    for (uint m = 0; m < 3; m++) { uint b = m * 32 + lane; uchar4 c = c4[b]; int4 xa = x[2 * b], xb = x[2 * b + 1];
        s += (int)(c.x & 15) * xa.x + (int)(c.x >> 4) * xa.y + (int)(c.y & 15) * xa.z + (int)(c.y >> 4) * xa.w
           + (int)(c.z & 15) * xb.x + (int)(c.z >> 4) * xb.y + (int)(c.w & 15) * xb.z + (int)(c.w >> 4) * xb.w; }
    return s; }
// stage 2: a word still in after the high byte meets its middle nibbles; the low nibble then adds between 15 x SN and 15 x SP
kernel void finish2(device const uchar* MIDT [[buffer(0)]], device const int* x16 [[buffer(1)]], device const long* CP [[buffer(2)]], device const int* we [[buffer(3)]],
                    device const int* ex [[buffer(4)]], device const long* LS [[buffer(5)]], device const long* bias [[buffer(6)]], device const long* EXP2 [[buffer(7)]],
                    device const uint* list [[buffer(8)]], device atomic_uint* cnt [[buffer(9)]], device const long* SP [[buffer(10)]], device const long* SN [[buffer(11)]],
                    device long* P2 [[buffer(12)]], device long* LO2 [[buffer(13)]], device long* HI2 [[buffer(14)]], device atomic_uint* cnt2 [[buffer(15)]], constant uint& p [[buffer(16)]],
                    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint g [[threadgroup_position_in_grid]]) {
    uint k = g * 8 + sg; if (g == 0 && sg == 0 && lane == 0) atomic_store_explicit(cnt2, 0u, memory_order_relaxed);
    if (k >= atomic_load_explicit(cnt, memory_order_relaxed)) return;
    uint j = list[k]; long M = sum64((long)nib384(MIDT, j, x16, p, lane));
    if (lane == 0) { long F = ((long)we[j] + ex[p] + 16) * 65536 - LS[p], pl = CP[j] + 16 * M;
        P2[k] = pl; LO2[k] = power(pl + 15 * SN[p], F, EXP2) + bias[j]; HI2[k] = power(pl + 15 * SP[p], F, EXP2) + bias[j]; } }
kernel void survive2(device const long* HI2 [[buffer(0)]], device const long* T5 [[buffer(1)]], device atomic_uint* cnt [[buffer(2)]], device const uint* list [[buffer(3)]],
                     device atomic_uint* cnt2 [[buffer(4)]], device uint* list2k [[buffer(5)]], device uint* list2j [[buffer(6)]],
                     uint k [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    bool keep = k < atomic_load_explicit(cnt, memory_order_relaxed) && HI2[k] >= T5[4];
    uint c = simd_sum(keep ? 1u : 0u), pre = simd_prefix_exclusive_sum(keep ? 1u : 0u); uint base = 0;
    if (lane == 0 && c > 0) base = atomic_fetch_add_explicit(cnt2, c, memory_order_relaxed);
    base = simd_broadcast(base, 0); if (keep) { list2k[base + pre] = k; list2j[base + pre] = list[k]; } }
// stage 3: a word still in after the middle nibble meets its low nibbles: its exact pile, gauge and standing amount
kernel void finish3(device const uchar* LOT4 [[buffer(0)]], device const int* x16 [[buffer(1)]], device const long* P2 [[buffer(2)]], device const int* we [[buffer(3)]],
                    device const int* ex [[buffer(4)]], device const long* LS [[buffer(5)]], device const long* bias [[buffer(6)]], device const long* EXP2 [[buffer(7)]],
                    device const uint* list2k [[buffer(8)]], device const uint* list2j [[buffer(9)]], device atomic_uint* cnt2 [[buffer(10)]], device long* SC [[buffer(11)]],
                    constant uint& p [[buffer(12)]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint g [[threadgroup_position_in_grid]]) {
    uint k2 = g * 8 + sg; if (k2 >= atomic_load_explicit(cnt2, memory_order_relaxed)) return;
    uint k = list2k[k2], j = list2j[k2]; long Lo = sum64((long)nib384(LOT4, j, x16, p, lane));
    if (lane == 0) { long F = ((long)we[j] + ex[p] + 16) * 65536 - LS[p]; SC[k2] = power(P2[k] + Lo, F, EXP2) + bias[j]; } }
kernel void chk2(device const long* LGP [[buffer(0)]], device const long* LO2 [[buffer(1)]], device const long* HI2 [[buffer(2)]], device const long* T5b [[buffer(3)]],
                 device const long* TPV [[buffer(4)]], device const uint* list [[buffer(5)]], device atomic_uint* cnt [[buffer(6)]], device atomic_uint* err [[buffer(7)]],
                 constant uint& p [[buffer(8)]], constant uint& V [[buffer(9)]], uint k [[thread_position_in_grid]]) {
    if (k >= atomic_load_explicit(cnt, memory_order_relaxed)) return; uint j = list[k]; long pl = LGP[p * V + j];
    if (pl < LO2[k] || pl > HI2[k]) atomic_fetch_add_explicit(&err[6], 1u, memory_order_relaxed);                     // stage 2's bound broken
    if (HI2[k] < T5b[4] && pl >= TPV[p * 5 + 4]) atomic_fetch_add_explicit(&err[7], 1u, memory_order_relaxed); }     // set aside at stage 2, yet not below
// ---- hold=1: THE LIGHT HELD STILL in each lane (24 lines a lane, read once) while the words still in stream past;
//      a fixed set of NSG simdgroups, each walking every NSG-th word still in ----
constant uint NSG = 1024;                                                       // 128 threadgroups of 256
kernel void finish_h(device const uchar* LOT [[buffer(0)]], device const int* x16 [[buffer(1)]], device const long* CP [[buffer(2)]], device const int* we [[buffer(3)]],
                     device const int* ex [[buffer(4)]], device const long* LS [[buffer(5)]], device const long* bias [[buffer(6)]], device const long* EXP2 [[buffer(7)]],
                     device const uint* list [[buffer(8)]], device atomic_uint* cnt [[buffer(9)]], device long* SC [[buffer(10)]], device uint* survN [[buffer(11)]],
                     constant uint& p [[buffer(12)]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint g [[threadgroup_position_in_grid]]) {
    uint gs = g * 8 + sg, n = atomic_load_explicit(cnt, memory_order_relaxed); if (gs == 0 && lane == 0) survN[p] = n;
    device const int4* x = (device const int4*)(x16 + p * 768); int4 xr[6]; for (uint m = 0; m < 6; m++) xr[m] = x[m * 32 + lane];
    long F0 = (long)(ex[p] + 16) * 65536 - LS[p];
    for (uint k = gs; k < n; k += 2 * NSG) { uint k1 = k + NSG; bool two = k1 < n;          // two words in flight: both words' bytes asked for first
        uint j = list[k], j1 = two ? list[k1] : j;
        device const uchar4* lo = (device const uchar4*)(LOT + (ulong)j * 768); device const uchar4* lo1 = (device const uchar4*)(LOT + (ulong)j1 * 768);
        uchar4 a[6], b[6]; for (uint m = 0; m < 6; m++) a[m] = lo[m * 32 + lane]; for (uint m = 0; m < 6; m++) b[m] = lo1[m * 32 + lane];
        int s = 0, s1 = 0;
        for (uint m = 0; m < 6; m++) { int4 l = int4(a[m]), l1 = int4(b[m]); s += l.x * xr[m].x + l.y * xr[m].y + l.z * xr[m].z + l.w * xr[m].w;
            s1 += l1.x * xr[m].x + l1.y * xr[m].y + l1.z * xr[m].z + l1.w * xr[m].w; }
        long Lw = sum64((long)s), Lw1 = sum64((long)s1);
        if (lane == 0) { SC[k] = power(CP[j] + Lw, (long)we[j] * 65536 + F0, EXP2) + bias[j];
            if (two) SC[k1] = power(CP[j1] + Lw1, (long)we[j1] * 65536 + F0, EXP2) + bias[j1]; } } }
inline int nibC(thread const uchar4* cc, thread const int4* xa, thread const int4* xb) {
    int s = 0;
    for (uint m = 0; m < 3; m++) { uchar4 c = cc[m];
        s += (int)(c.x & 15) * xa[m].x + (int)(c.x >> 4) * xa[m].y + (int)(c.y & 15) * xa[m].z + (int)(c.y >> 4) * xa[m].w
           + (int)(c.z & 15) * xb[m].x + (int)(c.z >> 4) * xb[m].y + (int)(c.w & 15) * xb[m].z + (int)(c.w >> 4) * xb[m].w; }
    return s; }
inline int nibH(device const uchar* plane, ulong j, thread const int4* xa, thread const int4* xb, uint lane) {
    device const uchar4* c4 = (device const uchar4*)(plane + j * 384); uchar4 cc[3]; for (uint m = 0; m < 3; m++) cc[m] = c4[m * 32 + lane]; return nibC(cc, xa, xb); }
kernel void finish2_h(device const uchar* MIDT [[buffer(0)]], device const int* x16 [[buffer(1)]], device const long* CP [[buffer(2)]], device const int* we [[buffer(3)]],
                      device const int* ex [[buffer(4)]], device const long* LS [[buffer(5)]], device const long* bias [[buffer(6)]], device const long* EXP2 [[buffer(7)]],
                      device const uint* list [[buffer(8)]], device atomic_uint* cnt [[buffer(9)]], device const long* SP [[buffer(10)]], device const long* SN [[buffer(11)]],
                      device long* P2 [[buffer(12)]], device long* LO2 [[buffer(13)]], device long* HI2 [[buffer(14)]], device atomic_uint* cnt2 [[buffer(15)]], device uint* survN [[buffer(16)]],
                      constant uint& p [[buffer(17)]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint g [[threadgroup_position_in_grid]]) {
    uint gs = g * 8 + sg, n = atomic_load_explicit(cnt, memory_order_relaxed);
    if (gs == 0 && lane == 0) { survN[p] = n; atomic_store_explicit(cnt2, 0u, memory_order_relaxed); }
    device const int4* x = (device const int4*)(x16 + p * 768); int4 xa[3], xb[3];
    for (uint m = 0; m < 3; m++) { uint b = m * 32 + lane; xa[m] = x[2 * b]; xb[m] = x[2 * b + 1]; }
    long F0 = (long)(ex[p] + 16) * 65536 - LS[p], sp = 15 * SP[p], sn = 15 * SN[p];
    for (uint k = gs; k < n; k += 2 * NSG) { uint k1 = k + NSG; bool two = k1 < n;          // two words in flight
        uint j = list[k], j1 = two ? list[k1] : j;
        device const uchar4* c4 = (device const uchar4*)(MIDT + (ulong)j * 384); device const uchar4* d4 = (device const uchar4*)(MIDT + (ulong)j1 * 384);
        uchar4 cc[3], dd[3]; for (uint m = 0; m < 3; m++) cc[m] = c4[m * 32 + lane]; for (uint m = 0; m < 3; m++) dd[m] = d4[m * 32 + lane];
        long M = sum64((long)nibC(cc, xa, xb)), M1 = sum64((long)nibC(dd, xa, xb));
        if (lane == 0) { long F = (long)we[j] * 65536 + F0, pl = CP[j] + 16 * M;
            P2[k] = pl; LO2[k] = power(pl + sn, F, EXP2) + bias[j]; HI2[k] = power(pl + sp, F, EXP2) + bias[j];
            if (two) { long F1 = (long)we[j1] * 65536 + F0, pl1 = CP[j1] + 16 * M1;
                P2[k1] = pl1; LO2[k1] = power(pl1 + sn, F1, EXP2) + bias[j1]; HI2[k1] = power(pl1 + sp, F1, EXP2) + bias[j1]; } } } }
kernel void finish3_h(device const uchar* LOT4 [[buffer(0)]], device const int* x16 [[buffer(1)]], device const long* P2 [[buffer(2)]], device const int* we [[buffer(3)]],
                      device const int* ex [[buffer(4)]], device const long* LS [[buffer(5)]], device const long* bias [[buffer(6)]], device const long* EXP2 [[buffer(7)]],
                      device const uint* list2k [[buffer(8)]], device const uint* list2j [[buffer(9)]], device atomic_uint* cnt2 [[buffer(10)]], device long* SC [[buffer(11)]],
                      device uint* survN [[buffer(12)]], constant uint& p [[buffer(13)]], constant uint& nsg [[buffer(14)]],
                      uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint g [[threadgroup_position_in_grid]]) {
    uint gs = g * 8 + sg, n = atomic_load_explicit(cnt2, memory_order_relaxed); if (gs == 0 && lane == 0) survN[p] = n;
    device const int4* x = (device const int4*)(x16 + p * 768); int4 xa[3], xb[3];
    for (uint m = 0; m < 3; m++) { uint b = m * 32 + lane; xa[m] = x[2 * b]; xb[m] = x[2 * b + 1]; }
    long F0 = (long)(ex[p] + 16) * 65536 - LS[p];
    for (uint k2 = gs; k2 < n; k2 += nsg) { uint k = list2k[k2], j = list2j[k2]; long Lo = sum64((long)nibH(LOT4, j, xa, xb, lane));
        if (lane == 0) SC[k2] = power(P2[k] + Lo, (long)we[j] * 65536 + F0, EXP2) + bias[j]; } }
// the five best in two passes: 64 threadgroups of 256 each give their five, then one simdgroup the five of those 320;
// with wr = 1 the first of them is the race's winner (largest score, equal scores to the lower slot)
kernel void top5a(device const long* val [[buffer(0)]], device const uint* list [[buffer(1)]], device atomic_uint* cnt [[buffer(2)]],
                  device int* partS [[buffer(3)]], device long* partV [[buffer(4)]], constant uint& n0 [[buffer(5)]], constant uint& off [[buffer(6)]],
                  constant uint& useList [[buffer(7)]], uint t [[thread_position_in_grid]], uint g [[threadgroup_position_in_grid]], uint nt [[threads_per_grid]],
                  uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup long tv[40]; threadgroup int ti[40];
    uint n = useList ? atomic_load_explicit(cnt, memory_order_relaxed) : n0;
    long v[5]; int ix[5]; for (int k = 0; k < 5; k++) { v[k] = LONG_MIN; ix[k] = IMAX; }
    for (uint k = t; k < n; k += nt) put5(v, ix, val[off + k], useList ? (int)list[k] : (int)k);
    long ov[5]; int oi[5]; merge5(v, ix, ov, oi);
    if (lane == 0) for (int r = 0; r < 5; r++) { tv[sg * 5 + r] = ov[r]; ti[sg * 5 + r] = oi[r]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) { for (int r = 0; r < 5; r++) { v[r] = lane < 8 ? tv[lane * 5 + r] : LONG_MIN; ix[r] = lane < 8 ? ti[lane * 5 + r] : IMAX; }
        merge5(v, ix, ov, oi);
        if (lane == 0) for (int r = 0; r < 5; r++) { partS[g * 5 + r] = oi[r]; partV[g * 5 + r] = ov[r]; } } }
kernel void top5b(device const int* partS [[buffer(0)]], device const long* partV [[buffer(1)]], device int* outS [[buffer(2)]], device long* outV [[buffer(3)]],
                  device int* word [[buffer(4)]], constant uint& nparts [[buffer(5)]], constant uint& p [[buffer(6)]], constant uint& wr [[buffer(7)]],
                  uint lane [[thread_index_in_simdgroup]]) {
    long v[5]; int ix[5]; for (int k = 0; k < 5; k++) { v[k] = LONG_MIN; ix[k] = IMAX; }
    for (uint k = lane; k < nparts; k += 32) put5(v, ix, partV[k], partS[k]);
    long ov[5]; int oi[5]; merge5(v, ix, ov, oi);
    if (lane == 0) { for (int r = 0; r < 5; r++) { outS[p * 5 + r] = oi[r]; outV[p * 5 + r] = ov[r]; } if (wr) word[p] = oi[0]; } }
// CHECK PATH (never used by the machine): against the plain read's scores LGP
kernel void chk_top(device const long* LGP [[buffer(0)]], device const long* LOS [[buffer(1)]], device const long* HIS [[buffer(2)]], device const long* T5 [[buffer(3)]],
                    device const long* TPV [[buffer(4)]], device const uint* list [[buffer(5)]], device atomic_uint* cnt [[buffer(6)]], device const long* SC [[buffer(7)]],
                    device atomic_uint* err [[buffer(8)]], constant uint& p [[buffer(9)]], constant uint& V [[buffer(10)]], uint j [[thread_position_in_grid]]) {
    if (j >= V) return; long pl = LGP[p * V + j];
    if (pl < LOS[j] || pl > HIS[j]) atomic_fetch_add_explicit(&err[1], 1u, memory_order_relaxed);                    // the bound broken
    if (HIS[j] < T5[4] && pl >= TPV[p * 5 + 4]) atomic_fetch_add_explicit(&err[2], 1u, memory_order_relaxed);        // set aside, yet not below the plain 5th best
    if (j < atomic_load_explicit(cnt, memory_order_relaxed)) { uint w = list[j]; if (SC[j] != LGP[p * V + w]) atomic_fetch_add_explicit(&err[0], 1u, memory_order_relaxed); } }

// ======================= CUT 2: LAYER 0'S FIRST PLATE FROM STORAGE (one new position) =======================
// the carry: the light's line less the word's and the position's rounded rows at the water level's shift: -1, 0 or +1
kernel void carry0(device const int* x16 [[buffer(0)]], device const int* ex [[buffer(1)]], device const int* seq [[buffer(2)]], device const long* wte [[buffer(3)]],
                   device const long* wpe [[buffer(4)]], device int* carry [[buffer(5)]], device atomic_uint* err [[buffer(6)]], device atomic_uint* litN [[buffer(7)]],
                   constant uint& p [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    if (i >= 768) return; long s = (long)ex[p] + 16; long w = seq[p];
    long c = (long)x16[p * 768 + i] - rsh(wte[w * 768 + i], s) - rsh(wpe[p * 768 + i], s);
    if (c < -1 || c > 1) atomic_fetch_add_explicit(&err[4], 1u, memory_order_relaxed);
    carry[i] = (int)c; if (c != 0) atomic_fetch_add_explicit(&litN[p], 1u, memory_order_relaxed); }
// the carry met with the plate: only the rows whose carry is lit are read (the same line for the whole simdgroup)
kernel void entry_gate1(device const short* W [[buffer(0)]], device const int* carry [[buffer(1)]], device long* part [[buffer(2)]],
                        constant uint& nout [[buffer(3)]], constant uint& span [[buffer(4)]], uint2 id [[thread_position_in_grid]]) {
    uint j = id.x; if (j >= nout) return; uint i0 = id.y * span; int acc = 0;
    for (uint i = i0; i < i0 + span; i++) { int c = carry[i]; if (c != 0) acc += c * (int)W[i * nout + j]; }
    part[id.y * nout + j] = acc; }
// pile = the word's stored row + the position's stored row + the carry's parts (at the shifts with a carry)
kernel void entry_add1(device const long* part [[buffer(0)]], device const long* rows [[buffer(1)]], device long* pile [[buffer(2)]], device const int* sB [[buffer(3)]],
                       device const int* ex [[buffer(4)]], device atomic_uint* err [[buffer(5)]], constant uint& nout [[buffer(6)]], constant uint& p [[buffer(7)]],
                       constant uint& ks [[buffer(8)]], uint j [[thread_position_in_grid]]) {
    if (j >= nout) return; int s = sB[p];
    if (j == 0 && s != ex[p] + 16) atomic_fetch_add_explicit(&err[5], 1u, memory_order_relaxed);
    long v = rows[(p * 2) * nout + j] + rows[(p * 2 + 1) * nout + j];
    if (s == 1 || s == 2) for (uint k = 0; k < ks; k++) v += part[k * nout + j];
    pile[p * nout + j] = v; }
kernel void chk_pile(device const long* a [[buffer(0)]], device const long* b [[buffer(1)]], device atomic_uint* err [[buffer(2)]], constant uint& off [[buffer(3)]],
                     constant uint& n [[buffer(4)]], uint k [[thread_position_in_grid]]) {
    if (k < n && a[off + k] != b[off + k]) atomic_fetch_add_explicit(&err[3], 1u, memory_order_relaxed); }
"""
setvbuf(stdout, nil, _IONBF, 0)
let dev = MTLCreateSystemDefaultDevice()!, queue = dev.makeCommandQueue()!
let lib = try! dev.makeLibrary(source: src, options: MTLCompileOptions())
func pipe(_ n: String) -> MTLComputePipelineState { return try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
let pTo16 = pipe("to16"), pGate1 = pipe("gate1"), pAdd1 = pipe("addparts1"), pGateN = pipe("gateN"), pAddN = pipe("addpartsN"), pGaugeLog = pipe("gauge_log"), pGauged = pipe("gauged"), pPlain = pipe("plain"), pEdge = pipe("edge")
let pHeads = pipe("heads16"), pAttend = pipe("attend"), pRace = pipe("race"), pCopy = pipe("copyl"), pNext = pipe("nextin")
let pGaugeLogFr = pipe("gauge_log_fr"), pCoarse = pipe("coarse1"), pBounds = pipe("bounds"), pTop5 = pipe("top5"), pSurvive = pipe("survive"), pArgs = pipe("args")
let pFinish = pipe("finish"), pRaceFr = pipe("race_fr"), pChkTop = pipe("chk_top"), pCarry = pipe("carry0"), pEGate = pipe("entry_gate1"), pEAdd = pipe("entry_add1"), pChkPile = pipe("chk_pile")
let pFinish2 = pipe("finish2"), pSurvive2 = pipe("survive2"), pFinish3 = pipe("finish3"), pChk2 = pipe("chk2")
let PL3 = FR && argInt("planes", 2) == 3, HOLD = FR && argInt("hold", 1) == 1
let pFinishH = pipe("finish_h"), pFinish2H = pipe("finish2_h"), pFinish3H = pipe("finish3_h"), pTop5a = pipe("top5a"), pTop5b = pipe("top5b")
func loadData(_ name: String) -> Data { return try! Data(contentsOf: URL(fileURLWithPath: dir + name)) }
func bufData(_ d: Data) -> MTLBuffer { return d.withUnsafeBytes { dev.makeBuffer(bytes: $0.baseAddress!, length: d.count, options: .storageModeShared)! } }
func zbuf(_ bytes: Int) -> MTLBuffer { return dev.makeBuffer(length: max(bytes, 8), options: .storageModeShared)! }
func i64(_ name: String) -> [Int64] { let d = loadData(name); return d.withUnsafeBytes { Array($0.bindMemory(to: Int64.self)) } }
struct Bank { let nin, nout: Int; let w, e, b: MTLBuffer; var sw: MTLBuffer? = nil }
var bankList = [(String, Int, Int)]()
for b in man["banks"] as! [[String: Any]] { bankList.append((b["name"] as! String, b["nin"] as! Int, b["nout"] as! Int)) }
func arr<T>(_ name: String, _ t: T.Type) -> [T] { let d = loadData(name); return d.withUnsafeBytes { Array($0.bindMemory(to: T.self)) } }
func buf<T>(_ a: [T]) -> MTLBuffer { return a.withUnsafeBytes { dev.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! } }
struct Seeded: RandomNumberGenerator { var x: UInt64; mutating func next() -> UInt64 { x &+= 0x9E3779B97F4A7C15; var z = x; z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9; z = (z ^ (z >> 27)) &* 0x94D049BB133111EB; return z ^ (z >> 31) } }
var G = Seeded(x: UInt64(argInt("seed", 0)))
let SHUF = ARG["seed"] != nil
func perm(_ n: Int) -> [Int] { return SHUF ? Array(0..<n).shuffled(using: &G) : Array(0..<n) }
let chan = perm(768)                                                      // the stream's channels: one wiring for every layer
var rowP = [String: [Int]](), colP = [String: [Int]]()
for d in 0..<12 {
    let hp = perm(12), dqk = perm(64), dv = perm(64), neu = perm(3072)
    var qkvCols = [Int](); for part in 0..<3 { for h in 0..<12 { for k in 0..<64 { qkvCols.append(part * 768 + hp[h] * 64 + (part < 2 ? dqk[k] : dv[k])) } } }
    var projRows = [Int](); for h in 0..<12 { for k in 0..<64 { projRows.append(hp[h] * 64 + dv[k]) } }
    rowP["qkv.\(d)"] = chan; colP["qkv.\(d)"] = qkvCols
    rowP["proj.\(d)"] = projRows; colP["proj.\(d)"] = chan
    rowP["fc.\(d)"] = chan; colP["fc.\(d)"] = neu
    rowP["mproj.\(d)"] = neu; colP["mproj.\(d)"] = chan }
let wordP = perm(50257)                                                   // the words at the top: slot j holds word wordP[j]
rowP["head"] = chan; colP["head"] = wordP
var banks = [String: Bank]()
var headW2 = [Int16]()
for (n, nin, nout) in bankList {
    let w = arr("\(n).wc", Int16.self), e = arr("\(n).e", Int32.self), b = arr("\(n).b", Int64.self), rp = rowP[n]!, cp = colP[n]!
    var w2 = [Int16](repeating: 0, count: w.count)
    for i in 0..<nin { let ri = rp[i] * nout, wi = i * nout; for j in 0..<nout { w2[wi + j] = w[ri + cp[j]] } }
    if DEV < 16 { for k in 0..<w2.count { let v = Int32(w2[k]); w2[k] = Int16(UP ? (v & ((1 << DEV) - 1)) : (v >> (16 - DEV)) << (16 - DEV)) } }
    var sw = [Int64](repeating: 0, count: nout); for i in 0..<nin { for j in 0..<nout { sw[j] += Int64(w2[i * nout + j]) } }
    if n == "head" { headW2 = w2 }
    banks[n] = Bank(nin: nin, nout: nout, w: buf(INV ? w2.map { ~$0 } : w2), e: buf(cp.map { e[$0] }), b: buf(cp.map { b[$0] }), sw: buf(sw)) }
let PHI = bufData(loadData("phi.bin")), ARR = bufData(loadData("arrival.bin")), LOG2 = bufData(loadData("log2.bin")), EXP2 = bufData(loadData("exp2.bin"))
let wte0 = i64("wte.bin"), wpe0 = i64("wpe.bin")
var wte = wte0, wpe = wpe0
for t in 0..<50257 { for i in 0..<768 { wte[t * 768 + i] = wte0[t * 768 + chan[i]] } }
for t in 0..<1024 { for i in 0..<768 { wpe[t * 768 + i] = wpe0[t * 768 + chan[i]] } }
let wteB = buf(wte), wpeB = buf(wpe), wordPB = buf(wordP.map { Int32($0) }), seqB = zbuf(PMAX * 4)
let ref = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: NSString(string: ARG["refs"] ?? "refs_many.json").expandingTildeInPath))) as! [[String: Any]]
let vocab: [Int: String] = {
    let base = NSString(string: "~/.cache/huggingface/hub/models--gpt2/snapshots").expandingTildeInPath
    let j = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: base + "/" + (try! FileManager.default.contentsOfDirectory(atPath: base)).first! + "/vocab.json"))) as! [String: Int]
    var v = [Int: String](); for (k, i) in j { v[i] = k.replacingOccurrences(of: "Ġ", with: " ").replacingOccurrences(of: "Ċ", with: "\\n") }; return v }()
func text(_ t: [Int]) -> String { return "'" + t.map { vocab[$0] ?? "?" }.joined() + "'" }

// ---- cut 1: the word table as two planes (built once from the machine's own head plate; nothing is rounded or dropped) ----
let NIN_H = 768
var hiPlane = [Int8](repeating: 0, count: NIN_H * V), loPlaneT = [UInt8](repeating: 0, count: V * NIN_H)
if FR {
    var bad = 0
    for i in 0..<NIN_H { for j in 0..<V { let w = Int32(headW2[i * V + j]); let h = w >> 8, l = w & 0xFF
        if h * 256 + l != w || h < -128 || h > 127 { bad += 1 }
        hiPlane[i * V + j] = Int8(truncatingIfNeeded: h); loPlaneT[j * NIN_H + i] = UInt8(truncatingIfNeeded: l) } }
    precondition(bad == 0, "a weight that is not hi x 256 + lo")
}
let hiB = buf(FR ? hiPlane : [Int8](repeating: 0, count: 8)), loTB = buf(FR ? loPlaneT : [UInt8](repeating: 0, count: 8))
hiPlane = []; loPlaneT = []
// planes=3: the low byte as two nibble planes, lb = mid x 16 + lo, each word by word, two lines a byte
var midT = [UInt8](repeating: 0, count: PL3 ? V * 384 : 8), lo4T = [UInt8](repeating: 0, count: PL3 ? V * 384 : 8)
if PL3 {
    for i in 0..<NIN_H { for j in 0..<V { let lb = Int32(headW2[i * V + j]) & 0xFF, mid = UInt8(lb >> 4), lo = UInt8(lb & 15), b = j * 384 + i / 2
        if i % 2 == 0 { midT[b] |= mid; lo4T[b] |= lo } else { midT[b] |= mid << 4; lo4T[b] |= lo << 4 } } }
    var bad = 0                                                           // every weight rebuilt from its three parts
    for i in 0..<NIN_H { for j in 0..<V { let w = Int32(headW2[i * V + j]), b = j * 384 + i / 2, sh = UInt8(i % 2 == 0 ? 0 : 4)
        let mid = Int32((midT[b] >> sh) & 15), lo = Int32((lo4T[b] >> sh) & 15); if (w >> 8) * 256 + mid * 16 + lo != w { bad += 1 } } }
    precondition(bad == 0, "a weight that is not hi x 256 + mid x 16 + lo")
}
let midTB = buf(midT), lo4TB = buf(lo4T)
midT = []; lo4T = []

// ---- cut 2: the stored entry, memory-mapped on the CPU side (never loaded whole) ----
final class Npy {
    let data: NSData; let base: UnsafeRawPointer; let shape: [Int]; let descr: String
    init(_ path: String) {
        data = try! NSData(contentsOfFile: path, options: .alwaysMapped)
        let b = data.bytes.assumingMemoryBound(to: UInt8.self)
        precondition(b[0] == 0x93 && b[1] == 0x4E, "not a .npy file: \(path)")
        let major = b[6]; var hlen = 0, off = 0
        if major == 1 { hlen = Int(b[8]) | Int(b[9]) << 8; off = 10 } else { hlen = Int(b[8]) | Int(b[9]) << 8 | Int(b[10]) << 16 | Int(b[11]) << 24; off = 12 }
        let header = String(bytes: UnsafeBufferPointer(start: b + off, count: hlen), encoding: .ascii)!
        precondition(header.contains("'fortran_order': False"), "fortran order: \(path)")
        let d0 = header.range(of: "'descr': '")!.upperBound; descr = String(header[d0...].prefix { $0 != "'" })
        let s0 = header.range(of: "'shape': (")!.upperBound; let sh = header[s0...].prefix { $0 != ")" }
        shape = sh.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        base = UnsafeRawPointer(b + off + hlen) } }
final class Split40 {                                                     // value = hi x 2^32 + lo (entry/manifest.json)
    let hi: Npy, lo: Npy, cols: Int; var inv: [Int32]
    init(_ name: String) {
        let e = NSString(string: "~/streaming-machine/entry/").expandingTildeInPath + "/"
        hi = Npy(e + name + "_hi.npy"); lo = Npy(e + name + "_lo.npy"); let ids = Npy(e + name + "_ids.npy")
        precondition(hi.descr == "|i1" && lo.descr == "<u4" && ids.descr == "<i4" && hi.shape == lo.shape && hi.shape[0] == ids.shape[0])
        cols = hi.shape[1]; let ip = ids.base.assumingMemoryBound(to: Int32.self); var mx = 0
        for r in 0..<ids.shape[0] { mx = max(mx, Int(ip[r])) }
        inv = [Int32](repeating: -1, count: mx + 1); for r in 0..<ids.shape[0] { inv[Int(ip[r])] = Int32(r) } }
    func row(_ key: Int, _ dst: UnsafeMutablePointer<Int64>) -> Bool {
        guard key < inv.count, inv[key] >= 0 else { return false }
        let r = Int(inv[key]); let h = hi.base.assumingMemoryBound(to: Int8.self) + r * cols, l = lo.base.assumingMemoryBound(to: UInt32.self) + r * cols
        for j in 0..<cols { dst[j] = Int64(h[j]) * 4294967296 + Int64(l[j]) }
        return true } }
let ETAB: [Int: (Split40, Split40)] = ENTRY ? [1: (Split40("T1"), Split40("P1")), 2: (Split40("T2"), Split40("P2"))] : [:]
func shiftAt(_ w: Int, _ p: Int) -> Int {                                  // the water level's shift for the light wte[w] + wpe[p] (to16's own rule)
    var o: UInt64 = 0; for i in 0..<D { let v = wte[w * D + i] + wpe[p * D + i]; o |= UInt64(v < 0 ? -v : v) }
    let top = o == 0 ? 0 : 64 - o.leadingZeroBitCount; return max(top - 15, 0) }

let R = (0...L).map { _ in zbuf(PMAX * D * 8) }
let qkvL = (0..<L).map { _ in zbuf(PMAX * 2304 * 8) }, q16L = (0..<L).map { _ in zbuf(PMAX * 2304 * 4) }, eqL = (0..<L).map { _ in zbuf(PMAX * 36 * 4) }
let sx = zbuf(PMAX * 8), parts = zbuf(12 * 64 * 3072 * 8 + 12 * 50264 * 8), x16 = zbuf((PMAX + 8) * 3072 * 4), ex = zbuf(PMAX * 4), pile = zbuf(PMAX * V * 8), LS = zbuf(PMAX * 8)
let att = zbuf(PMAX * D * 8), hid = zbuf(PMAX * 3072 * 8), lg = zbuf(PMAX * V * 8), words = zbuf(PMAX * 4)
// cut 1's buffers
let SPB = zbuf(PMAX * 8), SNB = zbuf(PMAX * 8), part32 = zbuf(12 * V * 4), CPB = zbuf(V * 8), LOSB = zbuf(V * 8), HISB = zbuf(V * 8), cntB = zbuf(16)
let t5S = zbuf(8 * 4), t5V = zbuf(8 * 8), listB = zbuf(V * 4), SCB = zbuf(V * 8), iaB = zbuf(16), survNB = zbuf(PMAX * 4)
let P2B = zbuf(V * 8), LO2B = zbuf(V * 8), HI2B = zbuf(V * 8), t5bS = zbuf(8 * 4), t5bV = zbuf(8 * 8), cnt2B = zbuf(16), list2k = zbuf(V * 4), list2j = zbuf(V * 4)
let SC2B = zbuf(V * 8), ia2B = zbuf(16), survN2B = zbuf(PMAX * 4), partS = zbuf(64 * 5 * 4), partV = zbuf(64 * 5 * 8)
let wordsPlain = zbuf(PMAX * 4), tpS = zbuf(PMAX * 5 * 4), tpV = zbuf(PMAX * 5 * 8), tfS = zbuf(PMAX * 5 * 4), tfV = zbuf(PMAX * 5 * 8), errB = zbuf(8 * 4)
// cut 2's buffers
let carryB = zbuf(D * 4), partsE = zbuf(12 * 2304 * 8), rowsB = zbuf(PMAX * 2 * 2304 * 8), sB = zbuf(PMAX * 4), litNB = zbuf(PMAX * 4), pileChk = zbuf(PMAX * 2304 * 8)
var P = 0
var CURDEPTH = 0
let NDEPTH = 12
let LVS: [Int] = (ARG["lv"] ?? (ARG["groups"] != nil ? "4096,4096,4096" : "0,0,0")).split(separator: ",").map { Int($0)! }
func shiftFor(_ levels: Int) -> Int { if levels <= 0 { return 0 }; var s = 0; while (32768 >> s) > levels { s += 1 }; return s }
func groupShift(_ bk: Bank) -> Int { return bk.nout > 50000 ? shiftFor(LVS[2]) : shiftFor(CURDEPTH < NDEPTH / 2 ? LVS[0] : LVS[1]) }
func enc(_ cb: MTLCommandBuffer, _ ps: MTLComputePipelineState, _ bufs: [MTLBuffer], _ consts: [Int32], threads: Int, tg: Int, groups: Int = 0, height: Int = 1) {
    let e = cb.makeComputeCommandEncoder()!; encOn(e, ps, bufs, consts, threads: threads, tg: tg, groups: groups, height: height); e.endEncoding()
}
func encOn(_ e: MTLComputeCommandEncoder, _ ps: MTLComputePipelineState, _ bufs: [MTLBuffer], _ consts: [Int32], threads: Int, tg: Int, groups: Int = 0, height: Int = 1) {
    e.setComputePipelineState(ps)
    for (i, b) in bufs.enumerated() { e.setBuffer(b, offset: 0, index: i) }
    for (k, c) in consts.enumerated() { var v = c; e.setBytes(&v, length: 4, index: bufs.count + k) }
    if groups > 0 { e.dispatchThreadgroups(MTLSize(width: groups, height: height, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)) }
    else { e.dispatchThreads(MTLSize(width: threads, height: height, depth: 1), threadsPerThreadgroup: MTLSize(width: min(tg, threads), height: 1, depth: 1)) }
}
func product(_ cb: MTLCommandBuffer, _ X: MTLBuffer, _ bk: Bank, r0: Int, nr: Int, stream: Bool = false) {   // positions r0..r0+nr onto 16 bits, then every core takes piles x a slice of inputs
    enc(cb, pTo16, [X, x16, ex, sx], [Int32(bk.nin), Int32(r0), Int32(stream && INV ? 1 : 0), Int32(groupShift(bk))], threads: 0, tg: 256, groups: nr)
    if nr == 1 { let span = argInt("span", 64), ks = bk.nin / span
        enc(cb, pGate1, [bk.w, x16, parts], [Int32(bk.nin), Int32(bk.nout), Int32(r0), Int32(span)], threads: bk.nout, tg: 256, height: ks)
        enc(cb, pAdd1, [parts, pile, sx, bk.sw!], [Int32(bk.nout), Int32(r0), Int32(ks), Int32(bk.nin)], threads: bk.nout, tg: 256); return }
    let span = bk.nin / 12, ks = 12                                                      // several positions: 8 at a time share each weight read
    let e = cb.makeComputeCommandEncoder()!; e.setComputePipelineState(pGateN)
    for (i, b) in [bk.w, x16, parts].enumerated() { e.setBuffer(b, offset: 0, index: i) }
    for (k, c) in [bk.nin, bk.nout, r0, nr, span].enumerated() { var v = Int32(c); e.setBytes(&v, length: 4, index: 3 + k) }
    e.dispatchThreads(MTLSize(width: bk.nout, height: ks, depth: (nr + 7) / 8), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)); e.endEncoding()
    enc(cb, pAddN, [parts, pile, sx, bk.sw!], [Int32(bk.nout), Int32(r0), Int32(ks), Int32(nr * bk.nout), Int32(bk.nin)], threads: nr * bk.nout, tg: 256)
}
var evRows: MTLSharedEvent? = nil, evWord: MTLSharedEvent? = nil
// cut 2: qkv.0 for one new position from the stored rows; only the carry meets the plate
func entryQKV(_ cb: MTLCommandBuffer, p: Int, bq: Bank) {
    let r = Int32(p)
    enc(cb, pTo16, [R[0], x16, ex, sx], [Int32(bq.nin), r, 0, Int32(groupShift(bq))], threads: 0, tg: 256, groups: 1)
    if CHECK {                                                                             // the plain plate read beside it, into pileChk
        enc(cb, pGate1, [bq.w, x16, parts], [Int32(bq.nin), Int32(bq.nout), r, 64], threads: bq.nout, tg: 256, height: 12)
        enc(cb, pAdd1, [parts, pileChk, sx, bq.sw!], [Int32(bq.nout), r, 12, Int32(bq.nin)], threads: bq.nout, tg: 256) }
    enc(cb, pCarry, [x16, ex, seqB, wteB, wpeB, carryB, errB, litNB], [r], threads: D, tg: 256)
    enc(cb, pEGate, [bq.w, carryB, partsE], [Int32(bq.nout), 64], threads: bq.nout, tg: 256, height: 12)
    cb.encodeWaitForEvent(evRows!, value: UInt64(p))                                       // the CPU has copied this token's two rows
    enc(cb, pEAdd, [partsE, rowsB, pile, sB, ex, errB], [Int32(bq.nout), r, 12], threads: bq.nout, tg: 256)
    if CHECK { enc(cb, pChkPile, [pile, pileChk, errB], [Int32(p * 2304), 2304], threads: 2304, tg: 256) }
}
func layer(_ cb: MTLCommandBuffer, _ d: Int, r0: Int, nr: Int, entry: Bool = false) {
    CURDEPTH = d
    CURDEPTH = d                 // positions r0..r0+nr through one layer, all at once
    let o = ORDERL[d]                                                        // which depth's plates sit at this depth (the swap test)
    let bq = banks["qkv.\(o)"]!, bp = banks["proj.\(o)"]!, bf = banks["fc.\(o)"]!, bm = banks["mproj.\(o)"]!
    let r = Int32(r0)
    if entry && d == 0 && nr == 1 { entryQKV(cb, p: r0, bq: bq) } else { product(cb, R[d], bq, r0: r0, nr: nr, stream: true) }
    enc(cb, pGaugeLog, [x16, ex, LS, LOG2], [r], threads: 0, tg: 32, groups: nr)
    enc(cb, pGauged, [pile, bq.e, ex, LS, bq.b, qkvL[d], EXP2], [2304, r, Int32(nr * 2304)], threads: nr * 2304, tg: 256)
    enc(cb, pHeads, [qkvL[d], q16L[d], eqL[d]], [r, Int32(nr * 36)], threads: nr * 36, tg: 36)
    enc(cb, pAttend, [qkvL[d], q16L[d], eqL[d], att, ARR, LOG2, EXP2], [r], threads: 0, tg: 32, groups: nr * 12)
    enc(cb, pCopy, [R[d], R[d + 1]], [Int32(r0 * D), Int32(nr * D)], threads: nr * D, tg: 256)
    product(cb, att, bp, r0: r0, nr: nr)
    enc(cb, pPlain, [pile, bp.e, ex, bp.b, R[d + 1]], [Int32(D), r, Int32(nr * D)], threads: nr * D, tg: 256)
    product(cb, R[d + 1], bf, r0: r0, nr: nr, stream: true); enc(cb, pGaugeLog, [x16, ex, LS, LOG2], [r], threads: 0, tg: 32, groups: nr)
    enc(cb, pGauged, [pile, bf.e, ex, LS, bf.b, hid, EXP2], [3072, r, Int32(nr * 3072)], threads: nr * 3072, tg: 256)
    enc(cb, pEdge, [hid, PHI], [Int32(r0 * 3072), Int32(nr * 3072)], threads: nr * 3072, tg: 256)
    product(cb, hid, bm, r0: r0, nr: nr)
    enc(cb, pPlain, [pile, bm.e, ex, bm.b, R[d + 1]], [Int32(D), r, Int32(nr * D)], threads: nr * D, tg: 256)
}
func readTopPlain(_ cb: MTLCommandBuffer, r0: Int, nr: Int, word: MTLBuffer) {             // base2_stream.swift's readTop, the race into `word`
    let bh = banks["head"]!
    product(cb, R[L], bh, r0: r0, nr: nr, stream: true); enc(cb, pGaugeLog, [x16, ex, LS, LOG2], [Int32(r0)], threads: 0, tg: 32, groups: nr)
    enc(cb, pGauged, [pile, bh.e, ex, LS, bh.b, lg, EXP2], [Int32(V), Int32(r0), Int32(nr * V)], threads: nr * V, tg: 256)
    enc(cb, pRace, [lg, word], [Int32(V), Int32(r0)], threads: 0, tg: 1024, groups: nr)
}
var STOP = 99                                                                              // bench only: stop the read after stage group STOP
let ONE = FR && argInt("one", 1) == 1                                                      // the collapsing read in one serial encoder
func readTopFr(_ cb: MTLCommandBuffer, p: Int) {                                           // cut 1: one read position
    let bh = banks["head"]!, r = Int32(p), v = Int32(V)
    if CHECK {                                                                             // the plain read beside it: scores into lg, its top 5, its race
        readTopPlain(cb, r0: p, nr: 1, word: wordsPlain)
        enc(cb, pTop5, [lg, listB, cntB, tpS, tpV], [v, Int32(p * V), 0, r], threads: 0, tg: 1024, groups: 1) }
    let E: MTLComputeCommandEncoder? = ONE ? cb.makeComputeCommandEncoder(dispatchType: .serial)! : nil
    defer { E?.endEncoding() }
    func D(_ ps: MTLComputePipelineState, _ bufs: [MTLBuffer], _ consts: [Int32], threads: Int, tg: Int, groups: Int = 0, height: Int = 1) {
        if let e = E { encOn(e, ps, bufs, consts, threads: threads, tg: tg, groups: groups, height: height) } else { enc(cb, ps, bufs, consts, threads: threads, tg: tg, groups: groups, height: height) } }
    func DI(_ ps: MTLComputePipelineState, _ bufs: [MTLBuffer], _ ia: MTLBuffer) {
        let e = E ?? cb.makeComputeCommandEncoder()!; e.setComputePipelineState(ps)
        for (i, b) in bufs.enumerated() { e.setBuffer(b, offset: 0, index: i) }
        var pp = UInt32(p); e.setBytes(&pp, length: 4, index: bufs.count)
        e.dispatchThreadgroups(indirectBuffer: ia, indirectBufferOffset: 0, threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)); if E == nil { e.endEncoding() } }
    // the five best of a list in two passes (hold=1); with word, its first is the race's winner at p
    func five(_ val: MTLBuffer, _ list: MTLBuffer, _ cnt: MTLBuffer, n0: Int, useList: Bool, _ oS: MTLBuffer, _ oV: MTLBuffer, at: Int, word: MTLBuffer? = nil) {
        D(pTop5a, [val, list, cnt, partS, partV], [Int32(n0), 0, useList ? 1 : 0], threads: 0, tg: 256, groups: 64)
        D(pTop5b, [partS, partV, oS, oV, word ?? oS], [320, Int32(at), word == nil ? 0 : 1], threads: 0, tg: 32, groups: 1) }
    // stage 1: the high byte for every word, every word's low and high score
    D(pTo16, [R[L], x16, ex, sx], [Int32(bh.nin), r, 0, Int32(groupShift(bh))], threads: 0, tg: 256, groups: 1)
    D(pGaugeLogFr, [x16, ex, LS, LOG2, SPB, SNB], [r], threads: 0, tg: 32, groups: 1)
    D(pCoarse, [hiB, x16, part32], [Int32(bh.nin), v, r, 64], threads: V, tg: 256, height: 12)
    D(pBounds, [part32, bh.e, ex, LS, bh.b, SPB, SNB, EXP2, CPB, LOSB, HISB, cntB], [v, r, 12], threads: V, tg: 256)
    if STOP == 1 { return }
    if HOLD { five(LOSB, listB, cntB, n0: V, useList: false, t5S, t5V, at: 0) }                        // T5: the 5th-best low
    else { D(pTop5, [LOSB, listB, cntB, t5S, t5V], [v, 0, 0, 0], threads: 0, tg: 1024, groups: 1) }
    D(pSurvive, [HISB, t5V, cntB, listB], [v], threads: V, tg: 256)
    if !HOLD { D(pArgs, [cntB, iaB, survNB], [r], threads: 1, tg: 1) }
    if STOP == 2 { return }
    if !PL3 {                                                                              // two planes: the low byte for the words still in
        if HOLD {
            D(pFinishH, [loTB, x16, CPB, bh.e, ex, LS, bh.b, EXP2, listB, cntB, SCB, survNB], [r], threads: 0, tg: 256, groups: 128)
            five(SCB, listB, cntB, n0: 0, useList: true, tfS, tfV, at: p, word: words) }
        else {
            DI(pFinish, [loTB, x16, CPB, bh.e, ex, LS, bh.b, EXP2, listB, cntB, SCB], iaB)
            D(pRaceFr, [SCB, listB, cntB, words], [r], threads: 0, tg: 1024, groups: 1)
            if CHECK { D(pTop5, [SCB, listB, cntB, tfS, tfV], [0, 0, 1, r], threads: 0, tg: 1024, groups: 1) } }
        if CHECK { D(pChkTop, [lg, LOSB, HISB, t5V, tpV, listB, cntB, SCB, errB], [r, v], threads: V, tg: 256) }
        return }
    // three planes: the middle nibble for the words still in, a second set-aside, the low nibble for the words still in after it
    if HOLD {
        D(pFinish2H, [midTB, x16, CPB, bh.e, ex, LS, bh.b, EXP2, listB, cntB, SPB, SNB, P2B, LO2B, HI2B, cnt2B, survNB], [r], threads: 0, tg: 256, groups: 128)
        if STOP == 3 { return }
        five(LO2B, listB, cntB, n0: 0, useList: true, t5bS, t5bV, at: 0)                            // the 5th-best low after stage 2
        D(pSurvive2, [HI2B, t5bV, cntB, listB, cnt2B, list2k, list2j], [], threads: V, tg: 256)
        D(pFinish3H, [lo4TB, x16, P2B, bh.e, ex, LS, bh.b, EXP2, list2k, list2j, cnt2B, SC2B, survN2B], [r, 128], threads: 0, tg: 256, groups: 16)
        five(SC2B, list2j, cnt2B, n0: 0, useList: true, tfS, tfV, at: p, word: words)
    } else {
        DI(pFinish2, [midTB, x16, CPB, bh.e, ex, LS, bh.b, EXP2, listB, cntB, SPB, SNB, P2B, LO2B, HI2B, cnt2B], iaB)
        if STOP == 3 { return }
        D(pTop5, [LO2B, listB, cntB, t5bS, t5bV], [0, 0, 1, 0], threads: 0, tg: 1024, groups: 1)
        D(pSurvive2, [HI2B, t5bV, cntB, listB, cnt2B, list2k, list2j], [], threads: V, tg: 256)
        D(pArgs, [cnt2B, ia2B, survN2B], [r], threads: 1, tg: 1)
        DI(pFinish3, [lo4TB, x16, P2B, bh.e, ex, LS, bh.b, EXP2, list2k, list2j, cnt2B, SC2B], ia2B)
        D(pRaceFr, [SC2B, list2j, cnt2B, words], [r], threads: 0, tg: 1024, groups: 1)
        if CHECK { D(pTop5, [SC2B, list2j, cnt2B, tfS, tfV], [0, 0, 1, r], threads: 0, tg: 1024, groups: 1) } }
    if CHECK {
        D(pChkTop, [lg, LOSB, HISB, t5V, tpV, list2j, cnt2B, SC2B, errB], [r, v], threads: V, tg: 256)
        D(pChk2, [lg, LO2B, HI2B, t5bV, tpV, listB, cntB, errB], [r, v], threads: V, tg: 256) }
}
func encIndirect(_ cb: MTLCommandBuffer, _ ps: MTLComputePipelineState, _ bufs: [MTLBuffer], _ p: Int, _ ia: MTLBuffer) {   // one simdgroup a word still in
    let e = cb.makeComputeCommandEncoder()!; e.setComputePipelineState(ps)
    for (i, b) in bufs.enumerated() { e.setBuffer(b, offset: 0, index: i) }
    var pp = UInt32(p); e.setBytes(&pp, length: 4, index: bufs.count)
    e.dispatchThreadgroups(indirectBuffer: ia, indirectBufferOffset: 0, threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)); e.endEncoding()
}
func readTop(_ cb: MTLCommandBuffer, p: Int) { if FR { readTopFr(cb, p: p) } else { readTopPlain(cb, r0: p, nr: 1, word: words) } }

// the bytes each read position reads from the word table and the plates (weights only; the light, piles and parts are apart)
let PLATE_BYTES = (0..<12).reduce(0) { a, d in a + ["qkv", "proj", "fc", "mproj"].reduce(0) { b, n in let bk = banks["\(n).\(d)"]!; return b + bk.nin * bk.nout * 2 } }
let HEAD_BYTES = 768 * V * 2, QKV0_BYTES = 768 * 2304 * 2

let CASES = ARG["all"] != nil ? Array(0..<ref.count) : [CASE]
var nIdentical = 0, report = [String](), totalWords = 0, totalTime = 0.0, afterWords = 0, afterTime = 0.0
var allIds = [[Int]](), survAll = [Int](), litAll = [Int](), top5Same = 0, top5Pos = 0, winSame = 0, perPrompt = [[String: Any]]()
var errTot = [Int](repeating: 0, count: 8), surv2All = [Int]()
for CASE in CASES {
    let ids = ref[CASE]["ids"] as! [Int], prompt = ref[CASE]["prompt"] as! String
    let N = min(WORDS, (ref[CASE]["ref"] as! [Int]).count, PMAX - ids.count), want = Array((ref[CASE]["ref"] as! [Int]).prefix(N)), S = ids.count
    P = S + N
    var gen = [Int]()
    memset(errB.contents(), 0, 32); memset(litNB.contents(), 0, PMAX * 4); memset(survNB.contents(), 0, PMAX * 4)
    if ENTRY { evRows = dev.makeSharedEvent()!; evWord = dev.makeSharedEvent()! }
    let t0 = Date()
    let r0p = R[0].contents().bindMemory(to: Int64.self, capacity: P * D)
    let seqp = seqB.contents().bindMemory(to: Int32.self, capacity: PMAX)
    for q in 0..<S { seqp[q] = Int32(ids[q]); for i in 0..<D { let e = wte[ids[q] * D + i] + wpe[q * D + i]; r0p[q * D + i] = INV ? ~e : e } }
    // the prompt all at once (one command buffer), then every new word flows straight into the next position (a second)
    let cb1 = queue.makeCommandBuffer()!
    for d in 0..<L { layer(cb1, d, r0: 0, nr: S) }
    readTop(cb1, p: S - 1)
    if ENTRY { cb1.encodeSignalEvent(evWord!, value: UInt64(S)) }
    cb1.commit()
    let cb2 = queue.makeCommandBuffer()!
    for p in S..<(S + N - 1) {
        enc(cb2, pNext, [words, wordPB, wteB, wpeB, R[0], seqB], [Int32(p - 1)], threads: 768, tg: 256)
        for d in 0..<L { layer(cb2, d, r0: p, nr: 1, entry: ENTRY) }
        readTop(cb2, p: p)
        if ENTRY { cb2.encodeSignalEvent(evWord!, value: UInt64(p + 1)) }
    }
    cb2.commit()
    let wd = words.contents().bindMemory(to: Int32.self, capacity: PMAX)
    if ENTRY {                                                                             // the CPU side: each new word's two stored rows, once a token
        let rp = rowsB.contents().bindMemory(to: Int64.self, capacity: PMAX * 2 * 2304), sp = sB.contents().bindMemory(to: Int32.self, capacity: PMAX)
        for p in S..<(S + N - 1) {
            while evWord!.signaledValue < UInt64(p) { }                                    // the race at p-1 has run
            let w = wordP[Int(wd[p - 1])], s = shiftAt(w, p)
            guard let tb = ETAB[s] else { fatalError("shift \(s) at a new position (\(p)): the entry has no carry form there") }; let (T, Pt) = tb
            precondition(T.row(w, rp + (p * 2) * 2304) && Pt.row(p, rp + (p * 2 + 1) * 2304), "a row that was never stored: word \(w) position \(p) shift \(s)")
            sp[p] = Int32(s)
            evRows!.signaledValue = UInt64(p)
        }
    }
    cb2.waitUntilCompleted()
    let wall = Date().timeIntervalSince(t0)
    precondition(cb1.status == .completed && cb2.status == .completed, "command buffer failed: \(String(describing: cb1.error)) \(String(describing: cb2.error))")
    let tAfter = N > 1 ? cb2.gpuEndTime - cb1.gpuEndTime : 0
    gen = (0..<N).map { wordP[Int(wd[S - 1 + $0])] }
    allIds.append(gen)
    totalWords += N; totalTime += wall; afterWords += N - 1; afterTime += tAfter
    let same = gen == want
    if same { nIdentical += 1 } else {
        let j = (0..<N).first { gen[$0] != want[$0] }!
        let mg = (ref[CASE]["margins"] as? [Double]).map { String(format: "%.4f", $0[j]) } ?? "?"
        report.append("  '\(prompt)': first differs at word \(j + 1): \(text([gen[j]])) where the model says \(text([want[j]])) (the model's own lead there: \(mg))") }
    var line = "'\(prompt)' -> \(text(gen))  \(same ? "IDENTICAL" : "DIFFERENT")  " + String(format: "%.3f s, %.1f words/s (after the prompt: %.1f)", wall, Double(N) / wall, tAfter > 0 ? Double(N - 1) / tAfter : 0)
    var rec: [String: Any] = ["prompt": prompt, "words": gen]
    if FR {
        let sv = survNB.contents().bindMemory(to: UInt32.self, capacity: PMAX); let surv = ((S - 1)..<(S + N - 1)).map { Int(sv[$0]) }
        survAll += surv; rec["survivors"] = surv
        line += "  survivors " + surv.map(String.init).joined(separator: ",")
        if PL3 { let s2 = survN2B.contents().bindMemory(to: UInt32.self, capacity: PMAX); let surv2 = ((S - 1)..<(S + N - 1)).map { Int(s2[$0]) }
            surv2All += surv2; rec["survivors_stage2"] = surv2; line += "  after the middle nibble " + surv2.map(String.init).joined(separator: ",") }
        if CHECK {
            let a = tfS.contents().bindMemory(to: Int32.self, capacity: PMAX * 5), b = tpS.contents().bindMemory(to: Int32.self, capacity: PMAX * 5)
            let av = tfV.contents().bindMemory(to: Int64.self, capacity: PMAX * 5), bv = tpV.contents().bindMemory(to: Int64.self, capacity: PMAX * 5)
            let wp = wordsPlain.contents().bindMemory(to: Int32.self, capacity: PMAX)
            var t5 = 0, ws = 0
            for q in (S - 1)..<(S + N - 1) { top5Pos += 1
                if (0..<5).allSatisfy({ a[q * 5 + $0] == b[q * 5 + $0] && av[q * 5 + $0] == bv[q * 5 + $0] }) { t5 += 1 }
                if wd[q] == wp[q] { ws += 1 } }
            top5Same += t5; winSame += ws; rec["top5_same"] = t5; rec["winner_same"] = ws
            line += "  | top 5 = plain \(t5)/\(N), winner = plain \(ws)/\(N)" } }
    if ENTRY {
        let ln = litNB.contents().bindMemory(to: UInt32.self, capacity: PMAX); let lit = (S..<(S + N - 1)).map { Int(ln[$0]) }
        litAll += lit; rec["lit_lines"] = lit
        line += "  lit carry lines " + lit.map(String.init).joined(separator: ",") }
    let eb = errB.contents().bindMemory(to: UInt32.self, capacity: 8); for k in 0..<8 { errTot[k] += Int(eb[k]) }
    perPrompt.append(rec)
    if !QUIET { print(line) }
}
print("\(nIdentical) of \(CASES.count) prompts identical to the model (fp32 GPT-2's words)  [top=\(FR ? (PL3 ? "fr, 3 planes" : "fr") + (HOLD ? ", light held" : "") + (ONE ? ", one encoder" : "") : "plain") entry=\(ENTRY ? 1 : 0)\(CHECK ? " check" : "")]")
report.forEach { print($0) }
print(String(format: "%d words in %.3f s: %.1f words/s, prompt included; after the prompt %.1f words/s (%d words in %.3f s of GPU time from the prompt's end)", totalWords, totalTime, Double(totalWords) / totalTime, Double(afterWords) / afterTime, afterWords, afterTime))
func med(_ a: [Int]) -> Double { let s = a.sorted(); return s.isEmpty ? 0 : (s.count % 2 == 1 ? Double(s[s.count / 2]) : Double(s[s.count / 2 - 1] + s[s.count / 2]) / 2) }
var summary: [String: Any] = ["top": FR ? "fr" : "plain", "planes": PL3 ? 3 : 2, "hold": HOLD, "one_encoder": ONE, "entry": ENTRY, "check": CHECK, "identical_to_fp32": nIdentical, "prompts": CASES.count,
                              "words_per_s_prompt_included": Double(totalWords) / totalTime, "words_per_s_after_prompt": Double(afterWords) / afterTime]
var bytesTok = Double(PLATE_BYTES + HEAD_BYTES)
if FR {
    let mean = Double(survAll.reduce(0, +)) / Double(max(survAll.count, 1))
    print(String(format: "survivors a read position (%d positions): median %.0f, mean %.1f, least %d, most %d", survAll.count, med(survAll), mean, survAll.min() ?? 0, survAll.max() ?? 0))
    summary["survivors_median"] = med(survAll); summary["survivors_mean"] = mean; summary["survivors_min"] = survAll.min() ?? 0; summary["survivors_max"] = survAll.max() ?? 0
    var topB = Double(768 * V) + mean * 768
    if !PL3 { print(String(format: "word table bytes a read position: %.2f MB (high plane %.2f MB + low bytes of the survivors %.2f MB) against %.2f MB plain: %.1f%%", topB / 1e6, Double(768 * V) / 1e6, mean * 768 / 1e6, Double(HEAD_BYTES) / 1e6, 100 * topB / Double(HEAD_BYTES))) }
    else { let m2 = Double(surv2All.reduce(0, +)) / Double(max(surv2All.count, 1)); topB = Double(768 * V) + mean * 384 + m2 * 384
        print(String(format: "after the middle nibble (%d positions): median %.0f, mean %.1f, least %d, most %d", surv2All.count, med(surv2All), m2, surv2All.min() ?? 0, surv2All.max() ?? 0))
        print(String(format: "word table bytes a read position: %.2f MB (high plane %.2f MB + middle nibbles %.2f MB + low nibbles %.4f MB) against %.2f MB plain: %.1f%%", topB / 1e6, Double(768 * V) / 1e6, mean * 384 / 1e6, m2 * 384 / 1e6, Double(HEAD_BYTES) / 1e6, 100 * topB / Double(HEAD_BYTES)))
        summary["survivors2_median"] = med(surv2All); summary["survivors2_mean"] = m2; summary["survivors2_min"] = surv2All.min() ?? 0; summary["survivors2_max"] = surv2All.max() ?? 0 }
    bytesTok = bytesTok - Double(HEAD_BYTES) + topB; summary["top_bytes"] = topB
    if CHECK {
        print("CHECK top read: top 5 (slots and scores) = the plain read \(top5Same)/\(top5Pos); race winner = the plain read \(winSame)/\(top5Pos); survivors' scores differing from the plain read \(errTot[0]); bound broken \(errTot[1]); set-aside words not below the plain 5th best \(errTot[2])" + (PL3 ? "; stage 2: bound broken \(errTot[6]), set aside yet not below the plain 5th best \(errTot[7])" : ""))
        summary["check_top5_same"] = top5Same; summary["check_positions"] = top5Pos; summary["check_winner_same"] = winSame
        summary["check_survivor_scores_differ"] = errTot[0]; summary["check_bound_broken"] = errTot[1]; summary["check_setaside_not_below"] = errTot[2]; summary["check_stage2_bound_broken"] = errTot[6]; summary["check_stage2_setaside_not_below"] = errTot[7] } }
if ENTRY {
    let mean = Double(litAll.reduce(0, +)) / Double(max(litAll.count, 1))
    let eB = mean * 2304 * 2 + 2 * 2304 * 8
    print(String(format: "stored layer 0: lit carry lines a new token median %.0f, mean %.1f (of 768): qkv.0 bytes a token %.3f MB (lit plate rows %.3f MB + two stored rows %.3f MB) against %.3f MB plain", med(litAll), mean, eB / 1e6, mean * 4608 / 1e6, 2.0 * 2304 * 8 / 1e6, Double(QKV0_BYTES) / 1e6))
    print("CHECK stored layer 0: carry outside -1..1 \(errTot[4]); shift differing from the water level's \(errTot[5])" + (CHECK ? "; qkv.0 piles differing from the plain plate read \(errTot[3])" : ""))
    bytesTok = bytesTok - Double(QKV0_BYTES) + eB
    summary["lit_mean"] = mean; summary["entry_carry_out_of_range"] = errTot[4]; summary["entry_shift_differs"] = errTot[5]; if CHECK { summary["check_entry_piles_differ"] = errTot[3] } }
print(String(format: "weights read a new token (plates + word table): %.2f MB (plain machine: %.2f MB = plates %.2f + word table %.2f)", bytesTok / 1e6, Double(PLATE_BYTES + HEAD_BYTES) / 1e6, Double(PLATE_BYTES) / 1e6, Double(HEAD_BYTES) / 1e6))
summary["weight_bytes_per_token"] = bytesTok; summary["per_prompt"] = perPrompt
if let o = ARG["out"] { try! JSONSerialization.data(withJSONObject: summary, options: [.sortedKeys]).write(to: URL(fileURLWithPath: o)) }
// bench=K (timing only, after every prompt has run and been recorded): K top reads at the last read position, then K whole
// new-token passes (12 layers + the top read) at the last position, each in one command buffer; GPU time per pass
if let kb = ARG["bench"].flatMap({ Int($0) }) {
    let p = P - 2                                                                          // the last read position the run made (its light is real)
    if FR { var st = [Double]()
        for k in [1, 2, 3, 99] { STOP = k; let c = queue.makeCommandBuffer()!; for _ in 0..<kb { readTop(c, p: p) }; c.commit(); c.waitUntilCompleted(); st.append((c.gpuEndTime - c.gpuStartTime) / Double(kb) * 1e6) }
        STOP = 99; let sv = survNB.contents().bindMemory(to: UInt32.self, capacity: PMAX)[p], s2 = survN2B.contents().bindMemory(to: UInt32.self, capacity: PMAX)[p]
        print(String(format: "BENCH stages (position %d, %d words in after the high byte%@): high byte + bounds %.1f us; + 5th-best low + set-aside %.1f us; + %@ %.1f us; whole read %.1f us", p, sv,
                     PL3 ? ", \(s2) after the middle nibble" : "", st[0], st[1], PL3 ? "middle nibbles" : "low bytes and race", st[2], st[3])) }
    let cbA = queue.makeCommandBuffer()!; for _ in 0..<kb { readTop(cbA, p: p) }; cbA.commit(); cbA.waitUntilCompleted()
    let cbB = queue.makeCommandBuffer()!; for _ in 0..<kb { for d in 0..<L { layer(cbB, d, r0: p, nr: 1) }; readTop(cbB, p: p) }; cbB.commit(); cbB.waitUntilCompleted()
    let cbC = queue.makeCommandBuffer()!; for _ in 0..<kb { for d in 0..<L { layer(cbC, d, r0: p, nr: 1) } }; cbC.commit(); cbC.waitUntilCompleted()
    print(String(format: "BENCH top=%@: top read %.1f us; a whole token (12 layers + top read) %.1f us; 12 layers alone %.1f us  (GPU time, %d passes each, position %d)",
                 FR ? (PL3 ? "fr3" : "fr2") : "plain", (cbA.gpuEndTime - cbA.gpuStartTime) / Double(kb) * 1e6, (cbB.gpuEndTime - cbB.gpuStartTime) / Double(kb) * 1e6,
                 (cbC.gpuEndTime - cbC.gpuStartTime) / Double(kb) * 1e6, kb, p))
}
