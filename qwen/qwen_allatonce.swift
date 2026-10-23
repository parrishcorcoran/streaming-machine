// QUANTUMQWEN3: Qwen3-0.6B, every number a whole number in base 2, no float while it runs. The prompt goes in all
// at once; then each new piece flows straight into the next spot, one unbroken run on the GPU.
//   the gauges : log2 = where the top 1 sits + a table read; a root is a halving, a divide a negation
//   the turns  : each pair in a head turned by its position's angle, from whole-number cos/sin tables
//   the gated bend: g * share(g) * u, the share from a table
//   the arrival: the clock starts at the race's winner; each score's distance reads the arrival table
//   the word   : the race at the top
//   swiftc -O -o qwen_stream qwen_stream.swift && ./qwen_stream all=1
import Foundation
import Metal

var ARG = [String: String]()
for a in CommandLine.arguments.dropFirst() { let p = a.split(separator: "=", maxSplits: 1).map(String.init); ARG[p[0]] = p.count > 1 ? p[1] : "1" }
func argInt(_ k: String, _ d: Int) -> Int { return Int(ARG[k] ?? "") ?? d }
let LG = ARG["groups"] != nil
let PARTHALF = 24 * 64 * 6144 + 8 * 8 * 151936, PILEHALF = 64 * 151936
let KV14 = ARG["kv"] != "full"
let CASE = argInt("case", 0), WORDS = argInt("words", 16), QUIET = ARG["quiet"] != nil
let dir = (ARG["dir"] ?? "qwen_gpu") + "/"
let man = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: dir + "manifest.json"))) as! [String: Any]
let EPS40 = man["eps40"] as! Int, ALEN = man["arrival_len"] as! Int, GEXP = man["gain_exps"] as! [Int]
let L = 28, D = 1024, HD = 128, V = 151936, PMAX = 64

let src = """
#include <metal_stdlib>
using namespace metal;
constant long EPS40 = \(EPS40);
constant long ALEN = \(ALEN);
constant bool KV14 = \(KV14);      // the stack (keys and values) held as 16,384 up / 16,384 down
constant bool LG = \(LG);
constant ulong PARTHALF = \(PARTHALF);   // the down parts sit after the up parts
constant ulong PILEHALF = \(PILEHALF);   // the down piles sit after the up piles         // the light entering every slab grouped by brightness: 4,096 up / 4,096 down
inline long rsh(long x, long s) { if (s <= 0) return x << (-s); return (x + (1L << (s - 1))) >> s; }
inline long log2fix(ulong v, device const long* LOG2) {
    int t = 63 - clz(v); ulong m = t >= 16 ? (v >> (t - 16)) & 0xFFFF : (v << (16 - t)) & 0xFFFF; return (long)t * 65536 + LOG2[m]; }
inline long power(long x, long F, device const long* EXP2) { long I = F >> 16, f = F & 0xFFFF; return rsh(x * EXP2[f], 16 - I); }
inline long shfl_down(long v, ushort o) { uint lo = simd_shuffle_down((uint)(ulong)v, o), hi = simd_shuffle_down((uint)((ulong)v >> 32), o); return (long)(((ulong)hi << 32) | lo); }
inline long bcast0(long v) { uint lo = simd_broadcast((uint)(ulong)v, 0), hi = simd_broadcast((uint)((ulong)v >> 32), 0); return (long)(((ulong)hi << 32) | lo); }
inline long sum64(long v) { for (ushort o = 16; o > 0; o >>= 1) v += shfl_down(v, o); return bcast0(v); }
inline ulong or64(ulong v) { uint hi = simd_or((uint)(v >> 32)), lo = simd_or((uint)v); return ((ulong)hi << 32) | lo; }
inline long epsterm(long n, int e) { int s = 40 + 2 * e; return s >= 0 ? ((EPS40 * n) >> s) : ((EPS40 * n) << -s); }

kernel void to16(device const long* X [[buffer(0)]], device int* x16 [[buffer(1)]], device int* ex [[buffer(2)]],
                 constant uint& n [[buffer(3)]], constant uint& r0 [[buffer(4)]], constant int& gsh [[buffer(5)]],
                 uint t [[thread_position_in_threadgroup]], uint g [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup ulong ors[8]; uint p = r0 + g; device const long* x = X + p * n;
    ulong o = 0; for (uint i = t; i < n; i += 256) { long v = x[i]; o |= (ulong)(v < 0 ? -v : v); }
    o = or64(o); if (lane == 0) ors[sg] = o; threadgroup_barrier(mem_flags::mem_threadgroup);
    ulong all = 0; for (int k = 0; k < 8; k++) all |= ors[k];
    int top = all ? 64 - clz(all) : 0; int sh = max(top - 15, 0);
    for (uint i = t; i < n; i += 256) { int v = (int)rsh(x[i], sh); if (gsh > 0) { int hb = 1 << (gsh - 1); v = v < 0 ? -(((-v + hb) >> gsh) << gsh) : (((v + hb) >> gsh) << gsh); } x16[p * n + i] = v; }
    if (t == 0) ex[p] = -16 + sh; }
kernel void keeptop(device int* x16 [[buffer(0)]], constant uint& n [[buffer(1)]], constant uint& r0 [[buffer(2)]], constant uint& nr [[buffer(3)]],
                    constant int& keep [[buffer(4)]], uint k [[thread_position_in_grid]]) {
    // the light into the top read at `keep` binary digits (16 = all): the bottom digits rounded away
    if (k >= n * nr || keep >= 16) return; uint i = r0 * n + k; int s = 16 - keep; x16[i] = (int)(((long)x16[i] + (1L << (s - 1))) >> s) << s; }
kernel void gate1(device const short* W [[buffer(0)]], device const int* x16 [[buffer(1)]], device long* part [[buffer(2)]],
                  constant uint& nin [[buffer(3)]], constant uint& nout [[buffer(4)]], constant uint& p [[buffer(5)]], constant uint& span [[buffer(6)]],
                  uint2 id [[thread_position_in_grid]]) {
    // every meeting's photons land by direction: an up pile and a down pile, both only adding
    uint j = id.x; if (j >= nout) return; uint i0 = id.y * span; device const int* x = x16 + p * nin; int ulo = 0, uhi = 0, dlo = 0, dhi = 0;
    for (uint i = i0; i < i0 + span; i++) { int pr = (int)W[i * nout + j] * x[i];
        if (pr >= 0) { ulo += pr & 0x7FFF; uhi += pr >> 15; } else { int q = -pr; dlo += q & 0x7FFF; dhi += q >> 15; } }
    part[id.y * nout + j] = ((long)uhi << 15) + ulo; part[PARTHALF + id.y * nout + j] = ((long)dhi << 15) + dlo; }
kernel void addparts1(device const long* part [[buffer(0)]], device long* pile [[buffer(1)]], constant uint& nout [[buffer(2)]], constant uint& p [[buffer(3)]],
                      constant uint& ks [[buffer(4)]], uint j [[thread_position_in_grid]]) {
    if (j >= nout) return; long u = 0, d = 0; for (uint k = 0; k < ks; k++) { u += part[k * nout + j]; d += part[PARTHALF + k * nout + j]; }
    pile[p * nout + j] = u; pile[PILEHALF + p * nout + j] = d; }
kernel void gateN(device const short* W [[buffer(0)]], device const int* x16 [[buffer(1)]], device long* part [[buffer(2)]],
                  constant uint& nin [[buffer(3)]], constant uint& nout [[buffer(4)]], constant uint& r0 [[buffer(5)]], constant uint& nr [[buffer(6)]],
                  constant uint& span [[buffer(7)]], uint3 id [[thread_position_in_grid]]) {
    uint j = id.x; if (j >= nout) return; uint i0 = id.y * span, rb = r0 + id.z * 8, rn = min(8u, r0 + nr - rb), nrp = (nr + 7) / 8 * 8;
    int ulo[8], uhi[8], dlo[8], dhi[8]; for (int r = 0; r < 8; r++) { ulo[r] = 0; uhi[r] = 0; dlo[r] = 0; dhi[r] = 0; }
    device const int* x = x16 + rb * nin;
    for (uint i = i0; i < i0 + span; i++) { int w = W[i * nout + j];
        #pragma unroll
        for (int r = 0; r < 8; r++) { int pr = w * x[r * nin + i];
            if (pr >= 0) { ulo[r] += pr & 0x7FFF; uhi[r] += pr >> 15; } else { int q = -pr; dlo[r] += q & 0x7FFF; dhi[r] += q >> 15; } } }
    for (uint r = 0; r < rn; r++) { ulong o = (id.y * nrp + rb - r0 + r) * nout + j;
        part[o] = ((long)uhi[r] << 15) + ulo[r]; part[PARTHALF + o] = ((long)dhi[r] << 15) + dlo[r]; } }
kernel void addpartsN(device const long* part [[buffer(0)]], device long* pile [[buffer(1)]], constant uint& nout [[buffer(2)]], constant uint& r0 [[buffer(3)]],
                      constant uint& ks [[buffer(4)]], constant uint& n [[buffer(5)]], uint t [[thread_position_in_grid]]) {
    if (t >= n) return; uint q = t / nout, p = r0 + q, j = t % nout, nrp = (n / nout + 7) / 8 * 8; long s = 0;
    long d = 0; for (uint k = 0; k < ks; k++) { s += part[(k * nrp + q) * nout + j]; d += part[PARTHALF + (k * nrp + q) * nout + j]; }
    pile[p * nout + j] = s; pile[PILEHALF + p * nout + j] = d; }
kernel void gauge_rms(device const int* x16 [[buffer(0)]], device const int* ex [[buffer(1)]], device long* LS [[buffer(2)]],
                      device const long* LOG2 [[buffer(3)]], constant uint& r0 [[buffer(4)]], uint lane [[thread_index_in_simdgroup]], uint g [[threadgroup_position_in_grid]]) {
    // log2 of the root mean square over 1024: half the log of the sum of squares, minus half of log2(1024)
    uint p = r0 + g; long s2 = 0;
    for (uint i = lane; i < 1024; i += 32) { long v = x16[p * 1024 + i]; s2 += v * v; }
    s2 = sum64(s2);
    if (lane == 0) LS[p] = (log2fix((ulong)(s2 + epsterm(1024, ex[p])), LOG2) >> 1) - 5 * 65536 + (long)ex[p] * 65536; }
kernel void gauged(device const long* pile [[buffer(0)]], device const int* we [[buffer(1)]], device const int* ex [[buffer(2)]], device const long* LS [[buffer(3)]],
                   device long* out [[buffer(4)]], device const long* EXP2 [[buffer(5)]], constant uint& nout [[buffer(6)]], constant uint& r0 [[buffer(7)]],
                   constant uint& n [[buffer(8)]], uint k [[thread_position_in_grid]]) {
    if (k >= n) return; uint i = r0 * nout + k, p = i / nout, c = i % nout;
    out[i] = power(pile[i] - pile[PILEHALF + i], ((long)we[c] + ex[p] + 16) * 65536 - LS[p], EXP2); }   // the edge takes the excess away
kernel void plain(device const long* pile [[buffer(0)]], device const int* we [[buffer(1)]], device const int* ex [[buffer(2)]],
                  device long* R [[buffer(3)]], constant uint& nout [[buffer(4)]], constant uint& r0 [[buffer(5)]],
                  constant uint& n [[buffer(6)]], uint k [[thread_position_in_grid]]) {
    if (k >= n) return; uint i = r0 * nout + k, p = i / nout, c = i % nout;
    R[i] += rsh(pile[i] - pile[PILEHALF + i], -((long)we[c] + ex[p] + 16)); }   // the edge takes the excess away
kernel void heads(device const long* QKV [[buffer(0)]], device int* KQ16 [[buffer(1)]], device int* eKQ [[buffer(2)]],
                  device const long* QG [[buffer(3)]], device const long* KG [[buffer(4)]], device const long* COS [[buffer(5)]], device const long* SIN [[buffer(6)]],
                  device const long* LOG2 [[buffer(7)]], device const long* EXP2 [[buffer(8)]], constant uint& r0 [[buffer(9)]],
                  constant int& qge [[buffer(10)]], constant int& kge [[buffer(11)]],
                  uint lane [[thread_index_in_simdgroup]], uint g [[threadgroup_position_in_grid]]) {
    // one simdgroup per (position, head 0..23: 16 query heads then 8 key heads): the head's gauge and gain, the turn, then 16 bits
    uint p = r0 + g / 24, h = g % 24; device const long* x = QKV + p * 4096 + h * 128;    // key heads start at 2048 = 16 * 128
    device const long* G = h < 16 ? QG : KG; int ge = h < 16 ? qge : kge;
    uint ix[4] = {lane, lane + 32, lane + 64, lane + 96}; long v[4];
    ulong o = 0; for (int k = 0; k < 4; k++) { v[k] = x[ix[k]]; o |= (ulong)(v[k] < 0 ? -v[k] : v[k]); }
    o = or64(o); int top = o ? 64 - clz(o) : 0; int sh = max(top - 15, 0); int e = -16 + sh;
    long s2 = 0; long x16[4]; for (int k = 0; k < 4; k++) { x16[k] = rsh(v[k], sh); s2 += x16[k] * x16[k]; }
    s2 = sum64(s2);
    long LSh = (log2fix((ulong)(s2 + epsterm(128, e)), LOG2) >> 1) - 229376 + (long)e * 65536;     // half of log2(128) = 3.5 = 229376/65536
    long n[4]; for (int k = 0; k < 4; k++) n[k] = power(x16[k] * G[ix[k]], ((long)ge + e + 16) * 65536 - LSh, EXP2);
    // the turn: pair (i, i+64), i = lane and lane + 32
    long t[4];
    for (int k = 0; k < 2; k++) { long c = COS[p * 64 + ix[k]], s = SIN[p * 64 + ix[k]]; long a = n[k], b = n[k + 2];
        t[k] = rsh(a * c - b * s, 30); t[k + 2] = rsh(b * c + a * s, 30); }
    ulong o2 = 0; for (int k = 0; k < 4; k++) o2 |= (ulong)(t[k] < 0 ? -t[k] : t[k]);
    o2 = or64(o2); int top2 = o2 ? 64 - clz(o2) : 0; int sh2 = max(top2 - 15, 0);
    for (int k = 0; k < 4; k++) { int q = (int)rsh(t[k], sh2); if (h >= 16 && KV14) q = q < 0 ? -(((-q + 1) >> 1) << 1) : (((q + 1) >> 1) << 1);   // a key: one of 16,384 up / 16,384 down
        KQ16[(p * 24 + h) * 128 + ix[k]] = q; }
    if (lane == 0) eKQ[p * 24 + h] = -16 + sh2; }
kernel void vgroup(device long* QKV [[buffer(0)]], constant uint& r0 [[buffer(1)]], uint lane [[thread_index_in_simdgroup]], uint g [[threadgroup_position_in_grid]]) {
    // a piece's values, per key/value head: one shared power of two, each value one of 16,384 up / 16,384 down (in place)
    if (!KV14) return; uint p = r0 + g / 8, j = g % 8; device long* v = QKV + p * 4096 + 3072 + j * 128;
    ulong o = 0; for (int k = 0; k < 4; k++) { long x = v[lane + 32 * k]; o |= (ulong)(x < 0 ? -x : x); }
    o = or64(o); int top = o ? 64 - clz(o) : 0; int sh = max(top - 14, 0);
    for (int k = 0; k < 4; k++) v[lane + 32 * k] = rsh(v[lane + 32 * k], sh) << sh; }
kernel void attend(device const long* QKV [[buffer(0)]], device const int* KQ16 [[buffer(1)]], device const int* eKQ [[buffer(2)]], device long* out [[buffer(3)]],
                   device const long* ARR [[buffer(4)]], device const long* LOG2 [[buffer(5)]], device const long* EXP2 [[buffer(6)]], constant uint& r0 [[buffer(7)]],
                   uint lane [[thread_index_in_simdgroup]], uint g [[threadgroup_position_in_grid]]) {
    uint p = r0 + g / 16, h = g % 16, kh = 16 + h / 2; long sc[64];
    for (uint s = 0; s <= p; s++) {
        long dot = 0; for (int k = 0; k < 4; k++) dot += (long)KQ16[(p * 24 + h) * 128 + lane + 32 * k] * KQ16[(s * 24 + kh) * 128 + lane + 32 * k];
        sc[s] = rsh(sum64(dot), -((long)eKQ[p * 24 + h] + eKQ[s * 24 + kh] + 10)); }
    uint best = 0; for (uint s = 1; s <= p; s++) if (sc[s] > sc[best]) best = s;
    long start = sc[best], num[4] = {0, 0, 0, 0}, Z = 0;
    for (uint s = 0; s <= p; s++) { long dist = start - sc[s]; long a = dist < ALEN ? ARR[dist] : 0; if (a == 0) continue; Z += a;
        for (int k = 0; k < 4; k++) num[k] += a * QKV[s * 4096 + 3072 + (h / 2) * 128 + lane + 32 * k]; }
    long F = -log2fix((ulong)Z, LOG2); long I = F >> 16, f = F & 0xFFFF;
    for (int k = 0; k < 4; k++) out[p * 2048 + h * 128 + lane + 32 * k] = rsh(num[k] * EXP2[f], 16 - I); }
kernel void bend(device const long* GU [[buffer(0)]], device long* H [[buffer(1)]], device const long* SIG [[buffer(2)]],
                 constant uint& r0 [[buffer(3)]], constant uint& n [[buffer(4)]], uint k [[thread_position_in_grid]]) {
    // the gated bend: g * share(g) * u
    if (k >= n) return; uint i = r0 * 3072 + k, p = i / 3072, j = i % 3072;
    long g = GU[p * 6144 + j], u = GU[p * 6144 + 3072 + j]; long idx = rsh(g, 4) + 65536;
    long share = idx < 0 ? 0 : (idx > 131071 ? 65536 : SIG[idx]);
    H[i] = rsh(rsh(g * share, 16) * u, 16); }
kernel void race(device const long* lg [[buffer(0)]], device int* word [[buffer(1)]], constant uint& VV [[buffer(2)]], constant uint& r0 [[buffer(3)]],
                 uint t [[thread_position_in_threadgroup]], uint g [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup long bv[32]; threadgroup int bi[32]; uint p = r0 + g; device const long* v = lg + p * VV;
    long b = LONG_MIN; int bw = 0;
    for (uint i = t; i < VV; i += 1024) if (v[i] > b) { b = v[i]; bw = (int)i; }
    for (ushort o = 16; o > 0; o >>= 1) { long ob = shfl_down(b, o); int ow = simd_shuffle_down(bw, o); if (ob > b || (ob == b && ow < bw)) { b = ob; bw = ow; } }
    if (lane == 0) { bv[sg] = b; bi[sg] = bw; } threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) { long B = bv[0]; int W = bi[0]; for (int k = 1; k < 32; k++) if (bv[k] > B || (bv[k] == B && bi[k] < W)) { B = bv[k]; W = bi[k]; } word[p] = W; } }
kernel void zero(device long* a [[buffer(0)]], constant uint& off [[buffer(1)]], constant uint& n [[buffer(2)]], uint k [[thread_position_in_grid]]) { if (k < n) a[off + k] = 0; }
kernel void pour(device const long* nw [[buffer(0)]], device long* last [[buffer(1)]], device long* piles [[buffer(2)]], constant uint& j [[buffer(3)]],
                 constant uint& off [[buffer(4)]], constant uint& n [[buffer(5)]], constant uint& stride [[buffer(6)]], uint k [[thread_position_in_grid]]) {
    // block j's change since it last poured lands on every pile above it; no change, nothing lands
    if (k >= n) return; uint i = off + k; long d = nw[i] - last[i]; if (d == 0) return; last[i] = nw[i];
    for (uint q = j + 1; q <= 56; q++) piles[q * stride + i] += d; }
kernel void copyl(device const long* a [[buffer(0)]], device long* b [[buffer(1)]], constant uint& off [[buffer(2)]], constant uint& n [[buffer(3)]], uint k [[thread_position_in_grid]]) {
    if (k < n) b[off + k] = a[off + k]; }
kernel void nextin(device const int* word [[buffer(0)]], device const int* EMB [[buffer(1)]], device long* R0 [[buffer(2)]], constant uint& p [[buffer(3)]],
                   uint i [[thread_position_in_grid]]) {
    // the word the race just gave at p goes straight in at p+1: nothing leaves the GPU
    if (i >= 1024) return; R0[(p + 1) * 1024 + i] = EMB[(long)word[p] * 1024 + i]; }
"""
setvbuf(stdout, nil, _IONBF, 0)
let dev = MTLCreateSystemDefaultDevice()!, queue = dev.makeCommandQueue()!
let lib = try! dev.makeLibrary(source: src, options: MTLCompileOptions())
func pipe(_ n: String) -> MTLComputePipelineState { return try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
let pTo16 = pipe("to16"), pGate1 = pipe("gate1"), pAdd1 = pipe("addparts1"), pGateN = pipe("gateN"), pAddN = pipe("addpartsN"), pGauge = pipe("gauge_rms")
let pGauged = pipe("gauged"), pPlain = pipe("plain"), pHeads = pipe("heads"), pAttend = pipe("attend"), pBend = pipe("bend"), pRace = pipe("race")
let pZero = pipe("zero"), pPour = pipe("pour"), pVgroup = pipe("vgroup"), pCopy = pipe("copyl"), pNext = pipe("nextin"), pKeep = pipe("keeptop")
let HEADDIGITS = argInt("headdigits", 16)
func loadData(_ name: String) -> Data { return try! Data(contentsOf: URL(fileURLWithPath: dir + name), options: .alwaysMapped) }
func bufData(_ d: Data) -> MTLBuffer { return d.withUnsafeBytes { dev.makeBuffer(bytes: $0.baseAddress!, length: d.count, options: .storageModeShared)! } }
func zbuf(_ bytes: Int) -> MTLBuffer { return dev.makeBuffer(length: max(bytes, 8), options: .storageModeShared)! }
struct Bank { let nin, nout: Int; let w, e: MTLBuffer }
var banks = [String: Bank]()
for b in man["banks"] as! [[String: Any]] { let n = b["name"] as! String
    banks[n] = Bank(nin: b["nin"] as! Int, nout: b["nout"] as! Int, w: bufData(loadData("\(n).wc")), e: bufData(loadData("\(n).e"))) }
let QG = (0..<L).map { bufData(loadData("qg.\($0).bin")) }, KG = (0..<L).map { bufData(loadData("kg.\($0).bin")) }
let COS = bufData(loadData("cos.bin")), SIN = bufData(loadData("sin.bin")), SIG = bufData(loadData("sig.bin")), ARR = bufData(loadData("arrival.bin"))
let LOG2 = bufData(loadData("log2.bin")), EXP2 = bufData(loadData("exp2.bin")), EMB = bufData(loadData("embed.bin"))
let ref = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: ARG["refs"] ?? "refs_qwen.json"))) as! [[String: Any]]

let R = (0...L).map { _ in zbuf(PMAX * D * 8) }
let qkvL = (0..<L).map { _ in zbuf(PMAX * 4096 * 8) }, kqL = (0..<L).map { _ in zbuf(PMAX * 24 * 128 * 4) }, ekqL = (0..<L).map { _ in zbuf(PMAX * 24 * 4) }
let parts = zbuf(2 * PARTHALF * 8), x16 = zbuf((PMAX + 8) * 3072 * 4), ex = zbuf(PMAX * 4), pile = zbuf(2 * PILEHALF * 8), LS = zbuf(PMAX * 8)
let att = zbuf(PMAX * 2048 * 8), gu = zbuf(PMAX * 6144 * 8), hid = zbuf(PMAX * 3072 * 8), lg = zbuf(PMAX * V * 8), words = zbuf(PMAX * 4)
// the light entering every slab grouped by brightness; lv=bottom,top,top read = groups per direction (bottom = lower half of the depths)
var CURDEPTH = 0
let NDEPTH = 28
let LVS: [Int] = (ARG["lv"] ?? (ARG["groups"] != nil ? "4096,4096,4096" : "0,0,0")).split(separator: ",").map { Int($0)! }
func shiftFor(_ levels: Int) -> Int { if levels <= 0 { return 0 }; var s = 0; while (32768 >> s) > levels { s += 1 }; return s }
func groupShift(_ bk: Bank) -> Int { return bk.nout > 50000 ? shiftFor(LVS[2]) : shiftFor(CURDEPTH < NDEPTH / 2 ? LVS[0] : LVS[1]) }
func enc(_ cb: MTLCommandBuffer, _ ps: MTLComputePipelineState, _ bufs: [MTLBuffer], _ consts: [Int32], threads: Int, tg: Int, groups: Int = 0, height: Int = 1, depth: Int = 1, offs: [Int] = []) {
    let e = cb.makeComputeCommandEncoder()!; e.setComputePipelineState(ps)
    for (i, b) in bufs.enumerated() { e.setBuffer(b, offset: i < offs.count ? offs[i] : 0, index: i) }
    for (k, c) in consts.enumerated() { var v = c; e.setBytes(&v, length: 4, index: bufs.count + k) }
    if groups > 0 { e.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)) }
    else { e.dispatchThreads(MTLSize(width: threads, height: height, depth: depth), threadsPerThreadgroup: MTLSize(width: min(tg, threads), height: 1, depth: 1)) }
    e.endEncoding()
}
func product(_ cb: MTLCommandBuffer, _ X: MTLBuffer, _ bk: Bank, r0: Int, nr: Int, xoff: Int = 0) {      // positions onto 16 bits, then every core takes piles x a slice of inputs
    enc(cb, pTo16, [X, x16, ex], [Int32(bk.nin), Int32(r0), Int32(groupShift(bk))], threads: 0, tg: 256, groups: nr, offs: [xoff])
    if nr == 1 { let span = 64, ks = bk.nin / span
        enc(cb, pGate1, [bk.w, x16, parts], [Int32(bk.nin), Int32(bk.nout), Int32(r0), Int32(span)], threads: bk.nout, tg: 256, height: ks)
        enc(cb, pAdd1, [parts, pile], [Int32(bk.nout), Int32(r0), Int32(ks)], threads: bk.nout, tg: 256); return }
    let span = 128, ks = bk.nin / span                                                    // several positions: 8 at a time share each weight read
    enc(cb, pGateN, [bk.w, x16, parts], [Int32(bk.nin), Int32(bk.nout), Int32(r0), Int32(nr), Int32(span)], threads: bk.nout, tg: 256, height: ks, depth: (nr + 7) / 8)
    enc(cb, pAddN, [parts, pile], [Int32(bk.nout), Int32(r0), Int32(ks), Int32(nr * bk.nout)], threads: nr * bk.nout, tg: 256)
}
func productNoTo16(_ cb: MTLCommandBuffer, _ bk: Bank, r0: Int, nr: Int) {            // the light is already on 16 bits in x16
    if nr == 1 { let span = 64, ks = bk.nin / span
        enc(cb, pGate1, [bk.w, x16, parts], [Int32(bk.nin), Int32(bk.nout), Int32(r0), Int32(span)], threads: bk.nout, tg: 256, height: ks)
        enc(cb, pAdd1, [parts, pile], [Int32(bk.nout), Int32(r0), Int32(ks)], threads: bk.nout, tg: 256); return }
    let span = 128, ks = bk.nin / span
    enc(cb, pGateN, [bk.w, x16, parts], [Int32(bk.nin), Int32(bk.nout), Int32(r0), Int32(nr), Int32(span)], threads: bk.nout, tg: 256, height: ks, depth: (nr + 7) / 8)
    enc(cb, pAddN, [parts, pile], [Int32(bk.nout), Int32(r0), Int32(ks), Int32(nr * bk.nout)], threads: nr * bk.nout, tg: 256)
}
func layer(_ cb: MTLCommandBuffer, _ d: Int, r0: Int, nr: Int) {
    CURDEPTH = d
    let bq = banks["qkv.\(d)"]!, bo = banks["o.\(d)"]!, bg = banks["gu.\(d)"]!, bd = banks["down.\(d)"]!; let r = Int32(r0)
    product(cb, R[d], bq, r0: r0, nr: nr); enc(cb, pGauge, [x16, ex, LS, LOG2], [r], threads: 0, tg: 32, groups: nr)
    enc(cb, pGauged, [pile, bq.e, ex, LS, qkvL[d], EXP2], [4096, r, Int32(nr * 4096)], threads: nr * 4096, tg: 256)
    enc(cb, pVgroup, [qkvL[d]], [r], threads: 0, tg: 32, groups: nr * 8)
    enc(cb, pHeads, [qkvL[d], kqL[d], ekqL[d], QG[d], KG[d], COS, SIN, LOG2, EXP2], [r, Int32(GEXP[2 * d]), Int32(GEXP[2 * d + 1])], threads: 0, tg: 32, groups: nr * 24)
    enc(cb, pAttend, [qkvL[d], kqL[d], ekqL[d], att, ARR, LOG2, EXP2], [r], threads: 0, tg: 32, groups: nr * 16)
    enc(cb, pCopy, [R[d], R[d + 1]], [Int32(r0 * D), Int32(nr * D)], threads: nr * D, tg: 256)
    product(cb, att, bo, r0: r0, nr: nr)
    enc(cb, pPlain, [pile, bo.e, ex, R[d + 1]], [Int32(D), r, Int32(nr * D)], threads: nr * D, tg: 256)
    product(cb, R[d + 1], bg, r0: r0, nr: nr); enc(cb, pGauge, [x16, ex, LS, LOG2], [r], threads: 0, tg: 32, groups: nr)
    enc(cb, pGauged, [pile, bg.e, ex, LS, gu, EXP2], [6144, r, Int32(nr * 6144)], threads: nr * 6144, tg: 256)
    enc(cb, pBend, [gu, hid, SIG], [r, Int32(nr * 3072)], threads: nr * 3072, tg: 256)
    product(cb, hid, bd, r0: r0, nr: nr)
    enc(cb, pPlain, [pile, bd.e, ex, R[d + 1]], [Int32(D), r, Int32(nr * D)], threads: nr * D, tg: 256)
}
func readTop(_ cb: MTLCommandBuffer, r0: Int, nr: Int = 1) {
    let bh = banks["head"]!
    enc(cb, pTo16, [R[L], x16, ex], [Int32(bh.nin), Int32(r0), Int32(groupShift(bh))], threads: 0, tg: 256, groups: nr)
    enc(cb, pGauge, [x16, ex, LS, LOG2], [Int32(r0)], threads: 0, tg: 32, groups: nr)          // the gauge from the full light
    enc(cb, pKeep, [x16], [Int32(bh.nin), Int32(r0), Int32(nr), Int32(HEADDIGITS)], threads: nr * bh.nin, tg: 256)
    productNoTo16(cb, bh, r0: r0, nr: nr)
    enc(cb, pGauged, [pile, bh.e, ex, LS, lg, EXP2], [Int32(V), Int32(r0), Int32(nr * V)], threads: nr * V, tg: 256)
    enc(cb, pRace, [lg, words], [Int32(V), Int32(r0)], threads: 0, tg: 1024, groups: nr)
}
// QUICK GUESSES, STACKED: the next pieces guessed from the text itself (where the latest pieces appeared before, what
// followed then), all sent through on the same trip of the hot weights, checked by the race; always exact.
func guess(_ seq: [Int], _ k: Int) -> [Int] {
    for n in [3, 2, 1] where seq.count > n {
        let tail = Array(seq.suffix(n))
        var i = seq.count - n - 1
        while i >= 0 { if Array(seq[i..<(i + n)]) == tail { return Array(seq[(i + n)..<min(seq.count, i + n + k)]) }; i -= 1 }
    }
    return []
}
// The base file: S, the tokens, the first word; then per depth, per position, per key/value head:
// 128 keys (16 bits) + their power of two, 128 values (16 bits: one of 16,384 up / 16,384 down) + their power of two.
var loadTime = 0.0, tLoadAll = 0.0
func saveBase(_ path: String, _ ids: [Int]) {
    let S = ids.count; var out = Data()
    func i32(_ v: Int) { var x = Int32(v); withUnsafeBytes(of: &x) { out.append(contentsOf: $0) } }
    func i16(_ v: Int) { var x = Int16(v); withUnsafeBytes(of: &x) { out.append(contentsOf: $0) } }
    i32(S); ids.forEach { i32($0) }; i32(Int(words.contents().bindMemory(to: Int32.self, capacity: PMAX)[S - 1]))
    for d in 0..<L {
        let kq = kqL[d].contents().bindMemory(to: Int32.self, capacity: PMAX * 24 * 128), ek = ekqL[d].contents().bindMemory(to: Int32.self, capacity: PMAX * 24)
        let qkv = qkvL[d].contents().bindMemory(to: Int64.self, capacity: PMAX * 4096)
        for p in 0..<S { for j in 0..<8 {
            for i in 0..<128 { i16(Int(kq[(p * 24 + 16 + j) * 128 + i]) >> 1) }          // a key is even (one of 16,384 up / down): its half fits 16 bits; out.append(UInt8(bitPattern: Int8(ek[p * 24 + 16 + j])))
            var mx: Int64 = 0; for i in 0..<128 { mx = max(mx, abs(qkv[p * 4096 + 3072 + j * 128 + i])) }
            var sh = max((mx == 0 ? 0 : 64 - mx.leadingZeroBitCount) - 14, 0)
            while sh > 0 && (0..<128).contains(where: { qkv[p * 4096 + 3072 + j * 128 + $0] & ((Int64(1) << Int64(sh)) - 1) != 0 }) { sh -= 1 }   // the shared power the values really have
            for i in 0..<128 { let v = qkv[p * 4096 + 3072 + j * 128 + i]; precondition(v == (v >> Int64(sh)) << Int64(sh)); i16(Int(v >> Int64(sh))) }
            out.append(UInt8(sh)) } }
    }
    try! out.write(to: URL(fileURLWithPath: path))
}
func loadBase(_ path: String, _ ids: [Int]) {
    FileHandle.standardError.write("load: reading \(path)\n".data(using: .utf8)!)
    let t = Date(); let d0 = try! Data(contentsOf: URL(fileURLWithPath: path)); var at = 0
    FileHandle.standardError.write("load: \(d0.count) bytes\n".data(using: .utf8)!)
    func i32() -> Int { let v = d0.subdata(in: at..<(at + 4)).withUnsafeBytes { $0.loadUnaligned(as: Int32.self) }; at += 4; return Int(v) }
    func i16() -> Int { let v = d0.subdata(in: at..<(at + 2)).withUnsafeBytes { $0.loadUnaligned(as: Int16.self) }; at += 2; return Int(v) }
    func u8() -> UInt8 { let v = d0[at]; at += 1; return v }
    let S = i32(); FileHandle.standardError.write("load: S=\(S)\n".data(using: .utf8)!); let toks = (0..<S).map { _ in i32() }; precondition(toks == ids, "this base was saved for another prompt")
    words.contents().bindMemory(to: Int32.self, capacity: PMAX)[S - 1] = Int32(i32())
    for d in 0..<L {
        let kq = kqL[d].contents().bindMemory(to: Int32.self, capacity: PMAX * 24 * 128), ek = ekqL[d].contents().bindMemory(to: Int32.self, capacity: PMAX * 24)
        let qkv = qkvL[d].contents().bindMemory(to: Int64.self, capacity: PMAX * 4096)
        for p in 0..<S { for j in 0..<8 {
            for i in 0..<128 { kq[(p * 24 + 16 + j) * 128 + i] = Int32(i16()) << 1 }; ek[p * 24 + 16 + j] = Int32(Int8(bitPattern: u8()))
            var vals = [Int](); for _ in 0..<128 { vals.append(i16()) }; let sh = Int(u8())
            for i in 0..<128 { qkv[p * 4096 + 3072 + j * 128 + i] = Int64(vals[i]) << Int64(sh) } } }
    }
    loadTime = Date().timeIntervalSince(t)
}

// ALL AT ONCE: pile k is block k's light (block 2d = depth d's arrival, 2d+1 its gated bend; pile 56 the top). Every pile =
// the piece's own row + everything the blocks below it have poured. A block fires on whatever its pile holds now and
// pours only its change upward; its edges re-land on every firing. together: every due block fires at once, all pour together.
let STR = PMAX * D, piles = zbuf(57 * STR * 8), newB = (0..<56).map { _ in zbuf(STR * 8) }, lastB = (0..<56).map { _ in zbuf(STR * 8) }
func fire(_ cb: MTLCommandBuffer, _ j: Int, r0: Int, nr: Int) {
    let d = j / 2; CURDEPTH = d; let r = Int32(r0), xo = j * STR * 8, out = newB[j]
    enc(cb, pZero, [out], [Int32(r0 * D), Int32(nr * D)], threads: nr * D, tg: 256)
    if j % 2 == 0 {
        let bq = banks["qkv.\(d)"]!, bo = banks["o.\(d)"]!
        product(cb, piles, bq, r0: r0, nr: nr, xoff: xo); enc(cb, pGauge, [x16, ex, LS, LOG2], [r], threads: 0, tg: 32, groups: nr)
        enc(cb, pGauged, [pile, bq.e, ex, LS, qkvL[d], EXP2], [4096, r, Int32(nr * 4096)], threads: nr * 4096, tg: 256)
        enc(cb, pVgroup, [qkvL[d]], [r], threads: 0, tg: 32, groups: nr * 8)
        enc(cb, pHeads, [qkvL[d], kqL[d], ekqL[d], QG[d], KG[d], COS, SIN, LOG2, EXP2], [r, Int32(GEXP[2 * d]), Int32(GEXP[2 * d + 1])], threads: 0, tg: 32, groups: nr * 24)
        enc(cb, pAttend, [qkvL[d], kqL[d], ekqL[d], att, ARR, LOG2, EXP2], [r], threads: 0, tg: 32, groups: nr * 16)
        product(cb, att, bo, r0: r0, nr: nr)
        enc(cb, pPlain, [pile, bo.e, ex, out], [Int32(D), r, Int32(nr * D)], threads: nr * D, tg: 256)
    } else {
        let bg = banks["gu.\(d)"]!, bd = banks["down.\(d)"]!
        product(cb, piles, bg, r0: r0, nr: nr, xoff: xo); enc(cb, pGauge, [x16, ex, LS, LOG2], [r], threads: 0, tg: 32, groups: nr)
        enc(cb, pGauged, [pile, bg.e, ex, LS, gu, EXP2], [6144, r, Int32(nr * 6144)], threads: nr * 6144, tg: 256)
        enc(cb, pBend, [gu, hid, SIG], [r, Int32(nr * 3072)], threads: nr * 3072, tg: 256)
        product(cb, hid, bd, r0: r0, nr: nr)
        enc(cb, pPlain, [pile, bd.e, ex, out], [Int32(D), r, Int32(nr * D)], threads: nr * D, tg: 256)
    }
}
func pourOut(_ cb: MTLCommandBuffer, _ j: Int, r0: Int, nr: Int) {
    enc(cb, pPour, [newB[j], lastB[j], piles], [Int32(j), Int32(r0 * D), Int32(nr * D), Int32(STR)], threads: nr * D, tg: 256) }
func readTopAll(_ cb: MTLCommandBuffer, r0: Int) {
    let bh = banks["head"]!; CURDEPTH = 28
    product(cb, piles, bh, r0: r0, nr: 1, xoff: 56 * STR * 8); enc(cb, pGauge, [x16, ex, LS, LOG2], [Int32(r0)], threads: 0, tg: 32, groups: 1)
    enc(cb, pGauged, [pile, bh.e, ex, LS, lg, EXP2], [Int32(V), Int32(r0), Int32(V)], threads: V, tg: 256)
    enc(cb, pRace, [lg, words], [Int32(V), Int32(r0)], threads: 0, tg: 1024, groups: 1)
}
let ORDER = ARG["order"] ?? "together"
var fires = 0, turns = 0
func settle(_ cb: MTLCommandBuffer, r0: Int, nr: Int) {
    var due = Set(0..<56)
    while !due.isEmpty {
        turns += 1
        if ORDER == "together" {
            let now = due.sorted(); for j in now { fire(cb, j, r0: r0, nr: nr); fires += 1 }
            for j in now { pourOut(cb, j, r0: r0, nr: nr) }
            due = Set(now.flatMap { ($0 + 1)..<56 })
        } else { let j = due.min()!; fire(cb, j, r0: r0, nr: nr); pourOut(cb, j, r0: r0, nr: nr); fires += 1; due.remove(j); for k in (j + 1)..<56 { due.insert(k) } }
    }
}
let CASES = ARG["all"] != nil ? Array(0..<ref.count) : [CASE]
var nIdentical = 0, report = [String](), totalWords = 0, totalTime = 0.0
let embI = EMB.contents().bindMemory(to: Int32.self, capacity: V * D), pp = piles.contents().bindMemory(to: Int64.self, capacity: 57 * STR)
for CASE in CASES {
    let ids = ref[CASE]["ids"] as! [Int], prompt = ref[CASE]["prompt"] as! String
    let N = min(WORDS, (ref[CASE]["ref"] as! [Int]).count), want = Array((ref[CASE]["ref"] as! [Int]).prefix(N)), S = ids.count
    let t0 = Date(); var seq = ids, gen = [Int]()
    func put(_ q: Int, _ w: Int) { for i in 0..<D { let v = Int64(embI[w * D + i]); for k in 0...56 { pp[k * STR + q * D + i] = v } } }
    for q in 0..<S { put(q, ids[q]) }; for j in 0..<56 { memset(lastB[j].contents(), 0, STR * 8) }
    var cb = queue.makeCommandBuffer()!
    settle(cb, r0: 0, nr: S); readTopAll(cb, r0: S - 1); cb.commit(); cb.waitUntilCompleted()
    var w = Int(words.contents().bindMemory(to: Int32.self, capacity: PMAX)[S - 1]); gen.append(w); seq.append(w)
    for p in S..<(S + N - 1) {
        put(p, w); for j in 0..<56 { memset(lastB[j].contents() + p * D * 8, 0, D * 8) }
        cb = queue.makeCommandBuffer()!; settle(cb, r0: p, nr: 1); readTopAll(cb, r0: p); cb.commit(); cb.waitUntilCompleted()
        w = Int(words.contents().bindMemory(to: Int32.self, capacity: PMAX)[p]); gen.append(w); seq.append(w)
    }
    let wall = Date().timeIntervalSince(t0); totalWords += N; totalTime += wall
    if gen == want { nIdentical += 1 } else { let j = (0..<N).first { gen[$0] != want[$0] }!; report.append("  '\(prompt)': first differs at word \(j + 1)") }
}
print("ALL AT ONCE (\(ORDER)): \(nIdentical) of \(CASES.count) prompts identical to fp32 Qwen3-0.6B")
report.forEach { print($0) }
print(String(format: "%d words in %.1f s: %.1f words/s; %d block firings in %d turns (in steps it would be %d firings)", totalWords, totalTime, Double(totalWords) / totalTime, fires, turns, 56 * totalWords))
