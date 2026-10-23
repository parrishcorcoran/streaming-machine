// THE UP/DOWN STREAMING MACHINE: every bank's meeting written without zeros -- each value 16 places, every place a photon
// going up or a hole (B = v + 32768); a pile = (count over whole rows, the token stack, the neuron stack) / 4. Otherwise as below.
// THE STREAMING MACHINE, ALL BASE 2. GPT-2 (124M), every number a whole number in base 2, no float while it runs
// (SPEC.md). The prompt goes in all at once, every token and every layer together. Then each new token adds only itself: everything before it stands (each layer keeps its keys and values),
// and the new position flows through every layer, meeting every weight once, spread over all the GPU's cores.
//
//   products : the chip's AND-gate grid (its integer multiplier); each pile's products land in two narrow cells,
//              the bottom 15 digits in one and the rest in the other, read once at the end
//   the gauge: log2 = where the top 1 sits + a table read; a root is a halving, a divide a negation
//   the edge : where the value arrives picks its share from a table
//   the arrival: the clock starts at the race's winner; each score's distance from it reads the arrival table
//   the word : the race at the top
//
//   swiftc -O -o base2_stream base2_stream.swift && ./base2_stream all=1
import Foundation
import Metal

var ARG = [String: String]()
for a in CommandLine.arguments.dropFirst() { let p = a.split(separator: "=", maxSplits: 1).map(String.init); ARG[p[0]] = p.count > 1 ? p[1] : "1" }
func argInt(_ k: String, _ d: Int) -> Int { return Int(ARG[k] ?? "") ?? d }
let CASE = argInt("case", 0), WORDS = argInt("words", 16), MAXSTEPS = argInt("steps", 200), QUIET = ARG["quiet"] != nil
let dir = (ARG["dir"] ?? "base2_gpu") + "/"
let man = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: dir + "manifest.json"))) as! [String: Any]
let LOG768 = man["log2_768"] as! Int, EPS40 = man["eps40"] as! Int, ALEN = man["arrival_len"] as! Int

var ORDERL = Array(0..<12); if let sw = ARG["swap"] { let ab = sw.split(separator: ",").map { Int($0)! }; ORDERL.swapAt(ab[0], ab[1]) }
let LG = ARG["groups"] != nil
let INV = ARG["inv"] != nil, DEV = argInt("digits", 16), UP = ARG["fromBottom"] != nil
let ROT = ARG["seed"] != nil ? (argInt("seed", 0) * 7919 + 13) % 1000 : 0
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

kernel void packW(device const short* W [[buffer(0)]], device uint* WR [[buffer(1)]], constant uint& nin [[buffer(2)]], constant uint& nout [[buffer(3)]],
                  uint t [[thread_position_in_grid]]) {
    // a neuron's 16 place rows, each place up (1) or a hole (0) of B = w + 32768; 32 lines a word; once, at load
    uint words = nin / 32; if (t >= nout * words) return; uint j = t / words, k = t % words; uint row[16];
    for (int b = 0; b < 16; b++) row[b] = 0;
    for (uint i = 0; i < 32; i++) { uint B = (uint)((int)W[(k * 32 + i) * nout + j] + 32768); for (int b = 0; b < 16; b++) row[b] |= ((B >> b) & 1u) << i; }
    for (int b = 0; b < 16; b++) WR[(j * 16 + b) * words + k] = row[b]; }
kernel void rowsX(device const int* x16 [[buffer(0)]], device uint* XR [[buffer(1)]], constant uint& nin [[buffer(2)]], constant uint& r0 [[buffer(3)]],
                  constant uint& n [[buffer(4)]], uint t [[thread_position_in_grid]]) {
    // the token's 16 place rows at each position: B = x + 32768
    if (t >= n) return; uint words = nin / 32; uint p = r0 + t / words, k = t % words; uint row[16];
    for (int a = 0; a < 16; a++) row[a] = 0;
    device const int* x = x16 + p * nin + k * 32;
    for (uint i = 0; i < 32; i++) { uint B = (uint)(x[i] + 32768); for (int a = 0; a < 16; a++) row[a] |= ((B >> a) & 1u) << i; }
    for (int a = 0; a < 16; a++) XR[(p * 16 + a) * words + k] = row[a]; }
kernel void updown(device const uint* WR [[buffer(0)]], device const uint* XR [[buffer(1)]], device long* pile [[buffer(2)]],
                   device const long* sx [[buffer(3)]], device const long* sw [[buffer(4)]],
                   constant uint& nin [[buffer(5)]], constant uint& nout [[buffer(6)]], constant uint& r0 [[buffer(7)]], constant uint& nr [[buffer(8)]],
                   uint gid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                   uint sgs [[simdgroups_per_threadgroup]]) {
    // one neuron a simdgroup; each lane XORs its words of the neuron's rows against the token's rows and counts the disagreements,
    // for all 256 pairs of places, weighted by 2^(a+b); the lanes add; the pile from the count and the two stacks
    uint j = gid * sgs + sg; if (j >= nout) return;
    uint words = nin / 32; long acc[8]; for (int r = 0; r < 8; r++) acc[r] = 0;
    for (uint k = lane; k < words; k += 32) {
        uint w[16]; for (int b = 0; b < 16; b++) w[b] = WR[(j * 16 + b) * words + k];
        for (uint r = 0; r < nr; r++) {
            device const uint* xr = XR + ((r0 + r) * 16) * words + k; long s = 0;
            for (int a = 0; a < 16; a++) { uint xa = xr[a * words]; long sa = 0;
                for (int b = 0; b < 16; b++) sa += (long)popcount(xa ^ w[b]) << b;
                s += sa << a; }
            acc[r] += s; } }
    for (uint r = 0; r < nr; r++) {
        long Dw = sum64(acc[r]);
        if (lane == 0) { uint p = r0 + r;
            long S = (long)nin * 65535L * 65535L - 2 * Dw;                     // sum over place pairs of 2^(a+b) (lines - 2 x disagreements)
            pile[p * nout + j] = (S - (2 * sx[p] + (long)nin) - (2 * sw[j] + (long)nin) + (long)nin) / 4; } } }
"""
setvbuf(stdout, nil, _IONBF, 0)
let dev = MTLCreateSystemDefaultDevice()!, queue = dev.makeCommandQueue()!
let lib = try! dev.makeLibrary(source: src, options: MTLCompileOptions())
func pipe(_ n: String) -> MTLComputePipelineState { return try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
let pTo16 = pipe("to16"), pGate1 = pipe("gate1"), pAdd1 = pipe("addparts1"), pGateN = pipe("gateN"), pAddN = pipe("addpartsN"), pGaugeLog = pipe("gauge_log"), pGauged = pipe("gauged"), pPlain = pipe("plain"), pEdge = pipe("edge")
let pPackW = pipe("packW"), pRowsX = pipe("rowsX"), pUpDown = pipe("updown")
let pHeads = pipe("heads16"), pAttend = pipe("attend"), pRace = pipe("race"), pCopy = pipe("copyl"), pNext = pipe("nextin")
func loadData(_ name: String) -> Data { return try! Data(contentsOf: URL(fileURLWithPath: dir + name)) }
func bufData(_ d: Data) -> MTLBuffer { return d.withUnsafeBytes { dev.makeBuffer(bytes: $0.baseAddress!, length: d.count, options: .storageModeShared)! } }
func zbuf(_ bytes: Int) -> MTLBuffer { return dev.makeBuffer(length: max(bytes, 8), options: .storageModeShared)! }
func i64(_ name: String) -> [Int64] { let d = loadData(name); return d.withUnsafeBytes { Array($0.bindMemory(to: Int64.self)) } }
struct Bank { let nin, nout: Int; let w, e, b: MTLBuffer; var sw: MTLBuffer? = nil; var wr: MTLBuffer? = nil }
// THE RANDOM ABELIAN WIRING (seed=N): channels, neurons, heads, the dimensions inside each head, the words at the top,
// and the order each sum's slices are added, all shuffled. Every number is whole, so every pile is the same to the last digit.
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
for (n, nin, nout) in bankList {
    let w = arr("\(n).wc", Int16.self), e = arr("\(n).e", Int32.self), b = arr("\(n).b", Int64.self), rp = rowP[n]!, cp = colP[n]!
    var w2 = [Int16](repeating: 0, count: w.count)
    for i in 0..<nin { let ri = rp[i] * nout, wi = i * nout; for j in 0..<nout { w2[wi + j] = w[ri + cp[j]] } }
    if DEV < 16 { for k in 0..<w2.count { let v = Int32(w2[k]); w2[k] = Int16(UP ? (v & ((1 << DEV) - 1)) : (v >> (16 - DEV)) << (16 - DEV)) } }
    var sw = [Int64](repeating: 0, count: nout); for i in 0..<nin { for j in 0..<nout { sw[j] += Int64(w2[i * nout + j]) } }
    banks[n] = Bank(nin: nin, nout: nout, w: buf(INV ? w2.map { ~$0 } : w2), e: buf(cp.map { e[$0] }), b: buf(cp.map { b[$0] }), sw: buf(sw)) }
do { let cb = queue.makeCommandBuffer()!                                   // every plate written as its up/hole place rows, once
    for (n, bk) in banks { var b2 = bk; b2.wr = zbuf(bk.nin * bk.nout * 2)
        enc(cb, pPackW, [bk.w, b2.wr!], [Int32(bk.nin), Int32(bk.nout)], threads: bk.nout * (bk.nin / 32), tg: 256); banks[n] = b2 }
    cb.commit(); cb.waitUntilCompleted() }
let PHI = bufData(loadData("phi.bin")), ARR = bufData(loadData("arrival.bin")), LOG2 = bufData(loadData("log2.bin")), EXP2 = bufData(loadData("exp2.bin"))
let wte0 = i64("wte.bin"), wpe0 = i64("wpe.bin")
var wte = wte0, wpe = wpe0
for t in 0..<50257 { for i in 0..<768 { wte[t * 768 + i] = wte0[t * 768 + chan[i]] } }
for t in 0..<1024 { for i in 0..<768 { wpe[t * 768 + i] = wpe0[t * 768 + chan[i]] } }
let wteB = buf(wte), wpeB = buf(wpe), wordPB = buf(wordP.map { Int32($0) }), seqB = zbuf(PMAX * 4)
let ref = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: NSString(string: ARG["refs"] ?? "refs_many.json").expandingTildeInPath))) as! [[String: Any]]
let L = 12, D = 768, V = 50257, PMAX = 64
let vocab: [Int: String] = {
    let base = NSString(string: "~/.cache/huggingface/hub/models--gpt2/snapshots").expandingTildeInPath
    let j = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: base + "/" + (try! FileManager.default.contentsOfDirectory(atPath: base)).first! + "/vocab.json"))) as! [String: Int]
    var v = [Int: String](); for (k, i) in j { v[i] = k.replacingOccurrences(of: "Ġ", with: " ").replacingOccurrences(of: "Ċ", with: "\\n") }; return v }()
func text(_ t: [Int]) -> String { return "'" + t.map { vocab[$0] ?? "?" }.joined() + "'" }

let R = (0...L).map { _ in zbuf(PMAX * D * 8) }
let qkvL = (0..<L).map { _ in zbuf(PMAX * 2304 * 8) }, q16L = (0..<L).map { _ in zbuf(PMAX * 2304 * 4) }, eqL = (0..<L).map { _ in zbuf(PMAX * 36 * 4) }
let sx = zbuf(PMAX * 8), parts = zbuf(12 * 64 * 3072 * 8 + 12 * 50264 * 8), x16 = zbuf((PMAX + 8) * 3072 * 4), ex = zbuf(PMAX * 4), pile = zbuf(PMAX * V * 8), LS = zbuf(PMAX * 8)
let xr = zbuf((PMAX + 8) * 16 * 96 * 4)
let att = zbuf(PMAX * D * 8), hid = zbuf(PMAX * 3072 * 8), lg = zbuf(PMAX * V * 8), words = zbuf(PMAX * 4)
var P = 0
// the light entering every slab grouped by brightness; lv=bottom,top,top read = groups per direction (bottom = lower half of the depths)
var CURDEPTH = 0
let NDEPTH = 12
let LVS: [Int] = (ARG["lv"] ?? (ARG["groups"] != nil ? "4096,4096,4096" : "0,0,0")).split(separator: ",").map { Int($0)! }
func shiftFor(_ levels: Int) -> Int { if levels <= 0 { return 0 }; var s = 0; while (32768 >> s) > levels { s += 1 }; return s }
func groupShift(_ bk: Bank) -> Int { return bk.nout > 50000 ? shiftFor(LVS[2]) : shiftFor(CURDEPTH < NDEPTH / 2 ? LVS[0] : LVS[1]) }
func enc(_ cb: MTLCommandBuffer, _ ps: MTLComputePipelineState, _ bufs: [MTLBuffer], _ consts: [Int32], threads: Int, tg: Int, groups: Int = 0, height: Int = 1) {
    let e = cb.makeComputeCommandEncoder()!; e.setComputePipelineState(ps)
    for (i, b) in bufs.enumerated() { e.setBuffer(b, offset: 0, index: i) }
    for (k, c) in consts.enumerated() { var v = c; e.setBytes(&v, length: 4, index: bufs.count + k) }
    if groups > 0 { e.dispatchThreadgroups(MTLSize(width: groups, height: height, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)) }
    else { e.dispatchThreads(MTLSize(width: threads, height: height, depth: 1), threadsPerThreadgroup: MTLSize(width: min(tg, threads), height: 1, depth: 1)) }
    e.endEncoding()
}
func product(_ cb: MTLCommandBuffer, _ X: MTLBuffer, _ bk: Bank, r0: Int, nr: Int, stream: Bool = false) {   // the light onto 16 bits, its place rows, then every neuron's count
    enc(cb, pTo16, [X, x16, ex, sx], [Int32(bk.nin), Int32(r0), 0, Int32(groupShift(bk))], threads: 0, tg: 256, groups: nr)
    let words = bk.nin / 32
    enc(cb, pRowsX, [x16, xr], [Int32(bk.nin), Int32(r0), Int32(nr * words)], threads: nr * words, tg: 256)
    var r = r0
    while r < r0 + nr { let m = min(8, r0 + nr - r)
        enc(cb, pUpDown, [bk.wr!, xr, pile, sx, bk.sw!], [Int32(bk.nin), Int32(bk.nout), Int32(r), Int32(m)], threads: 0, tg: 256, groups: (bk.nout + 7) / 8)
        r += m }
}
func layer(_ cb: MTLCommandBuffer, _ d: Int, r0: Int, nr: Int) {
    CURDEPTH = d
    CURDEPTH = d                 // positions r0..r0+nr through one layer, all at once
    let o = ORDERL[d]                                                        // which depth's plates sit at this depth (the swap test)
    let bq = banks["qkv.\(o)"]!, bp = banks["proj.\(o)"]!, bf = banks["fc.\(o)"]!, bm = banks["mproj.\(o)"]!
    let r = Int32(r0)
    product(cb, R[d], bq, r0: r0, nr: nr, stream: true); enc(cb, pGaugeLog, [x16, ex, LS, LOG2], [r], threads: 0, tg: 32, groups: nr)
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
func readTop(_ cb: MTLCommandBuffer, r0: Int, nr: Int) {
    let bh = banks["head"]!
    product(cb, R[L], bh, r0: r0, nr: nr, stream: true); enc(cb, pGaugeLog, [x16, ex, LS, LOG2], [Int32(r0)], threads: 0, tg: 32, groups: nr)
    enc(cb, pGauged, [pile, bh.e, ex, LS, bh.b, lg, EXP2], [Int32(V), Int32(r0), Int32(nr * V)], threads: nr * V, tg: 256)
    enc(cb, pRace, [lg, words], [Int32(V), Int32(r0)], threads: 0, tg: 1024, groups: nr)
}

let CASES = ARG["all"] != nil ? Array(0..<ref.count) : [CASE]
var nIdentical = 0, report = [String](), totalWords = 0, totalTime = 0.0
for CASE in CASES {
    let ids = ref[CASE]["ids"] as! [Int], prompt = ref[CASE]["prompt"] as! String
    let N = min(WORDS, (ref[CASE]["ref"] as! [Int]).count, PMAX - ids.count), want = Array((ref[CASE]["ref"] as! [Int]).prefix(N)), S = ids.count
    P = S + N
    var seq = ids, gen = [Int]()
    let t0 = Date(); var tGen = Date()
    let r0p = R[0].contents().bindMemory(to: Int64.self, capacity: P * D)
    for q in 0..<S { for i in 0..<D { let e = wte[seq[q] * D + i] + wpe[q * D + i]; r0p[q * D + i] = INV ? ~e : e } }
    // one unbroken run: the prompt all at once, then every new word flows straight into the next position, left to right
    let cb = queue.makeCommandBuffer()!
    for d in 0..<L { layer(cb, d, r0: 0, nr: S) }
    readTop(cb, r0: S - 1, nr: 1)
    for p in S..<(S + N - 1) {
        enc(cb, pNext, [words, wordPB, wteB, wpeB, R[0], seqB], [Int32(p - 1)], threads: 768, tg: 256)
        for d in 0..<L { layer(cb, d, r0: p, nr: 1) }
        readTop(cb, r0: p, nr: 1)
    }
    cb.commit(); cb.waitUntilCompleted()
    let wd = words.contents().bindMemory(to: Int32.self, capacity: P)
    gen = (0..<N).map { wordP[Int(wd[S - 1 + $0])] }
    if ARG["check"] != nil {
        let lgp = lg.contents().bindMemory(to: Int64.self, capacity: P * V); var slot = [Int](repeating: 0, count: V); for (j, w) in wordP.enumerated() { slot[w] = j }
        print("  channels \(chan[0]),\(chan[1]),\(chan[2]) | ' the' at slot \(slot[262]) | piles at the last word: " + [262, 3139, 11, 290, 50256].map { String(lgp[(P - 2) * V + slot[$0]]) }.joined(separator: " ")) }
    let wall = Date().timeIntervalSince(t0), wallGen = wall
    totalWords += N; totalTime += wall
    let same = gen == want
    if same { nIdentical += 1 } else {
        let j = (0..<N).first { gen[$0] != want[$0] }!
        let mg = (ref[CASE]["margins"] as? [Double]).map { String(format: "%.4f", $0[j]) } ?? "?"
        report.append("  '\(prompt)': first differs at word \(j + 1): \(text([gen[j]])) where the model says \(text([want[j]])) (the model's own lead there: \(mg))") }
    if !QUIET { print("'\(prompt)' -> \(text(gen))  \(same ? "IDENTICAL" : "DIFFERENT")  " + String(format: "%.3f s, %.1f words/s (after the prompt: %.1f)", wall, Double(N) / wall, Double(N - 1) / wallGen)) }
}
print("\(nIdentical) of \(CASES.count) prompts identical to the model" + (SHUF ? " (random wiring, seed \(argInt("seed", 0)))" : "") + (INV ? " (the whole machine on the negative)" : ""))
report.forEach { print($0) }
print(String(format: "%d words in %.2f s: %.1f words/s, prompt included", totalWords, totalTime, Double(totalWords) / totalTime))
