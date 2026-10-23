// QUANTUMGRANITE ON THE GPU: Granite-3.1-1B-A400M. Every number is a whole number in base 2, with no float while it
// runs. It is granite_base2.py's machine, number for number, as one unbroken run on the GPU.
//   - Every position goes through all 24 layers in turn; the keys and values of earlier positions stay on the GPU.
//   - Each layer: the arrival, then the router's race picks 8 of the 32 experts right on the GPU. Only those 8 plates
//     are read; the other 24 are never touched.
//   - The word the race gives goes straight in at the next position. Nothing leaves the GPU until the text is done.
// Plates are mapped straight from granite_base2/*.npy (row-order copies from granite_gpu/), with no copy.
//   swiftc -O -o granite_stream granite_stream.swift && ./granite_stream all=1
//   ./granite_stream case=0 dump=dump0.bin   (the stream after the last layer at every position, and every lit set)
import Foundation
import Metal

var ARG = [String: String]()
for a in CommandLine.arguments.dropFirst() { let p = a.split(separator: "=", maxSplits: 1).map(String.init); ARG[p[0]] = p.count > 1 ? p[1] : "1" }
func argInt(_ k: String, _ d: Int) -> Int { return Int(ARG[k] ?? "") ?? d }
let CASE = argInt("case", 0), WORDS = argInt("words", 16), QUIET = ARG["quiet"] != nil
let L = 24, D = 1024, V = 49152, NE = 32, I = 512, PMAX = 64
let dev = MTLCreateSystemDefaultDevice()!, queue = dev.makeCommandQueue()!

struct NB { let buf: MTLBuffer; let off: Int; let bytes: Int }       // a mapped .npy: the buffer and where its numbers start
func mapNpy(_ name: String) -> NB {
    let fm = FileManager.default
    let path = fm.fileExists(atPath: "granite_gpu/\(name).npy") ? "granite_gpu/\(name).npy" : "granite_base2/\(name).npy"
    let fd = open(path, O_RDONLY); precondition(fd >= 0, "cannot open \(path)")
    var st = stat(); fstat(fd, &st); let size = Int(st.st_size), page = Int(getpagesize()), len = (size + page - 1) / page * page
    let ptr = mmap(nil, len, PROT_READ, MAP_SHARED, fd, 0)!; close(fd)
    let b = ptr.assumingMemoryBound(to: UInt8.self); precondition(b[1] == 0x4E && b[2] == 0x55, "not an npy file: \(path)")
    let major = b[6]; let hlen = major == 1 ? Int(b[8]) | Int(b[9]) << 8 : Int(b[8]) | Int(b[9]) << 8 | Int(b[10]) << 16 | Int(b[11]) << 24
    let off = (major == 1 ? 10 : 12) + hlen
    let header = String(bytes: UnsafeBufferPointer(start: b + (major == 1 ? 10 : 12), count: hlen), encoding: .ascii)!
    precondition(header.contains("'fortran_order': False"), "\(path) is not in row order")
    return NB(buf: dev.makeBuffer(bytesNoCopy: ptr, length: len, options: .storageModeShared, deallocator: nil)!, off: off, bytes: size - off)
}
let ARRn = mapNpy("arrival"), ALEN = ARRn.bytes / 8
let PARTHALF = 16 * V, PILEHALF = PMAX * V, EPH = 8 * D

let src = """
#include <metal_stdlib>
using namespace metal;
constant long EPS40 = 1099512;        // round(1e-6 * 2^40), as in granite_base2/eps40.npy
constant long ALEN = \(ALEN);
constant uint PARTHALF = \(PARTHALF);   // the down parts sit after the up parts
constant uint PILEHALF = \(PILEHALF);   // the down piles sit after the up piles
constant uint EPH = \(EPH);             // the 8 lit experts' piles: up, then down
// shifts of 64 places or more, by numpy's rule (granite_base2.py's): left gives 0, right gives the sign (0 or -1).
// They happen at the few neurons whose weights are all zero (their power of two is about 2^-115).
inline long lsh(long x, long b) { return b >= 64 ? 0 : x << b; }
inline long rshn(long x, long b) { return b >= 64 ? (x < 0 ? -1 : 0) : x >> b; }
inline long rsh(long x, long s) { if (s <= 0) return lsh(x, -s); return rshn(x + lsh(1L, s - 1), s); }
inline long log2fix(ulong v, device const long* LOG2) {
    int t = 63 - clz(v); ulong m = t >= 16 ? (v >> (t - 16)) & 0xFFFF : (v << (16 - t)) & 0xFFFF; return (long)t * 65536 + LOG2[m]; }
inline long power(long x, long F, device const long* EXP2) { long I = F >> 16, f = F & 0xFFFF; return rsh(x * EXP2[f], 16 - I); }
inline long shfl_down(long v, ushort o) { uint lo = simd_shuffle_down((uint)(ulong)v, o), hi = simd_shuffle_down((uint)((ulong)v >> 32), o); return (long)(((ulong)hi << 32) | lo); }
inline long bcast0(long v) { uint lo = simd_broadcast((uint)(ulong)v, 0), hi = simd_broadcast((uint)((ulong)v >> 32), 0); return (long)(((ulong)hi << 32) | lo); }
inline long sum64(long v) { for (ushort o = 16; o > 0; o >>= 1) v += shfl_down(v, o); return bcast0(v); }
inline ulong or64(ulong v) { uint hi = simd_or((uint)(v >> 32)), lo = simd_or((uint)v); return ((ulong)hi << 32) | lo; }
inline ulong mag(long v) { return (ulong)(v < 0 ? -v : v); }
inline int bitlen(ulong v) { return v ? 64 - clz(v) : 0; }
inline long epsterm(long n, int e) { int s = 40 + 2 * e; return s >= 0 ? rshn(EPS40 * n, s) : lsh(EPS40 * n, -s); }

kernel void to16(device const long* X [[buffer(0)]], device int* x16 [[buffer(1)]], device int* ex [[buffer(2)]],
                 constant uint& n [[buffer(3)]], constant uint& r0 [[buffer(4)]],
                 uint t [[thread_position_in_threadgroup]], uint g [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    // each row on 16 bits, its own power of two (from the OR of its sizes)
    threadgroup ulong ors[8]; uint p = r0 + g; device const long* x = X + p * n;
    ulong o = 0; for (uint i = t; i < n; i += 256) o |= mag(x[i]);
    o = or64(o); if (lane == 0) ors[sg] = o; threadgroup_barrier(mem_flags::mem_threadgroup);
    ulong all = 0; for (int k = 0; k < 8; k++) all |= ors[k];
    int sh = max(bitlen(all) - 15, 0);
    for (uint i = t; i < n; i += 256) x16[p * n + i] = (int)rsh(x[i], sh);
    if (t == 0) ex[p] = -16 + sh; }
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
kernel void gauge_rms(device const int* x16 [[buffer(0)]], device const int* ex [[buffer(1)]], device long* LS [[buffer(2)]],
                      device const long* LOG2 [[buffer(3)]], constant uint& p [[buffer(4)]], uint lane [[thread_index_in_simdgroup]]) {
    // log2 of the root mean square over 1024: half the log of the sum of squares, minus half of log2(1024)
    long s2 = 0; for (uint i = lane; i < 1024; i += 32) { long v = x16[p * 1024 + i]; s2 += v * v; }
    s2 = sum64(s2);
    if (lane == 0) LS[p] = (log2fix((ulong)(s2 + epsterm(1024, ex[p])), LOG2) >> 1) - 5 * 65536 + (long)ex[p] * 65536; }
kernel void gauged(device const long* pile [[buffer(0)]], device const long* we [[buffer(1)]], device const int* ex [[buffer(2)]], device const long* LS [[buffer(3)]],
                   device long* out [[buffer(4)]], device const long* EXP2 [[buffer(5)]], constant uint& nout [[buffer(6)]], constant uint& p [[buffer(7)]],
                   uint c [[thread_position_in_grid]]) {
    if (c >= nout) return; uint i = p * nout + c;
    out[i] = power(pile[i] - pile[PILEHALF + i], (we[c] + ex[p] + 16) * 65536 - LS[p], EXP2); }
kernel void plain(device const long* pile [[buffer(0)]], device const long* we [[buffer(1)]], device const int* ex [[buffer(2)]],
                  device long* R [[buffer(3)]], constant uint& p [[buffer(4)]], uint c [[thread_position_in_grid]]) {
    if (c >= 1024) return; uint i = p * 1024 + c;
    R[i] += rsh(pile[i] - pile[PILEHALF + i], -(we[c] + ex[p] + 16)); }
kernel void heads(device const long* QKV [[buffer(0)]], device int* KQ16 [[buffer(1)]], device int* eKQ [[buffer(2)]],
                  device const long* COS [[buffer(3)]], device const long* SIN [[buffer(4)]], constant uint& p [[buffer(5)]],
                  uint lane [[thread_index_in_simdgroup]], uint h [[threadgroup_position_in_grid]]) {
    // one simdgroup per head (16 query heads, then 8 key heads): the turn of pair (i, i+32), then 16 bits
    device const long* x = QKV + p * 2048 + (h < 16 ? h * 64 : 1024 + (h - 16) * 64);
    long a = x[lane], b = x[lane + 32], c = COS[p * 32 + lane], s = SIN[p * 32 + lane];
    long t0 = rsh(a * c - b * s, 30), t1 = rsh(b * c + a * s, 30);
    int sh = max(bitlen(or64(mag(t0) | mag(t1))) - 15, 0);
    KQ16[(p * 24 + h) * 64 + lane] = (int)rsh(t0, sh); KQ16[(p * 24 + h) * 64 + lane + 32] = (int)rsh(t1, sh);
    if (lane == 0) eKQ[p * 24 + h] = -16 + sh; }
kernel void attend(device const long* QKV [[buffer(0)]], device const int* KQ16 [[buffer(1)]], device const int* eKQ [[buffer(2)]], device long* out [[buffer(3)]],
                   device const long* ARR [[buffer(4)]], device const long* LOG2 [[buffer(5)]], device const long* EXP2 [[buffer(6)]], constant uint& p [[buffer(7)]],
                   uint lane [[thread_index_in_simdgroup]], uint h [[threadgroup_position_in_grid]]) {
    // the arrival: q meets every earlier key; the clock starts at the race's winner; the shares mix the values
    uint kh = 16 + h / 2; long sc[\(PMAX)]; long eq = eKQ[p * 24 + h];
    for (uint s = 0; s <= p; s++) {
        long dot = (long)KQ16[(p * 24 + h) * 64 + lane] * KQ16[(s * 24 + kh) * 64 + lane] + (long)KQ16[(p * 24 + h) * 64 + lane + 32] * KQ16[(s * 24 + kh) * 64 + lane + 32];
        sc[s] = rsh(sum64(dot), -(eq + eKQ[s * 24 + kh] + 10)); }
    long start = sc[0]; for (uint s = 1; s <= p; s++) start = max(start, sc[s]);
    long n0 = 0, n1 = 0, Z = 0; device const long* v = QKV + 1536 + (h / 2) * 64;
    for (uint s = 0; s <= p; s++) { long dist = start - sc[s]; long a = dist < ALEN ? ARR[dist] : 0; if (a == 0) continue; Z += a;
        n0 += a * v[s * 2048 + lane]; n1 += a * v[s * 2048 + lane + 32]; }
    int t = max(bitlen(or64(mag(n0) | mag(n1))) - 46, 0); long F = (long)t * 65536 - log2fix((ulong)Z, LOG2);
    out[p * 1024 + h * 64 + lane] = power(rsh(n0, t), F, EXP2); out[p * 1024 + h * 64 + lane + 32] = power(rsh(n1, t), F, EXP2); }
kernel void route(device const long* RP [[buffer(0)]], device int* LIT [[buffer(1)]], device long* A [[buffer(2)]], device const long* ARR [[buffer(3)]],
                  constant uint& p [[buffer(4)]], uint lane [[thread_index_in_simdgroup]]) {
    // the router's race, 8 times: the largest pile not yet lit (a tie goes to the lower expert); its share e^-(lead - pile)
    long v = RP[p * 32 + lane]; bool taken = false; long lead0 = 0;
    for (int s = 0; s < 8; s++) {
        long b = taken ? LONG_MIN : v; int bi = (int)lane;
        for (ushort o = 16; o > 0; o >>= 1) { long ob = shfl_down(b, o); int oi = simd_shuffle_down(bi, o); if (ob > b || (ob == b && oi < bi)) { b = ob; bi = oi; } }
        long B = bcast0(b); int BI = simd_broadcast(bi, 0); if ((int)lane == BI) taken = true;
        long lead = rsh(B, 6); if (s == 0) lead0 = lead;
        if (lane == 0) { LIT[p * 8 + s] = BI; long d = lead0 - lead; A[p * 8 + s] = d < ALEN ? ARR[d] : 0; } } }
kernel void egate1(device const short* W [[buffer(0)]], device const int* X [[buffer(1)]], device long* part [[buffer(2)]], device const int* LIT [[buffer(3)]],
                   constant uint& nin [[buffer(4)]], constant uint& nout [[buffer(5)]], constant uint& p [[buffer(6)]], constant uint& span [[buffer(7)]],
                   constant uint& xbase [[buffer(8)]], constant uint& xstride [[buffer(9)]], uint3 id [[thread_position_in_grid]]) {
    // the same meetings on a lit expert's plate: slot s reads expert LIT[s]'s plate and no other
    uint j = id.x; if (j >= nout) return; uint s = id.z, ks = nin / span, i0 = id.y * span;
    device const short* w = W + (ulong)LIT[p * 8 + s] * nin * nout; device const int* x = X + xbase + s * xstride;
    int ulo = 0, uhi = 0, dlo = 0, dhi = 0;
    for (uint i = i0; i < i0 + span; i++) { int pr = (int)w[i * nout + j] * x[i];
        if (pr >= 0) { ulo += pr & 0x7FFF; uhi += pr >> 15; } else { int q = -pr; dlo += q & 0x7FFF; dhi += q >> 15; } }
    uint o = (s * ks + id.y) * nout + j; part[o] = ((long)uhi << 15) + ulo; part[PARTHALF + o] = ((long)dhi << 15) + dlo; }
kernel void eaddparts(device const long* part [[buffer(0)]], device long* pile [[buffer(1)]], constant uint& nout [[buffer(2)]], constant uint& ks [[buffer(3)]],
                      uint t [[thread_position_in_grid]]) {
    if (t >= 8 * nout) return; uint s = t / nout, j = t % nout; long u = 0, d = 0;
    for (uint k = 0; k < ks; k++) { u += part[(s * ks + k) * nout + j]; d += part[PARTHALF + (s * ks + k) * nout + j]; }
    pile[t] = u; pile[EPH + t] = d; }
kernel void egauged(device const long* pile [[buffer(0)]], device const long* we [[buffer(1)]], device const int* ex [[buffer(2)]], device const long* LS [[buffer(3)]],
                    device const int* LIT [[buffer(4)]], device long* out [[buffer(5)]], device const long* EXP2 [[buffer(6)]], constant uint& p [[buffer(7)]],
                    uint k [[thread_position_in_grid]]) {
    if (k >= 8 * 1024) return; uint s = k / 1024, c = k % 1024;
    out[k] = power(pile[k] - pile[EPH + k], (we[LIT[p * 8 + s] * 1024 + c] + ex[p] + 16) * 65536 - LS[p], EXP2); }
kernel void ebend(device const long* GU [[buffer(0)]], device long* H [[buffer(1)]], device const long* SIG [[buffer(2)]], uint k [[thread_position_in_grid]]) {
    // the gated bend: g * share(g) * u, in each lit expert
    if (k >= 8 * 512) return; uint s = k / 512, j = k % 512;
    long g = GU[s * 1024 + j], u = GU[s * 1024 + 512 + j]; long idx = rsh(g, 4) + 65536;
    long share = idx < 0 ? 0 : (idx > 131071 ? 65536 : SIG[idx]);
    H[k] = rsh(rsh(g * share, 16) * u, 16); }
kernel void eplain(device const long* pile [[buffer(0)]], device const long* we [[buffer(1)]], device const int* exe [[buffer(2)]], device const int* LIT [[buffer(3)]],
                   device long* Y [[buffer(4)]], constant uint& p [[buffer(5)]], uint k [[thread_position_in_grid]]) {
    if (k >= 8 * 1024) return; uint s = k / 1024, c = k % 1024;
    Y[k] = rsh(pile[k] - pile[EPH + k], -(we[LIT[p * 8 + s] * 1024 + c] + exe[s] + 16)); }
kernel void emix(device const long* Y [[buffer(0)]], device const long* A [[buffer(1)]], device long* R [[buffer(2)]],
                 device const long* LOG2 [[buffer(3)]], device const long* EXP2 [[buffer(4)]], constant uint& p [[buffer(5)]],
                 uint i [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    // the 8 writes mixed by their shares, kept to 46 bits before the 2^x table, then divided by the shares' total
    threadgroup ulong ors[32]; long S = 0, Z = 0;
    for (int s = 0; s < 8; s++) { long a = A[p * 8 + s]; S += a * Y[s * 1024 + i]; Z += a; }
    ulong o = or64(mag(S)); if (lane == 0) ors[sg] = o; threadgroup_barrier(mem_flags::mem_threadgroup);
    ulong all = 0; for (int k = 0; k < 32; k++) all |= ors[k];
    int t = max(bitlen(all) - 46, 0);
    R[p * 1024 + i] += power(rsh(S, t), (long)t * 65536 - log2fix((ulong)Z, LOG2), EXP2); }
kernel void race(device const long* lg [[buffer(0)]], device int* word [[buffer(1)]], constant uint& VV [[buffer(2)]], constant uint& p [[buffer(3)]],
                 uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup long bv[32]; threadgroup int bi[32]; device const long* v = lg + p * VV;
    long b = LONG_MIN; int bw = 0;
    for (uint i = t; i < VV; i += 1024) if (v[i] > b) { b = v[i]; bw = (int)i; }
    for (ushort o = 16; o > 0; o >>= 1) { long ob = shfl_down(b, o); int ow = simd_shuffle_down(bw, o); if (ob > b || (ob == b && ow < bw)) { b = ob; bw = ow; } }
    if (lane == 0) { bv[sg] = b; bi[sg] = bw; } threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) { long B = bv[0]; int W = bi[0]; for (int k = 1; k < 32; k++) if (bv[k] > B || (bv[k] == B && bi[k] < W)) { B = bv[k]; W = bi[k]; } word[p] = W; } }
kernel void keeprow(device const long* R [[buffer(0)]], device long* DR [[buffer(1)]], constant uint& o [[buffer(2)]], constant uint& p [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) { if (i < 1024) DR[o + i] = R[p * 1024 + i]; }
kernel void keepn(device const long* S [[buffer(0)]], device long* DS [[buffer(1)]], constant uint& o [[buffer(2)]], constant uint& n [[buffer(3)]],
                  uint i [[thread_position_in_grid]]) { if (i < n) DS[o + i] = S[i]; }
kernel void nextin(device const int* word [[buffer(0)]], device const int* EMB [[buffer(1)]], device long* R [[buffer(2)]], constant uint& p [[buffer(3)]],
                   uint i [[thread_position_in_grid]]) {
    // the word the race just gave at p - 1 goes straight in at p
    if (i >= 1024) return; R[p * 1024 + i] = EMB[(long)word[p - 1] * 1024 + i]; }
"""
setvbuf(stdout, nil, _IONBF, 0)
let lib = try! dev.makeLibrary(source: src, options: MTLCompileOptions())
func pipe(_ n: String) -> MTLComputePipelineState { return try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
let pTo16 = pipe("to16"), pGate1 = pipe("gate1"), pAdd1 = pipe("addparts1"), pGauge = pipe("gauge_rms"), pGauged = pipe("gauged"), pPlain = pipe("plain")
let pHeads = pipe("heads"), pAttend = pipe("attend"), pRoute = pipe("route"), pEGate = pipe("egate1"), pEAdd = pipe("eaddparts"), pEGauged = pipe("egauged")
let pEBend = pipe("ebend"), pEPlain = pipe("eplain"), pEMix = pipe("emix"), pRace = pipe("race"), pNext = pipe("nextin"), pKeep = pipe("keeprow"), pKeepN = pipe("keepn")
let DUMP = ARG["dump"] != nil

let T = Dictionary(uniqueKeysWithValues: ["cos", "sin", "sig", "log2", "exp2", "embed"].map { ($0, mapNpy($0)) })
let EMB = T["embed"]!, COS = T["cos"]!, SIN = T["sin"]!, SIG = T["sig"]!, LOG2 = T["log2"]!, EXP2 = T["exp2"]!
struct Plate { let w, e: NB }
func plate(_ n: String) -> Plate { return Plate(w: mapNpy(n + ".w"), e: mapNpy(n + ".e")) }
let QKVp = (0..<L).map { plate("qkv.\($0)") }, Op = (0..<L).map { plate("o.\($0)") }, RTp = (0..<L).map { plate("router.\($0)") }
let GUp = (0..<L).map { plate("gu.\($0)") }, DNp = (0..<L).map { plate("down.\($0)") }, HEADp = plate("head")
func zbuf(_ bytes: Int) -> NB { return NB(buf: dev.makeBuffer(length: max(bytes, 8), options: .storageModeShared)!, off: 0, bytes: bytes) }
let R = zbuf(PMAX * D * 8), x16 = zbuf(PMAX * D * 4), ex = zbuf(PMAX * 4), LS = zbuf(PMAX * 8)
let parts = zbuf(2 * PARTHALF * 8), pile = zbuf(2 * PILEHALF * 8), att = zbuf(PMAX * D * 8), lg = zbuf(PMAX * V * 8), words = zbuf(PMAX * 4)
let QKV = (0..<L).map { _ in zbuf(PMAX * 2048 * 8) }, KQ = (0..<L).map { _ in zbuf(PMAX * 24 * 64 * 4) }, EKQ = (0..<L).map { _ in zbuf(PMAX * 24 * 4) }
let RP = zbuf(PMAX * 32 * 8), LIT = (0..<L).map { _ in zbuf(PMAX * 8 * 4) }, A = zbuf(PMAX * 8 * 8)
let DR = zbuf(DUMP ? L * PMAX * D * 8 : 8), DA = zbuf(DUMP ? L * PMAX * D * 8 : 8), D0 = zbuf(DUMP ? PMAX * 20480 * 8 : 8)
let epile = zbuf(2 * EPH * 8), EGU = zbuf(8 * 1024 * 8), EH = zbuf(8 * 512 * 8), x16e = zbuf(8 * 512 * 4), exe = zbuf(8 * 4), Y = zbuf(8 * 1024 * 8)

var E: MTLComputeCommandEncoder! = nil
func run(_ ps: MTLComputePipelineState, _ bufs: [NB], _ consts: [UInt32], threads: Int = 0, height: Int = 1, depth: Int = 1, tg: Int = 256, groups: Int = 0) {
    E.setComputePipelineState(ps)
    for (i, b) in bufs.enumerated() { E.setBuffer(b.buf, offset: b.off, index: i) }
    for (k, c) in consts.enumerated() { var v = c; E.setBytes(&v, length: 4, index: bufs.count + k) }
    if groups > 0 { E.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)) }
    else { E.dispatchThreads(MTLSize(width: threads, height: height, depth: depth), threadsPerThreadgroup: MTLSize(width: min(tg, threads), height: 1, depth: 1)) }
}
func product(_ X: NB, _ w: NB, nout: Int, p: Int) {                 // the light onto 16 bits, then every core takes piles x a slice of lines
    run(pTo16, [X, x16, ex], [1024, UInt32(p)], tg: 256, groups: 1)
    run(pGate1, [w, x16, parts], [1024, UInt32(nout), UInt32(p), 64], threads: nout, height: 16)
    run(pAdd1, [parts, pile], [UInt32(nout), UInt32(p), 16], threads: nout)
}
func layer(_ d: Int, _ p: Int) {
    let P = UInt32(p)
    product(R, QKVp[d].w, nout: 2048, p: p); run(pGauge, [x16, ex, LS, LOG2], [P], tg: 32, groups: 1)
    run(pGauged, [pile, QKVp[d].e, ex, LS, QKV[d], EXP2], [2048, P], threads: 2048)
    run(pHeads, [QKV[d], KQ[d], EKQ[d], COS, SIN], [P], tg: 32, groups: 24)
    run(pAttend, [QKV[d], KQ[d], EKQ[d], att, T["arrival"] ?? ARRn, LOG2, EXP2], [P], tg: 32, groups: 16)
    product(att, Op[d].w, nout: 1024, p: p); run(pPlain, [pile, Op[d].e, ex, R], [P], threads: 1024)
    if DUMP { run(pKeep, [R, DA], [UInt32((d * PMAX + p) * D), P], threads: 1024) }
    // the router and the experts meet the same light: R after the arrival, on 16 bits, with one gauge
    product(R, RTp[d].w, nout: 32, p: p); run(pGauge, [x16, ex, LS, LOG2], [P], tg: 32, groups: 1)
    run(pGauged, [pile, RTp[d].e, ex, LS, RP, EXP2], [32, P], threads: 32)
    run(pRoute, [RP, LIT[d], A, ARRn], [P], tg: 32, groups: 1)
    run(pEGate, [GUp[d].w, x16, parts, LIT[d]], [1024, 1024, P, 64, UInt32(p * 1024), 0], threads: 1024, height: 16, depth: 8)
    run(pEAdd, [parts, epile], [1024, 16], threads: 8 * 1024)
    run(pEGauged, [epile, GUp[d].e, ex, LS, LIT[d], EGU, EXP2], [P], threads: 8 * 1024)
    run(pEBend, [EGU, EH, SIG], [], threads: 8 * 512)
    run(pTo16, [EH, x16e, exe], [512, 0], tg: 256, groups: 8)
    run(pEGate, [DNp[d].w, x16e, parts, LIT[d]], [512, 1024, P, 64, 0, 512], threads: 1024, height: 8, depth: 8)
    run(pEAdd, [parts, epile], [1024, 8], threads: 8 * 1024)
    run(pEPlain, [epile, DNp[d].e, exe, LIT[d], Y], [P], threads: 8 * 1024)
    if DUMP && d == 0 { run(pKeepN, [EGU, D0], [UInt32(p * 20480), 8192], threads: 8192); run(pKeepN, [EH, D0], [UInt32(p * 20480 + 8192), 4096], threads: 4096)
                        run(pKeepN, [Y, D0], [UInt32(p * 20480 + 12288), 8192], threads: 8192) }
    run(pEMix, [Y, A, R, LOG2, EXP2], [P], tg: 1024, groups: 1)
    if DUMP { run(pKeep, [R, DR], [UInt32((d * PMAX + p) * D), P], threads: 1024) }
}
func readTop(_ p: Int) {
    let P = UInt32(p)
    product(R, HEADp.w, nout: V, p: p); run(pGauge, [x16, ex, LS, LOG2], [P], tg: 32, groups: 1)
    run(pGauged, [pile, HEADp.e, ex, LS, lg, EXP2], [UInt32(V), P], threads: V)
    run(pRace, [lg, words], [UInt32(V), P], tg: 1024, groups: 1)
}
let ref = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: "refs_granite.json"))) as! [[String: Any]]
let CASES = ARG["all"] != nil ? Array(0..<ref.count) : [CASE]
var nSame = 0, totW = 0, totGen = 0, tAll = 0.0, tGen = 0.0, out = [[String: Any]]()
for c in CASES {
    let ids = ref[c]["ids"] as! [Int], want = Array((ref[c]["ref"] as! [Int]).prefix(WORDS)), S = ids.count, N = want.count
    precondition(S + N - 1 <= PMAX)
    let rp = R.buf.contents().bindMemory(to: Int64.self, capacity: PMAX * D)
    let emb = (EMB.buf.contents() + EMB.off).bindMemory(to: Int32.self, capacity: V * D)
    for q in 0..<S { for i in 0..<D { rp[q * D + i] = Int64(emb[ids[q] * D + i]) } }
    let t0 = Date()
    var cb = queue.makeCommandBuffer()!; E = cb.makeComputeCommandEncoder()!      // the prompt, one position after another
    for p in 0..<S { for d in 0..<L { layer(d, p) } }
    readTop(S - 1); E.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    let t1 = Date()
    cb = queue.makeCommandBuffer()!; E = cb.makeComputeCommandEncoder()!           // one unbroken run: every new word flows straight into the next spot
    for p in S..<(S + N - 1) { run(pNext, [words, EMB, R], [UInt32(p)], threads: 1024); for d in 0..<L { layer(d, p) }; readTop(p) }
    E.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    let t2 = Date(); tAll += t2.timeIntervalSince(t0); tGen += t2.timeIntervalSince(t1); totW += N; totGen += N - 1
    let wd = words.buf.contents().bindMemory(to: Int32.self, capacity: PMAX); let got = (0..<N).map { Int(wd[S - 1 + $0]) }
    if got == want { nSame += 1 }
    out.append(["case": c, "words": got])
    if !QUIET { print(String(format: "case %2d  %@  %.3f s (prompt %d positions %.3f s; %d words after it %.3f s, %.1f words/s)", c, got == want ? "IDENTICAL" : "DIFFERENT",
                             t2.timeIntervalSince(t0), S, t1.timeIntervalSince(t0), N - 1, t2.timeIntervalSince(t1), Double(N - 1) / t2.timeIntervalSince(t1))) }
    if let dp = ARG["dump"] {                                                       // the stream after the last layer, and every lit set
        var d = Data(); let P = S + N - 1
        d.append(Data(bytes: R.buf.contents(), count: P * D * 8))
        for l in 0..<L { d.append(Data(bytes: LIT[l].buf.contents(), count: P * 8 * 4)) }
        for l in 0..<L { d.append(Data(bytes: QKV[l].buf.contents(), count: P * 2048 * 8)) }
        for l in 0..<L { d.append(Data(bytes: DR.buf.contents() + l * PMAX * D * 8, count: P * D * 8)) }
        for l in 0..<L { d.append(Data(bytes: DA.buf.contents() + l * PMAX * D * 8, count: P * D * 8)) }
        d.append(Data(bytes: D0.buf.contents(), count: P * 20480 * 8))
        try! d.write(to: URL(fileURLWithPath: dp))
    }
}
try! JSONSerialization.data(withJSONObject: out).write(to: URL(fileURLWithPath: ARG["out"] ?? "gpu_words.json"))
print("\(nSame) of \(CASES.count) texts identical to fp32 Granite's own words")
print(String(format: "%d words in %.2f s: %.1f words/s with the prompt; after the prompt %d words in %.2f s: %.1f words/s", totW, tAll, Double(totW) / tAll, totGen, tGen, Double(totGen) / tGen))
