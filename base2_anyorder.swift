// THE STREAMING MACHINE, ALL BASE 2, IN ANY ORDER IN TIME. GPT-2 (124M), every number a whole number in base 2, no float while it runs
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

let ROT = ARG["seed"] != nil ? (argInt("seed", 0) * 7919 + 13) % 1000 : 0
let src = """
#include <metal_stdlib>
using namespace metal;
constant long LOG768 = \(LOG768);
constant long EPS40 = \(EPS40);
constant long ALEN = \(ALEN);
#define RC 8
constant uint ROT = \(ROT);
inline long rsh(long x, long s) { if (s <= 0) return x << (-s); return (x + (1L << (s - 1))) >> s; }
inline long log2fix(ulong v, device const long* LOG2) {
    int t = 63 - clz(v); ulong m = t >= 16 ? (v >> (t - 16)) & 0xFFFF : (v << (16 - t)) & 0xFFFF; return (long)t * 65536 + LOG2[m]; }
inline long power(long x, long F, device const long* EXP2) { long I = F >> 16, f = F & 0xFFFF; return rsh(x * EXP2[f], 16 - I); }
inline long shfl_down(long v, ushort o) { uint lo = simd_shuffle_down((uint)(ulong)v, o), hi = simd_shuffle_down((uint)((ulong)v >> 32), o); return (long)(((ulong)hi << 32) | lo); }
inline long bcast0(long v) { uint lo = simd_broadcast((uint)(ulong)v, 0), hi = simd_broadcast((uint)((ulong)v >> 32), 0); return (long)(((ulong)hi << 32) | lo); }
inline long sum64(long v) { for (ushort o = 16; o > 0; o >>= 1) v += shfl_down(v, o); return bcast0(v); }

kernel void to16(device const long* X [[buffer(0)]], device int* x16 [[buffer(1)]], device int* ex [[buffer(2)]],
                 constant uint& n [[buffer(3)]], constant uint& r0 [[buffer(4)]],
                 uint t [[thread_position_in_threadgroup]], uint g [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint orh[8], orl[8]; uint p = r0 + g; device const long* x = X + p * n;
    ulong o = 0; for (uint i = t; i < n; i += 256) { long v = x[i]; o |= (ulong)(v < 0 ? -v : v); }
    uint hi = simd_or((uint)(o >> 32)), lo = simd_or((uint)o);
    if (lane == 0) { orh[sg] = hi; orl[sg] = lo; } threadgroup_barrier(mem_flags::mem_threadgroup);
    ulong all = 0; for (int k = 0; k < 8; k++) all |= ((ulong)orh[k] << 32) | orl[k];
    int top = all ? 64 - clz(all) : 0; int sh = max(top - 15, 0);
    for (uint i = t; i < n; i += 256) x16[p * n + i] = (int)rsh(x[i], sh);
    if (t == 0) ex[p] = -16 + sh; }
kernel void gate1(device const short* W [[buffer(0)]], device const int* x16 [[buffer(1)]], device long* part [[buffer(2)]],
                  constant uint& nin [[buffer(3)]], constant uint& nout [[buffer(4)]], constant uint& p [[buffer(5)]], constant uint& span [[buffer(6)]],
                  uint2 id [[thread_position_in_grid]]) {
    // one new position: thread (pile, slice of inputs); neighbouring threads read neighbouring piles' weights; two narrow cells, read once
    uint j = id.x; if (j >= nout) return; uint i0 = id.y * span;
    device const int* x = x16 + p * nin; int lo = 0, hi = 0;
    for (uint i = i0; i < i0 + span; i++) { int pr = (int)W[i * nout + j] * x[i]; lo += pr & 0x7FFF; hi += pr >> 15; }
    part[id.y * nout + j] = ((long)hi << 15) + lo; }
kernel void addparts1(device const long* part [[buffer(0)]], device long* pile [[buffer(1)]], constant uint& nout [[buffer(2)]], constant uint& p [[buffer(3)]],
                      constant uint& ks [[buffer(4)]], uint j [[thread_position_in_grid]]) {
    if (j >= nout) return; long s = 0; for (uint k = 0; k < ks; k++) s += part[((k + ROT) % ks) * nout + j]; pile[p * nout + j] = s; }
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
kernel void addpartsN(device const long* part [[buffer(0)]], device long* pile [[buffer(1)]], constant uint& nout [[buffer(2)]], constant uint& r0 [[buffer(3)]],
                      constant uint& ks [[buffer(4)]], constant uint& n [[buffer(5)]], uint t [[thread_position_in_grid]]) {
    if (t >= n) return; uint p = r0 + t / nout, j = t % nout; long s = 0;
    for (uint k = 0; k < ks; k++) s += part[(((k + ROT) % ks) * 64 + p) * nout + j]; pile[p * nout + j] = s; }
kernel void gauge_log(device const int* x16 [[buffer(0)]], device const int* ex [[buffer(1)]], device long* LS [[buffer(2)]],
                      device const long* LOG2 [[buffer(3)]], constant uint& r0 [[buffer(4)]], uint lane [[thread_index_in_simdgroup]], uint g [[threadgroup_position_in_grid]]) {
    uint p = r0 + g; long s1 = 0, s2 = 0;
    for (uint i = lane; i < 768; i += 32) { long v = x16[p * 768 + i]; s1 += v; s2 += v * v; }
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
    R[i] += rsh(pile[i], -((long)we[c] + ex[p] + 16)) + bias[c]; }
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
kernel void race(device const long* lg [[buffer(0)]], device int* word [[buffer(1)]], constant uint& V [[buffer(2)]], constant uint& r0 [[buffer(3)]],
                 uint t [[thread_position_in_threadgroup]], uint g [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup long bv[32]; threadgroup int bi[32]; uint p = r0 + g; device const long* v = lg + p * V;
    long b = LONG_MIN; int bw = 0;
    for (uint i = t; i < V; i += 1024) if (v[i] > b) { b = v[i]; bw = (int)i; }
    for (ushort o = 16; o > 0; o >>= 1) { long ob = shfl_down(b, o); int ow = simd_shuffle_down(bw, o); if (ob > b || (ob == b && ow < bw)) { b = ob; bw = ow; } }
    if (lane == 0) { bv[sg] = b; bi[sg] = bw; } threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) { long B = bv[0]; int W = bi[0]; for (int k = 1; k < 32; k++) if (bv[k] > B || (bv[k] == B && bi[k] < W)) { B = bv[k]; W = bi[k]; } word[p] = W; } }
kernel void zero(device long* a [[buffer(0)]], constant uint& off [[buffer(1)]], constant uint& n [[buffer(2)]], uint k [[thread_position_in_grid]]) { if (k < n) a[off + k] = 0; }
kernel void pour(device const long* nw [[buffer(0)]], device long* last [[buffer(1)]], device long* piles [[buffer(2)]], constant uint& j [[buffer(3)]],
                 constant uint& off [[buffer(4)]], constant uint& n [[buffer(5)]], constant uint& stride [[buffer(6)]], uint k [[thread_position_in_grid]]) {
    // block j's change since it last poured (new - old) lands on every pile above it; no change, nothing lands
    if (k >= n) return; uint i = off + k; long d = nw[i] - last[i]; if (d == 0) return; last[i] = nw[i];
    for (uint q = j + 1; q <= 24; q++) piles[q * stride + i] += d; }
kernel void copyl(device const long* a [[buffer(0)]], device long* b [[buffer(1)]], constant uint& off [[buffer(2)]], constant uint& n [[buffer(3)]], uint k [[thread_position_in_grid]]) {
    if (k < n) b[off + k] = a[off + k]; }
"""
setvbuf(stdout, nil, _IONBF, 0)
let dev = MTLCreateSystemDefaultDevice()!, queue = dev.makeCommandQueue()!
let lib = try! dev.makeLibrary(source: src, options: MTLCompileOptions())
func pipe(_ n: String) -> MTLComputePipelineState { return try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
let pTo16 = pipe("to16"), pGate1 = pipe("gate1"), pAdd1 = pipe("addparts1"), pGateN = pipe("gateN"), pAddN = pipe("addpartsN"), pGaugeLog = pipe("gauge_log"), pGauged = pipe("gauged"), pPlain = pipe("plain"), pEdge = pipe("edge")
let pHeads = pipe("heads16"), pAttend = pipe("attend"), pRace = pipe("race"), pCopy = pipe("copyl"), pZero = pipe("zero"), pPour = pipe("pour")
func loadData(_ name: String) -> Data { return try! Data(contentsOf: URL(fileURLWithPath: dir + name)) }
func bufData(_ d: Data) -> MTLBuffer { return d.withUnsafeBytes { dev.makeBuffer(bytes: $0.baseAddress!, length: d.count, options: .storageModeShared)! } }
func zbuf(_ bytes: Int) -> MTLBuffer { return dev.makeBuffer(length: max(bytes, 8), options: .storageModeShared)! }
func i64(_ name: String) -> [Int64] { let d = loadData(name); return d.withUnsafeBytes { Array($0.bindMemory(to: Int64.self)) } }
struct Bank { let nin, nout: Int; let w, e, b: MTLBuffer }
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
    banks[n] = Bank(nin: nin, nout: nout, w: buf(w2), e: buf(cp.map { e[$0] }), b: buf(cp.map { b[$0] })) }
let PHI = bufData(loadData("phi.bin")), ARR = bufData(loadData("arrival.bin")), LOG2 = bufData(loadData("log2.bin")), EXP2 = bufData(loadData("exp2.bin"))
let wte0 = i64("wte.bin"), wpe0 = i64("wpe.bin")
var wte = wte0, wpe = wpe0
for t in 0..<50257 { for i in 0..<768 { wte[t * 768 + i] = wte0[t * 768 + chan[i]] } }
for t in 0..<1024 { for i in 0..<768 { wpe[t * 768 + i] = wpe0[t * 768 + chan[i]] } }
let ref = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: NSString(string: ARG["refs"] ?? "refs_many.json").expandingTildeInPath))) as! [[String: Any]]
let L = 12, D = 768, V = 50257, PMAX = 64
let vocab: [Int: String] = {
    let base = NSString(string: "~/.cache/huggingface/hub/models--gpt2/snapshots").expandingTildeInPath
    let j = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: base + "/" + (try! FileManager.default.contentsOfDirectory(atPath: base)).first! + "/vocab.json"))) as! [String: Int]
    var v = [Int: String](); for (k, i) in j { v[i] = k.replacingOccurrences(of: "Ġ", with: " ").replacingOccurrences(of: "Ċ", with: "\\n") }; return v }()
func text(_ t: [Int]) -> String { return "'" + t.map { vocab[$0] ?? "?" }.joined() + "'" }

let qkvL = (0..<L).map { _ in zbuf(PMAX * 2304 * 8) }, q16L = (0..<L).map { _ in zbuf(PMAX * 2304 * 4) }, eqL = (0..<L).map { _ in zbuf(PMAX * 36 * 4) }
let parts = zbuf(12 * 64 * 3072 * 8 + 12 * 50264 * 8), x16 = zbuf((PMAX + 8) * 3072 * 4), ex = zbuf(PMAX * 4), pile = zbuf(PMAX * V * 8), LS = zbuf(PMAX * 8)
let att = zbuf(PMAX * D * 8), hid = zbuf(PMAX * 3072 * 8), lg = zbuf(PMAX * V * 8), words = zbuf(PMAX * 4)
var P = 0
func enc(_ cb: MTLCommandBuffer, _ ps: MTLComputePipelineState, _ bufs: [MTLBuffer], _ consts: [Int32], threads: Int, tg: Int, groups: Int = 0, height: Int = 1, offs: [Int] = []) {
    let e = cb.makeComputeCommandEncoder()!; e.setComputePipelineState(ps)
    for (i, b) in bufs.enumerated() { e.setBuffer(b, offset: i < offs.count ? offs[i] : 0, index: i) }
    for (k, c) in consts.enumerated() { var v = c; e.setBytes(&v, length: 4, index: bufs.count + k) }
    if groups > 0 { e.dispatchThreadgroups(MTLSize(width: groups, height: height, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)) }
    else { e.dispatchThreads(MTLSize(width: threads, height: height, depth: 1), threadsPerThreadgroup: MTLSize(width: min(tg, threads), height: 1, depth: 1)) }
    e.endEncoding()
}
func product(_ cb: MTLCommandBuffer, _ X: MTLBuffer, _ bk: Bank, r0: Int, nr: Int, xoff: Int = 0) {   // positions r0..r0+nr onto 16 bits, then every core takes piles x a slice of inputs
    enc(cb, pTo16, [X, x16, ex], [Int32(bk.nin), Int32(r0)], threads: 0, tg: 256, groups: nr, offs: [xoff])
    if nr == 1 { let span = argInt("span", 64), ks = bk.nin / span
        enc(cb, pGate1, [bk.w, x16, parts], [Int32(bk.nin), Int32(bk.nout), Int32(r0), Int32(span)], threads: bk.nout, tg: 256, height: ks)
        enc(cb, pAdd1, [parts, pile], [Int32(bk.nout), Int32(r0), Int32(ks)], threads: bk.nout, tg: 256); return }
    let span = bk.nin / 12, ks = 12                                                      // several positions: 8 at a time share each weight read
    let e = cb.makeComputeCommandEncoder()!; e.setComputePipelineState(pGateN)
    for (i, b) in [bk.w, x16, parts].enumerated() { e.setBuffer(b, offset: 0, index: i) }
    for (k, c) in [bk.nin, bk.nout, r0, nr, span].enumerated() { var v = Int32(c); e.setBytes(&v, length: 4, index: 3 + k) }
    e.dispatchThreads(MTLSize(width: bk.nout, height: ks, depth: (nr + 7) / 8), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)); e.endEncoding()
    enc(cb, pAddN, [parts, pile], [Int32(bk.nout), Int32(r0), Int32(ks), Int32(nr * bk.nout)], threads: nr * bk.nout, tg: 256)
}
// THE PILES: pile k is block k's input; pile 24 is the top. Every pile = the token + everything poured by the blocks below it.
// Block 2d is layer d's attention, block 2d+1 its edge part. A block fires on whatever its pile holds now and pours its change.
let STR = PMAX * D, piles = zbuf(25 * STR * 8), newB = (0..<24).map { _ in zbuf(STR * 8) }, lastB = (0..<24).map { _ in zbuf(STR * 8) }
func fire(_ cb: MTLCommandBuffer, _ j: Int, r0: Int, nr: Int) {
    let d = j / 2, r = Int32(r0), xo = j * STR * 8, out = newB[j]
    enc(cb, pZero, [out], [Int32(r0 * D), Int32(nr * D)], threads: nr * D, tg: 256)
    if j % 2 == 0 {
        let bq = banks["qkv.\(d)"]!, bp = banks["proj.\(d)"]!
        product(cb, piles, bq, r0: r0, nr: nr, xoff: xo); enc(cb, pGaugeLog, [x16, ex, LS, LOG2], [r], threads: 0, tg: 32, groups: nr)
        enc(cb, pGauged, [pile, bq.e, ex, LS, bq.b, qkvL[d], EXP2], [2304, r, Int32(nr * 2304)], threads: nr * 2304, tg: 256)
        enc(cb, pHeads, [qkvL[d], q16L[d], eqL[d]], [r, Int32(nr * 36)], threads: nr * 36, tg: 36)
        enc(cb, pAttend, [qkvL[d], q16L[d], eqL[d], att, ARR, LOG2, EXP2], [r], threads: 0, tg: 32, groups: nr * 12)
        product(cb, att, bp, r0: r0, nr: nr)
        enc(cb, pPlain, [pile, bp.e, ex, bp.b, out], [Int32(D), r, Int32(nr * D)], threads: nr * D, tg: 256)
    } else {
        let bf = banks["fc.\(d)"]!, bm = banks["mproj.\(d)"]!
        product(cb, piles, bf, r0: r0, nr: nr, xoff: xo); enc(cb, pGaugeLog, [x16, ex, LS, LOG2], [r], threads: 0, tg: 32, groups: nr)
        enc(cb, pGauged, [pile, bf.e, ex, LS, bf.b, hid, EXP2], [3072, r, Int32(nr * 3072)], threads: nr * 3072, tg: 256)
        enc(cb, pEdge, [hid, PHI], [Int32(r0 * 3072), Int32(nr * 3072)], threads: nr * 3072, tg: 256)
        product(cb, hid, bm, r0: r0, nr: nr)
        enc(cb, pPlain, [pile, bm.e, ex, bm.b, out], [Int32(D), r, Int32(nr * D)], threads: nr * D, tg: 256)
    }
}
func pourOut(_ cb: MTLCommandBuffer, _ j: Int, r0: Int, nr: Int) {
    enc(cb, pPour, [newB[j], lastB[j], piles], [Int32(j), Int32(r0 * D), Int32(nr * D), Int32(STR)], threads: nr * D, tg: 256) }
func readTop(_ cb: MTLCommandBuffer, r0: Int, nr: Int) {
    let bh = banks["head"]!
    product(cb, piles, bh, r0: r0, nr: nr, xoff: 24 * STR * 8); enc(cb, pGaugeLog, [x16, ex, LS, LOG2], [Int32(r0)], threads: 0, tg: 32, groups: nr)
    enc(cb, pGauged, [pile, bh.e, ex, LS, bh.b, lg, EXP2], [Int32(V), Int32(r0), Int32(nr * V)], threads: nr * V, tg: 256)
    enc(cb, pRace, [lg, words], [Int32(V), Int32(r0)], threads: 0, tg: 1024, groups: nr)
}
// THE ORDER IN TIME (order=up|random|together): which blocks fire when. A block is due when its pile may have changed since it
// last fired. up: bottom to top. random: any due block, picked at random, again and again. together: every due block fires at
// once on the piles as they stand, then all pour together (everything everywhere at the same time).
let ORDER = ARG["order"] ?? "up"
var T = Seeded(x: UInt64(argInt("tseed", 1)) &* 0x9E37 &+ 7)
var fires = 0, rounds = 0, roundWords = [Int](), firstRight = [Int](repeating: 0, count: 26), firstSeen = [Int](repeating: 0, count: 26), lanesHist = [Int](repeating: 0, count: 26)
let WATCH = ARG["watch"] != nil
func settle(_ cb0: MTLCommandBuffer, r0: Int, nr: Int) -> MTLCommandBuffer {
    var cb = cb0
    var due = Set(0..<24)
    while !due.isEmpty {
        rounds += 1
        if ORDER == "together" {
            let now = due.sorted(); for j in now { fire(cb, j, r0: r0, nr: nr); fires += 1 }
            for j in now { pourOut(cb, j, r0: r0, nr: nr) }
            due = Set(now.flatMap { ($0 + 1)..<24 })
            if WATCH { readTop(cb, r0: r0 + nr - 1, nr: 1); cb.commit(); cb.waitUntilCompleted()       // what the top says after this round
                roundWords.append(Int(words.contents().bindMemory(to: Int32.self, capacity: PMAX)[r0 + nr - 1])); cb = queue.makeCommandBuffer()! }
        } else {
            let j = ORDER == "random" ? due.randomElement(using: &T)! : due.min()!
            fire(cb, j, r0: r0, nr: nr); pourOut(cb, j, r0: r0, nr: nr); fires += 1
            due.remove(j); for k in (j + 1)..<24 { due.insert(k) }
        }
    }
    return cb
}

let CASES = ARG["all"] != nil ? Array(0..<ref.count) : [CASE]
var nIdentical = 0, report = [String](), totalWords = 0, totalTime = 0.0
for CASE in CASES {
    let ids = ref[CASE]["ids"] as! [Int], prompt = ref[CASE]["prompt"] as! String
    let N = min(WORDS, (ref[CASE]["ref"] as! [Int]).count, PMAX - ids.count), want = Array((ref[CASE]["ref"] as! [Int]).prefix(N)), S = ids.count
    P = S + N
    var seq = ids, gen = [Int]()
    let t0 = Date(); var tGen = Date()
    var p = 0
    while p < S + N - 1 {                                          // the prompt: every token at once; then each new word as soon as it is read
        let nr = p == 0 ? S : 1
        let pp = piles.contents().bindMemory(to: Int64.self, capacity: 25 * STR)
        for q in p..<(p + nr) { for i in 0..<D { let v = wte[seq[q] * D + i] + wpe[q * D + i]; for k in 0...24 { pp[k * STR + q * D + i] = v } } }
        for j in 0..<24 { memset(lastB[j].contents() + p * D * 8, 0, nr * D * 8) }
        roundWords = []
        let cb = settle(queue.makeCommandBuffer()!, r0: p, nr: nr)
        readTop(cb, r0: p + nr - 1, nr: 1)
        cb.commit(); cb.waitUntilCompleted()
        if p == 0 { tGen = Date() }
        p += nr
        if WATCH { let fin = roundWords.last!; var k = roundWords.count; while k > 0 && roundWords[k - 1] == fin { k -= 1 }; firstRight[k + 1] += 1
            firstSeen[roundWords.firstIndex(of: fin)! + 1] += 1; lanesHist[Set(roundWords).count] += 1 }
        let w = wordP[Int(words.contents().bindMemory(to: Int32.self, capacity: P)[p - 1])]; gen.append(w); seq.append(w)
    }
    if ARG["check"] != nil {                                   // the top's raw piles for a few words, found wherever this wiring put them
        let lgp = lg.contents().bindMemory(to: Int64.self, capacity: P * V); var slot = [Int](repeating: 0, count: V); for (j, w) in wordP.enumerated() { slot[w] = j }
        print("  channels \(chan[0]),\(chan[1]),\(chan[2]) | ' the' at slot \(slot[262]) | piles at the last word: " + [262, 3139, 11, 290, 50256].map { String(lgp[(P - 2) * V + slot[$0]]) }.joined(separator: " ")) }
    let wall = Date().timeIntervalSince(t0), wallGen = Date().timeIntervalSince(tGen)
    totalWords += N; totalTime += wall
    let same = gen == want
    if same { nIdentical += 1 } else {
        let j = (0..<N).first { gen[$0] != want[$0] }!
        let mg = (ref[CASE]["margins"] as? [Double]).map { String(format: "%.4f", $0[j]) } ?? "?"
        report.append("  '\(prompt)': first differs at word \(j + 1): \(text([gen[j]])) where the model says \(text([want[j]])) (the model's own lead there: \(mg))") }
    if !QUIET { print("'\(prompt)' -> \(text(gen))  \(same ? "IDENTICAL" : "DIFFERENT")  " + String(format: "%.3f s, %.1f words/s (after the prompt: %.1f)", wall, Double(N) / wall, Double(N - 1) / wallGen)) }
}
print("\(nIdentical) of \(CASES.count) prompts identical to the model" + (SHUF ? " (random wiring, seed \(argInt("seed", 0)))" : ""))
report.forEach { print($0) }
if WATCH { print("round (of 24) from which the top's word is already its final word, how many words:"); for k in 1...24 where firstRight[k] > 0 { print(String(format: "  round %2d: %4d", k, firstRight[k])) } }
if WATCH { print("first round the final word appears at the top:"); for k in 1...24 where firstSeen[k] > 0 { print(String(format: "  round %2d: %4d", k, firstSeen[k])) }
    print("distinct words the top shows over the rounds (lanes a token needs):"); for k in 1...25 where lanesHist[k] > 0 { print(String(format: "  %2d lanes: %4d", k, lanesHist[k])) } }
print("order in time: \(ORDER); \(fires) block firings for \(totalWords) words (bottom to top would be \(24 * (totalWords - CASES.count) + 24 * CASES.count))")
print(String(format: "%d words in %.2f s: %.1f words/s, prompt included", totalWords, totalTime, Double(totalWords) / totalTime))
