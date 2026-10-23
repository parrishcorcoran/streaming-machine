// THE ALL-BASE-2 SOUP, only what moves. Same machine as base2_soup.swift (SPEC.md): every number a whole number in
// base 2, the soup flowing every layer and every position at once. Positions settle from left to right; a settled
// position can never change again, so each step touches only the positions from the first one still moving onward.
// Everything before it keeps its numbers (and each layer keeps its keys and values for the moving positions to read).
//
//   products : each pile's thread sweeps the inputs once; every moving position's input meets the weight (the chip's
//              AND-gate grid); products land in two narrow cells, read once.
//   swiftc -O -o base2_soup2 base2_soup2.swift && ./base2_soup2 all=1 refs=~/streaming-machine/refs_many.json
import Foundation
import Metal

var ARG = [String: String]()
for a in CommandLine.arguments.dropFirst() { let p = a.split(separator: "=", maxSplits: 1).map(String.init); ARG[p[0]] = p.count > 1 ? p[1] : "1" }
func argInt(_ k: String, _ d: Int) -> Int { return Int(ARG[k] ?? "") ?? d }
let CASE = argInt("case", 0), WORDS = argInt("words", 16), MAXSTEPS = argInt("steps", 200), QUIET = ARG["quiet"] != nil
let dir = NSString(string: "~/streaming-machine/base2_gpu/").expandingTildeInPath + "/"
let man = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: dir + "manifest.json"))) as! [String: Any]
let LOG768 = man["log2_768"] as! Int, EPS40 = man["eps40"] as! Int, ALEN = man["arrival_len"] as! Int

let src = """
#include <metal_stdlib>
using namespace metal;
constant long LOG768 = \(LOG768);
constant long EPS40 = \(EPS40);
constant long ALEN = \(ALEN);
#define RC 8
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
kernel void gates(device const short* W [[buffer(0)]], device const int* x16 [[buffer(1)]], device long* pile [[buffer(2)]],
                  constant uint& nin [[buffer(3)]], constant uint& nout [[buffer(4)]], constant uint& r0 [[buffer(5)]], constant uint& nr [[buffer(6)]],
                  uint2 id [[thread_position_in_grid]]) {
    // one thread, one pile, up to RC moving positions: the weights are swept once, each meeting every moving input
    uint j = id.x; if (j >= nout) return; uint rb = r0 + id.y * RC, rn = min((uint)RC, r0 + nr - rb);
    int lo[RC], hi[RC]; for (int r = 0; r < RC; r++) { lo[r] = 0; hi[r] = 0; }
    for (uint i = 0; i < nin; i++) { int w = W[i * nout + j];
        for (uint r = 0; r < rn; r++) { int pr = w * x16[(rb + r) * nin + i]; lo[r] += pr & 0x7FFF; hi[r] += pr >> 15; } }
    for (uint r = 0; r < rn; r++) pile[(rb + r) * nout + j] = ((long)hi[r] << 15) + lo[r]; }
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
kernel void copyl(device const long* a [[buffer(0)]], device long* b [[buffer(1)]], constant uint& off [[buffer(2)]], constant uint& n [[buffer(3)]], uint k [[thread_position_in_grid]]) {
    if (k < n) b[off + k] = a[off + k]; }
kernel void moved(device const long* now [[buffer(0)]], device long* last [[buffer(1)]], device atomic_int* cnt [[buffer(2)]], constant uint& off [[buffer(3)]],
                  constant uint& n [[buffer(4)]], uint k [[thread_position_in_grid]]) {
    if (k >= n) return; uint i = off + k; if (now[i] != last[i]) atomic_fetch_add_explicit(cnt + i / 768, 1, memory_order_relaxed); last[i] = now[i]; }
"""
setvbuf(stdout, nil, _IONBF, 0)
let dev = MTLCreateSystemDefaultDevice()!, queue = dev.makeCommandQueue()!
let lib = try! dev.makeLibrary(source: src, options: MTLCompileOptions())
func pipe(_ n: String) -> MTLComputePipelineState { return try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
let pTo16 = pipe("to16"), pGates = pipe("gates"), pGaugeLog = pipe("gauge_log"), pGauged = pipe("gauged"), pPlain = pipe("plain"), pEdge = pipe("edge")
let pHeads = pipe("heads16"), pAttend = pipe("attend"), pRace = pipe("race"), pCopy = pipe("copyl"), pMoved = pipe("moved")
func loadData(_ name: String) -> Data { return try! Data(contentsOf: URL(fileURLWithPath: dir + name)) }
func bufData(_ d: Data) -> MTLBuffer { return d.withUnsafeBytes { dev.makeBuffer(bytes: $0.baseAddress!, length: d.count, options: .storageModeShared)! } }
func zbuf(_ bytes: Int) -> MTLBuffer { return dev.makeBuffer(length: max(bytes, 8), options: .storageModeShared)! }
func i64(_ name: String) -> [Int64] { let d = loadData(name); return d.withUnsafeBytes { Array($0.bindMemory(to: Int64.self)) } }
struct Bank { let nin, nout: Int; let w, e, b: MTLBuffer }
var banks = [String: Bank]()
for b in man["banks"] as! [[String: Any]] { let n = b["name"] as! String
    banks[n] = Bank(nin: b["nin"] as! Int, nout: b["nout"] as! Int, w: bufData(loadData("\(n).wc")), e: bufData(loadData("\(n).e")), b: bufData(loadData("\(n).b"))) }
let PHI = bufData(loadData("phi.bin")), ARR = bufData(loadData("arrival.bin")), LOG2 = bufData(loadData("log2.bin")), EXP2 = bufData(loadData("exp2.bin"))
let wte = i64("wte.bin"), wpe = i64("wpe.bin")
let ref = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: NSString(string: ARG["refs"] ?? "~/tick-machine/gpt2_dial/reference.json").expandingTildeInPath))) as! [[String: Any]]
let L = 12, D = 768, V = 50257, PMAX = 64
let vocab: [Int: String] = {
    let base = NSString(string: "~/.cache/huggingface/hub/models--gpt2/snapshots").expandingTildeInPath
    let j = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: base + "/" + (try! FileManager.default.contentsOfDirectory(atPath: base)).first! + "/vocab.json"))) as! [String: Int]
    var v = [Int: String](); for (k, i) in j { v[i] = k.replacingOccurrences(of: "Ġ", with: " ").replacingOccurrences(of: "Ċ", with: "\\n") }; return v }()
func text(_ t: [Int]) -> String { return "'" + t.map { vocab[$0] ?? "?" }.joined() + "'" }

let R = (0...L).map { _ in zbuf(PMAX * D * 8) }, last = (0...L).map { _ in zbuf(PMAX * D * 8) }
let qkvL = (0..<L).map { _ in zbuf(PMAX * 2304 * 8) }, q16L = (0..<L).map { _ in zbuf(PMAX * 2304 * 4) }, eqL = (0..<L).map { _ in zbuf(PMAX * 36 * 4) }
let x16 = zbuf(PMAX * 3072 * 4), ex = zbuf(PMAX * 4), pile = zbuf(PMAX * V * 8), LS = zbuf(PMAX * 8)
let att = zbuf(PMAX * D * 8), hid = zbuf(PMAX * 3072 * 8), lg = zbuf(PMAX * V * 8), words = zbuf(PMAX * 4), cnt = zbuf(PMAX * 4)
var P = 0
func enc(_ cb: MTLCommandBuffer, _ ps: MTLComputePipelineState, _ bufs: [MTLBuffer], _ consts: [Int32], threads: Int, tg: Int, groups: Int = 0, height: Int = 1) {
    let e = cb.makeComputeCommandEncoder()!; e.setComputePipelineState(ps)
    for (i, b) in bufs.enumerated() { e.setBuffer(b, offset: 0, index: i) }
    for (k, c) in consts.enumerated() { var v = c; e.setBytes(&v, length: 4, index: bufs.count + k) }
    if groups > 0 { e.dispatchThreadgroups(MTLSize(width: groups, height: height, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)) }
    else { e.dispatchThreads(MTLSize(width: threads, height: height, depth: 1), threadsPerThreadgroup: MTLSize(width: min(tg, threads), height: 1, depth: 1)) }
    e.endEncoding()
}
func product(_ cb: MTLCommandBuffer, _ X: MTLBuffer, _ bk: Bank, r0: Int, nr: Int) {          // rows r0..r0+nr: onto 16 bits, then the gate sweep
    enc(cb, pTo16, [X, x16, ex], [Int32(bk.nin), Int32(r0)], threads: 0, tg: 256, groups: nr)
    enc(cb, pGates, [bk.w, x16, pile], [Int32(bk.nin), Int32(bk.nout), Int32(r0), Int32(nr)], threads: bk.nout, tg: 256, height: (nr + 7) / 8)
}
func layer(_ cb: MTLCommandBuffer, _ d: Int, r0: Int, nr: Int) {
    let bq = banks["qkv.\(d)"]!, bp = banks["proj.\(d)"]!, bf = banks["fc.\(d)"]!, bm = banks["mproj.\(d)"]!
    let r = Int32(r0)
    product(cb, R[d], bq, r0: r0, nr: nr); enc(cb, pGaugeLog, [x16, ex, LS, LOG2], [r], threads: 0, tg: 32, groups: nr)
    enc(cb, pGauged, [pile, bq.e, ex, LS, bq.b, qkvL[d], EXP2], [2304, r, Int32(nr * 2304)], threads: nr * 2304, tg: 256)
    enc(cb, pHeads, [qkvL[d], q16L[d], eqL[d]], [r, Int32(nr * 36)], threads: nr * 36, tg: 36)
    enc(cb, pAttend, [qkvL[d], q16L[d], eqL[d], att, ARR, LOG2, EXP2], [r], threads: 0, tg: 32, groups: nr * 12)
    enc(cb, pCopy, [R[d], R[d + 1]], [Int32(r0 * D), Int32(nr * D)], threads: nr * D, tg: 256)
    product(cb, att, bp, r0: r0, nr: nr)
    enc(cb, pPlain, [pile, bp.e, ex, bp.b, R[d + 1]], [Int32(D), r, Int32(nr * D)], threads: nr * D, tg: 256)
    product(cb, R[d + 1], bf, r0: r0, nr: nr); enc(cb, pGaugeLog, [x16, ex, LS, LOG2], [r], threads: 0, tg: 32, groups: nr)
    enc(cb, pGauged, [pile, bf.e, ex, LS, bf.b, hid, EXP2], [3072, r, Int32(nr * 3072)], threads: nr * 3072, tg: 256)
    enc(cb, pEdge, [hid, PHI], [Int32(r0 * 3072), Int32(nr * 3072)], threads: nr * 3072, tg: 256)
    product(cb, hid, bm, r0: r0, nr: nr)
    enc(cb, pPlain, [pile, bm.e, ex, bm.b, R[d + 1]], [Int32(D), r, Int32(nr * D)], threads: nr * D, tg: 256)
}
func readTop(_ cb: MTLCommandBuffer, r0: Int, nr: Int) {
    let bh = banks["head"]!
    product(cb, R[L], bh, r0: r0, nr: nr); enc(cb, pGaugeLog, [x16, ex, LS, LOG2], [Int32(r0)], threads: 0, tg: 32, groups: nr)
    enc(cb, pGauged, [pile, bh.e, ex, LS, bh.b, lg, EXP2], [Int32(V), Int32(r0), Int32(nr * V)], threads: nr * V, tg: 256)
    enc(cb, pRace, [lg, words], [Int32(V), Int32(r0)], threads: 0, tg: 1024, groups: nr)
}

let CASES = ARG["all"] != nil ? Array(0..<ref.count) : [CASE]
var nIdentical = 0, report = [String](), totalWords = 0, totalTime = 0.0, totalSteps = 0
for CASE in CASES {
    let ids = ref[CASE]["ids"] as! [Int], prompt = ref[CASE]["prompt"] as! String
    let N = min(WORDS, (ref[CASE]["ref"] as! [Int]).count, PMAX - ids.count), want = Array((ref[CASE]["ref"] as! [Int]).prefix(N)), S = ids.count
    P = S + N
    for k in 0...L { memset(R[k].contents(), 0, PMAX * D * 8); memset(last[k].contents(), 0, PMAX * D * 8) }
    if !QUIET { print("'\(prompt)' + \(N) words\nmodel : \(text(want))") }
    var T = ids + [Int](repeating: ids.last!, count: N), first = 0, step = 0, rest: Int? = nil, rowsDone = 0
    let t0 = Date()
    while step < MAXSTEPS {
        step += 1
        let nr = P - first
        let r0p = R[0].contents().bindMemory(to: Int64.self, capacity: P * D)
        for p in first..<P { for i in 0..<D { r0p[p * D + i] = wte[T[p] * D + i] + wpe[p * D + i] } }
        memset(cnt.contents(), 0, PMAX * 4)
        let cb = queue.makeCommandBuffer()!
        for d in 0..<L { layer(cb, d, r0: first, nr: nr) }
        for k in 1...L { enc(cb, pMoved, [R[k], last[k], cnt], [Int32(first * D), Int32(nr * D)], threads: nr * D, tg: 256) }
        let h0 = max(first, S - 1); readTop(cb, r0: h0, nr: P - 1 - h0)
        cb.commit(); cb.waitUntilCompleted()
        rowsDone += nr
        let wd = words.contents().bindMemory(to: Int32.self, capacity: P), mv = cnt.contents().bindMemory(to: Int32.self, capacity: PMAX)
        var newT = T; for j in 0..<N where S - 1 + j >= h0 { newT[S + j] = Int(wd[S - 1 + j]) }
        let changedRows = (0..<P).filter { newT[$0] != T[$0] }; T = newT
        let movedRows = (first..<P).filter { mv[$0] > 0 }
        // the first position still moving: everything before it has settled for good
        first = min(changedRows.first ?? P, movedRows.first ?? P)
        let right = (0..<N).filter { T[S + $0] == want[$0] }.count
        if !QUIET { print(String(format: "  step %3d: %2d/%d right, moving from position %2d (%2d positions computed)   ", step, right, N, first, nr) + text(Array(T[S...]))) }
        if first >= P { rest = step; break }
    }
    let wall = Date().timeIntervalSince(t0)
    totalWords += N; totalTime += wall; totalSteps += step
    let same = Array(T[S...]) == want
    if same { nIdentical += 1 } else {
        let j = (0..<N).first { T[S + $0] != want[$0] }!
        let mg = (ref[CASE]["margins"] as? [Double]).map { String(format: "%.4f", $0[j]) } ?? "?"
        report.append("  '\(prompt)': first differs at word \(j + 1): \(text([T[S + j]])) where the model says \(text([want[j]])) (the model's own lead there: \(mg))") }
    if !QUIET { print("\n\(rest != nil ? "came to rest at step \(rest!)" : "still moving"); \(same ? "IDENTICAL" : "DIFFERENT") to the model; " +
                      String(format: "%.3f s, %.0f ms a step, %.1f words/s; %d position-steps computed (all positions every step: %d)\n", wall, wall / Double(step) * 1000, Double(N) / wall, rowsDone, step * P)) }
}
print("\(nIdentical) of \(CASES.count) prompts identical to the model")
report.forEach { print($0) }
print(String(format: "%d words in %.2f s: %.1f words/s, %.1f ms a step", totalWords, totalTime, Double(totalWords) / totalTime, totalTime / Double(totalSteps) * 1000))
