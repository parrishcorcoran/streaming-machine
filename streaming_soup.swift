// THE SOUP. GPT-2 on circles, base 2, on the M4's GPU: every layer, every position, every step, all at once.
//
// The prompt and the next N words are all in the machine from the start. Every step every layer fires, each taking
// what is below it now (flow=now, bottom to top; flow=shuffle, a random order; flow=lagged, what was below it last
// step), for every position at once. Every future position's word is whatever the top says at the position before
// it, right now, re-read every step. Nothing waits and nothing is reset; wrong words are overwritten as the flow
// corrects itself. Each position only looks backward, so the soup can only come to rest on the model's own text.
//
//   weights   : 16-bit edges on a 65,536 circle, poured in base 2 (each set binary place adds the input, shifted);
//               zero places, zero edges, zero inputs skipped. Exact whole-number piles.
//   the bend  : the lap's edge, a looked-up share.     layer norm, attention: small float laws, back onto dials.
//
//   lap=lanes|apart|once   lanes (default): one GPU lane per position, every lane meets the same edge at once;
//                          apart: each position reads the weights by itself; once: 16 positions per thread
//   flow=now|shuffle|lagged  base=2|8|4096  digits=SEED  case=0|1|2  words=N  init=noise
//   channels=SEED neurons=SEED heads=SEED landings=SEED perm=SEED swap=A,B
//   swiftc -O -o streaming_soup streaming_soup.swift && ./streaming_soup words=16
import Foundation
import Metal

var ARG = [String: String]()
for a in CommandLine.arguments.dropFirst() { let p = a.split(separator: "=", maxSplits: 1).map(String.init); ARG[p[0]] = p.count > 1 ? p[1] : "1" }
func argInt(_ k: String, _ d: Int) -> Int { return Int(ARG[k] ?? "") ?? d }
func seed(_ k: String) -> Int? { return ARG[k].flatMap { Int($0) } }
let FLOW = ARG["flow"] ?? "now", NOISE = ARG["init"] == "noise", CASE = argInt("case", 0), WORDS = argInt("words", 16), MAXSTEPS = argInt("steps", 400)
let BASE = argInt("base", 2), BASEBITS = BASE == 2 ? 1 : (BASE == 8 ? 3 : 12), NDIGITS = (15 + BASEBITS - 1) / BASEBITS

let src = """
#include <metal_stdlib>
using namespace metal;
#define SUB 8
#define PILES 32
constant int ORD[] = { DIGIT_ORDER };
inline long fire(int x, int c, thread uint& n) {
    if (x == 0) return 0;
    int v = c < 32768 ? c : c - 65536; int m = v < 0 ? -v : v; long xv = x;
    if (m == 0) return 0;
#if BASEBITS == 12
    long acc = ((xv * (m >> 12)) << 12) + xv * (m & 4095); n += 2;
#elif BASEBITS == 1 && !DIGITS_SHUFFLED
    long acc = 0; uint mm = (uint)m;
    while (mm) { int j = ctz(mm); acc += xv << j; n++; mm &= mm - 1; }
#else
    long acc = 0;
    for (int o = 0; o < NDIGITS; o++) { int j = ORD[o] * BASEBITS; int d = (m >> j) & ((1 << BASEBITS) - 1); if (d == 0) continue;
        long p = 0; for (int k = 0; k < d; k++) p += xv; n += d; acc += p << j; }
#endif
    return v < 0 ? -acc : acc;
}
kernel void lap(device const ushort* edges [[buffer(0)]], device const int* x [[buffer(1)]], device const long* bias [[buffer(2)]], device long* out [[buffer(3)]],
                device atomic_uint* adds [[buffer(4)]], constant uint& nin [[buffer(5)]], constant uint& nout [[buffer(6)]],
                uint2 t2 [[thread_position_in_threadgroup]], uint2 g [[threadgroup_position_in_grid]]) {
    threadgroup long part[SUB][PILES];
    uint t = t2.x; uint pile = g.x * PILES + (t % PILES), sub = t / PILES, p = g.y;
    device const int* xp = x + p * nin;
    long acc = 0; uint n = 0;
    if (pile < nout) for (uint i = sub; i < nin; i += SUB) acc += fire(xp[i], (int)edges[i * nout + pile], n);
    atomic_fetch_add_explicit(adds, n, memory_order_relaxed);
    part[sub][t % PILES] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sub == 0 && pile < nout) { long s = bias[pile]; for (uint k = 0; k < SUB; k++) s += part[k][t % PILES]; out[p * nout + pile] = s; }
}

#define PCH 16
#define SUBP 4
kernel void lapP(device const ushort* edges [[buffer(0)]], device const int* x [[buffer(1)]], device const long* bias [[buffer(2)]], device long* out [[buffer(3)]],
                 device atomic_uint* adds [[buffer(4)]], constant uint& nin [[buffer(5)]], constant uint& nout [[buffer(6)]], constant uint& np [[buffer(7)]],
                 uint2 t2 [[thread_position_in_threadgroup]], uint2 g [[threadgroup_position_in_grid]]) {
    // one pile, up to PCH positions at once: each edge is read once and poured into every position's pile
    threadgroup long part[SUBP][PILES][PCH];
    uint t = t2.x, pile = g.x * PILES + (t % PILES), sub = t / PILES, p0 = g.y * PCH;
    uint npc = min((uint)PCH, np - p0);
    long acc[PCH]; for (uint q = 0; q < PCH; q++) acc[q] = 0; uint n = 0;
    if (pile < nout) for (uint i = sub; i < nin; i += SUBP) {
        int c = (int)edges[i * nout + pile]; int v = c < 32768 ? c : c - 65536; int m = v < 0 ? -v : v;
        if (m == 0) continue;
        for (uint q = 0; q < npc; q++) {
            long xv = x[(p0 + q) * nin + i]; if (xv == 0) continue;
            if (v < 0) xv = -xv;
#if BASEBITS == 12
            acc[q] += ((xv * (m >> 12)) << 12) + xv * (m & 4095); n += 2;
#elif BASEBITS == 1 && !DIGITS_SHUFFLED
            uint mm = (uint)m; while (mm) { int j = ctz(mm); acc[q] += xv << j; n++; mm &= mm - 1; }
#else
            for (int o = 0; o < NDIGITS; o++) { int j = ORD[o] * BASEBITS; int d = (m >> j) & ((1 << BASEBITS) - 1); if (d == 0) continue;
                long pp = 0; for (int k = 0; k < d; k++) pp += xv; n += d; acc[q] += pp << j; }
#endif
        }
    }
    atomic_fetch_add_explicit(adds, n, memory_order_relaxed);
    for (uint q = 0; q < PCH; q++) part[sub][t % PILES][q] = acc[q];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sub == 0 && pile < nout) for (uint q = 0; q < npc; q++) { long s = bias[pile]; for (uint k = 0; k < SUBP; k++) s += part[k][t % PILES][q]; out[(p0 + q) * nout + pile] = s; }
}

kernel void to_ticks_t(device const float* y [[buffer(0)]], device const float* uin [[buffer(1)]], device const uint* order [[buffer(2)]], device int* xt [[buffer(3)]],
                       constant uint& np [[buffer(4)]], constant uint& nin [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    // the inputs laid out input-major: all positions of input i side by side, padded to 32
    if (i >= nin * 32) return; uint c = i / 32, p = i % 32;
    xt[i] = p < np ? (int)rint(y[p * nin + order[c]] / uin[c]) : 0; }
kernel void lapW(device const ushort* edges [[buffer(0)]], device const int* xt [[buffer(1)]], device const long* bias [[buffer(2)]], device long* out [[buffer(3)]],
                 device atomic_uint* adds [[buffer(4)]], constant uint& nin [[buffer(5)]], constant uint& nout [[buffer(6)]], constant uint& np [[buffer(7)]],
                 uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint g [[threadgroup_position_in_grid]]) {
    // one simdgroup = one pile; one lane = one position. Every lane meets the same edge at the same moment.
    uint pile = g * 8 + sg; if (pile >= nout) return;
    long acc = 0; uint n = 0;
    for (uint i = 0; i < nin; i++) {
        int c = (int)edges[i * nout + pile]; int v = c < 32768 ? c : c - 65536; int m = v < 0 ? -v : v;
        if (m == 0) continue;
        long xv = xt[i * 32 + lane]; if (v < 0) xv = -xv;
#if BASEBITS == 12
        acc += ((xv * (m >> 12)) << 12) + xv * (m & 4095); n += 2;
#elif BASEBITS == 1 && !DIGITS_SHUFFLED
        uint mm = (uint)m; while (mm) { int j = ctz(mm); acc += xv << j; mm &= mm - 1; n++; }
#else
        for (int o = 0; o < NDIGITS; o++) { int j = ORD[o] * BASEBITS; int d = (m >> j) & ((1 << BASEBITS) - 1); if (d == 0) continue;
            long pp = 0; for (int k = 0; k < d; k++) pp += xv; n += d; acc += pp << j; }
#endif
    }
    if (lane < np) { out[lane * nout + pile] = acc + bias[pile]; atomic_fetch_add_explicit(adds, n, memory_order_relaxed); }
}
kernel void gauge_dial(device const long* level [[buffer(0)]], device const float* gauge [[buffer(1)]], device const float* uout [[buffer(2)]], device float* y [[buffer(3)]],
                       constant uint& n [[buffer(4)]], constant uint& nout [[buffer(5)]], uint j [[thread_position_in_grid]]) {
    if (j >= n) return; uint c = j % nout; long v = level[j]; long hi = v >> 20, lo = v & 0xFFFFF;
    float r = ((float)hi * 1048576.0f + (float)lo) * gauge[c];
    y[j] = uout[c] > 0 ? rint(r / uout[c]) * uout[c] : r;
}
kernel void to_ticks(device const float* y [[buffer(0)]], device const float* uin [[buffer(1)]], device const uint* order [[buffer(2)]], device int* x [[buffer(3)]],
                     constant uint& n [[buffer(4)]], constant uint& nin [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if (i < n) { uint c = i % nin; x[i] = (int)rint(y[(i - c) + order[c]] / uin[c]); } }
kernel void edge_bend(device float* y [[buffer(0)]], device const float* share [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return; float v = y[i]; float f = rint((v + 16.0f) / 32.0f * 1048575.0f);
    float s = f < 0.0f ? 0.0f : (f > 1048575.0f ? 1.0f : share[(uint)f]); y[i] = v * s; }
kernel void resadd(device int* res [[buffer(0)]], device const float* y [[buffer(1)]], device const float* ures [[buffer(2)]], constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < n) res[i] += (int)rint(y[i] / ures[i % 768]); }
kernel void copy_res(device const int* a [[buffer(0)]], device int* b [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) { if (i < n) b[i] = a[i]; }
kernel void count_moved(device const int* now [[buffer(0)]], device int* last [[buffer(1)]], device atomic_int* moved [[buffer(2)]], constant uint& k [[buffer(3)]],
                        constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return; if (now[i] != last[i]) atomic_fetch_add_explicit(&moved[k], 1, memory_order_relaxed); last[i] = now[i]; }
kernel void water(device const int* res [[buffer(0)]], device const float* ures [[buffer(1)]], device const float* w [[buffer(2)]], device const float* b [[buffer(3)]],
                  device float* y [[buffer(4)]], uint t [[thread_position_in_threadgroup]], uint p [[threadgroup_position_in_grid]],
                  uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float ps[8], pq[8]; float v[3]; float s = 0, q = 0; uint o = p * 768;
    for (int k = 0; k < 3; k++) { v[k] = (float)res[o + t + 256 * k] * ures[t + 256 * k]; s += v[k]; }
    s = simd_sum(s); if (lane == 0) ps[sg] = s; threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = (ps[0]+ps[1]+ps[2]+ps[3]+ps[4]+ps[5]+ps[6]+ps[7]) / 768.0f;
    for (int k = 0; k < 3; k++) q += (v[k] - mean) * (v[k] - mean);
    q = simd_sum(q); if (lane == 0) pq[sg] = q; threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rsqrt((pq[0]+pq[1]+pq[2]+pq[3]+pq[4]+pq[5]+pq[6]+pq[7]) / 768.0f + 1e-5f);
    for (int k = 0; k < 3; k++) { uint i = t + 256 * k; y[o + i] = (v[k] - mean) * inv * w[i] + b[i]; }
}
kernel void attn(device const float* qkv [[buffer(0)]], device float* o [[buffer(1)]], uint id [[thread_position_in_grid]]) {
    uint h = id % 12, p = id / 12; float sc[128]; float mx = -1e30f;             // every position's query against every earlier key
    for (uint s = 0; s <= p; s++) { float d = 0; for (uint k = 0; k < 64; k++) d += qkv[p*2304 + h*64 + k] * qkv[s*2304 + 768 + h*64 + k]; sc[s] = d / 8.0f; mx = max(mx, sc[s]); }
    float z = 0; for (uint s = 0; s <= p; s++) { sc[s] = exp(sc[s] - mx); z += sc[s]; }
    for (uint k = 0; k < 64; k++) { float a = 0; for (uint s = 0; s <= p; s++) a += sc[s] / z * qkv[s*2304 + 1536 + h*64 + k]; o[p*768 + h*64 + k] = a; }
}
"""
setvbuf(stdout, nil, _IONBF, 0)
let dir = NSString(string: "~/tick-machine/gpt2_dial/optical/").expandingTildeInPath + "/"
let dev = MTLCreateSystemDefaultDevice()!, queue = dev.makeCommandQueue()!
struct Rng { var s: UInt64; mutating func next() -> UInt64 { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s } }
func shuffled(_ n: Int, _ sd: Int?) -> [Int] {
    var p = Array(0..<n); guard let sd = sd else { return p }
    var r = Rng(s: UInt64(sd) &+ 0x9E3779B97F4A7C15); for i in stride(from: n - 1, to: 0, by: -1) { p.swapAt(i, Int(r.next() % UInt64(i + 1))) }; return p
}
let digitOrder = shuffled(NDIGITS, seed("digits"))
let copts = MTLCompileOptions()
copts.preprocessorMacros = ["BASEBITS": BASEBITS as NSNumber, "NDIGITS": NDIGITS as NSNumber, "DIGITS_SHUFFLED": (ARG["digits"] != nil ? 1 : 0) as NSNumber,
                            "DIGIT_ORDER": digitOrder.map(String.init).joined(separator: ",") as NSString]
let lib = try! dev.makeLibrary(source: src, options: copts)
func pipe(_ n: String) -> MTLComputePipelineState { return try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
let pLap = pipe("lap"), pLapP = pipe("lapP"), pLapW = pipe("lapW"), pTicksT = pipe("to_ticks_t"), LAYOUT = ARG["lap"] ?? "lanes", pGauge = pipe("gauge_dial"), pTicks = pipe("to_ticks"), pBend = pipe("edge_bend"), pRes = pipe("resadd")
let pWater = pipe("water"), pAttn = pipe("attn"), pCopy = pipe("copy_res"), pCount = pipe("count_moved")
func load<T>(_ name: String, _ t: T.Type) -> [T] { let d = try! Data(contentsOf: URL(fileURLWithPath: dir + name)); return d.withUnsafeBytes { Array($0.bindMemory(to: T.self)) } }
func buf<T>(_ a: [T]) -> MTLBuffer { return dev.makeBuffer(bytes: a, length: max(a.count, 1) * MemoryLayout<T>.stride, options: .storageModeShared)! }
func zbuf(_ bytes: Int) -> MTLBuffer { return dev.makeBuffer(length: max(bytes, 4), options: .storageModeShared)! }

// ---- reference, positions
let ref = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: dir + "../reference.json"))) as! [[String: Any]]
let ids = ref[CASE]["ids"] as! [Int], refAll = ref[CASE]["ref"] as! [Int], prompt = ref[CASE]["prompt"] as! String
let N = min(WORDS, refAll.count), want = Array(refAll.prefix(N)), S = ids.count, P = S + N
let L = 12, D = 768, V = 50257

// ---- weights, with every abelian shuffle applied consistently
let pc = shuffled(D, seed("channels")), pn = shuffled(3072, seed("neurons")), ph = shuffled(12, seed("heads"))
let headCols: [Int] = (0..<3).flatMap { b in (0..<12).flatMap { h in (0..<64).map { b * 768 + ph[h] * 64 + $0 } } }
let headRows: [Int] = (0..<12).flatMap { h in (0..<64).map { ph[h] * 64 + $0 } }
func pick<T>(_ a: [T], _ idx: [Int]) -> [T] { return idx.map { a[$0] } }
func rows(_ e: [UInt16], _ nout: Int, _ idx: [Int]) -> [UInt16] { var o = [UInt16](); o.reserveCapacity(e.count); for i in idx { o += e[(i * nout)..<((i + 1) * nout)] }; return o }
func cols(_ e: [UInt16], _ nin: Int, _ nout: Int, _ idx: [Int]) -> [UInt16] { var o = [UInt16](repeating: 0, count: e.count); for i in 0..<nin { for (j, c) in idx.enumerated() { o[i * nout + j] = e[i * nout + c] } }; return o }
struct Bank { let nin, nout: Int; let edges, gauge, uIn, uOut, bias, order, x, level, y, adds: MTLBuffer }
var banks = [String: Bank](), bytes = 0
let man = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: dir + "manifest.json"))) as! [[String: Any]]
for (bi, b) in man.enumerated() {
    let n = b["name"] as! String, nin = b["nin"] as! Int, nout = b["nout"] as! Int, kind = n.split(separator: ".")[0]
    var e = load("\(n).edges.bin", UInt16.self), uIn = load("\(n).u_in.bin", Float.self)
    var gauge = load("\(n).gauge.bin", Float.self), uOut = load("\(n).u_out.bin", Float.self), bias = load("\(n).bias.bin", Int64.self)
    var rIdx = Array(0..<nin), cIdx = Array(0..<nout)
    if nin == D && kind != "proj" && kind != "mproj" { rIdx = pc }
    if kind == "proj" { rIdx = headRows; cIdx = pc }
    if kind == "mproj" { rIdx = pn; cIdx = pc }
    if kind == "fc" { cIdx = pn }
    if kind == "qkv" { cIdx = headCols }
    if rIdx != Array(0..<nin) || cIdx != Array(0..<nout) { e = cols(rows(e, nout, rIdx), nin, nout, cIdx) }
    uIn = pick(uIn, rIdx); gauge = pick(gauge, cIdx); uOut = pick(uOut, cIdx); bias = pick(bias, cIdx)
    let land = shuffled(nin, seed("landings").map { $0 + bi })
    if seed("landings") != nil { e = rows(e, nout, land); uIn = pick(uIn, land) }
    bytes += e.count * 2
    banks[n] = Bank(nin: nin, nout: nout, edges: buf(e), gauge: buf(gauge), uIn: buf(uIn), uOut: buf(uOut), bias: buf(bias), order: buf(land.map { UInt32($0) }),
                    x: zbuf(max(P, 32) * nin * 4), level: zbuf(P * nout * 8), y: zbuf(P * nout * 4), adds: zbuf(4))
}
let uRes = buf(pick(load("u_res.bin", Float.self), pc))
let wteAll = load("wte.ticks.bin", Int32.self), wpeAll = load("wpe.ticks.bin", Int32.self)
func lnb(_ n: String) -> MTLBuffer { return buf(pick(load(n, Float.self), pc)) }
let ln1w = (0..<L).map { lnb("ln.\($0).ln_1.weight.bin") }, ln1b = (0..<L).map { lnb("ln.\($0).ln_1.bias.bin") }
let ln2w = (0..<L).map { lnb("ln.\($0).ln_2.weight.bin") }, ln2b = (0..<L).map { lnb("ln.\($0).ln_2.bias.bin") }
let lnfw = lnb("ln_f.weight.bin"), lnfb = lnb("ln_f.bias.bin")
let share = buf((0..<(1 << 20)).map { i -> Float in let p = -16.0 + 32.0 * Double(i) / Double((1 << 20) - 1); return Float(0.5 * (1.0 + erf(p / 2.0.squareRoot()))) })
var wiring = Array(0..<L)
if let s = seed("perm") { wiring = shuffled(L, s) }
if let sw = ARG["swap"] { let ab = sw.split(separator: ",").compactMap { Int($0) }; if ab.count == 2 { wiring.swapAt(ab[0], ab[1]) } }

// ---- the soup's state: every boundary holds every position
let res = (0...L).map { _ in zbuf(P * D * 4) }, snap = (0...L).map { _ in zbuf(P * D * 4) }, last = (0...L).map { _ in zbuf(P * D * 4) }
let work = zbuf(P * D * 4), nrm = zbuf(P * D * 4), att = zbuf(P * D * 4), moved = zbuf((L + 1) * 4), zeros = buf([Float](repeating: 0, count: V))
func enc(_ cb: MTLCommandBuffer, _ ps: MTLComputePipelineState, _ bufs: [MTLBuffer], _ consts: [UInt32], threads: Int, tg: Int, groups: Int = 0, height: Int = 1) {
    let e = cb.makeComputeCommandEncoder()!; e.setComputePipelineState(ps)
    for (i, b) in bufs.enumerated() { e.setBuffer(b, offset: 0, index: i) }
    for (k, c) in consts.enumerated() { var v = c; e.setBytes(&v, length: 4, index: bufs.count + k) }
    if groups > 0 { e.dispatchThreadgroups(MTLSize(width: groups, height: height, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)) }
    else { e.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)) }
    e.endEncoding()
}
var adds = 0
func lapOf(_ cb: MTLCommandBuffer, _ bk: Bank, from y: MTLBuffer, dial: Bool) {           // every position's lap through one bank
    if LAYOUT == "lanes" {
        enc(cb, pTicksT, [y, bk.uIn, bk.order, bk.x], [UInt32(P), UInt32(bk.nin)], threads: 32 * bk.nin, tg: 256)
        enc(cb, pLapW, [bk.edges, bk.x, bk.bias, bk.level, bk.adds], [UInt32(bk.nin), UInt32(bk.nout), UInt32(P)], threads: 0, tg: 256, groups: (bk.nout + 7) / 8)
    } else { enc(cb, pTicks, [y, bk.uIn, bk.order, bk.x], [UInt32(P * bk.nin), UInt32(bk.nin)], threads: P * bk.nin, tg: 256) }
    if LAYOUT == "once" { enc(cb, pLapP, [bk.edges, bk.x, bk.bias, bk.level, bk.adds], [UInt32(bk.nin), UInt32(bk.nout), UInt32(P)], threads: 0, tg: 128, groups: (bk.nout + 31) / 32, height: (P + 15) / 16) }
    else if LAYOUT == "apart" { enc(cb, pLap, [bk.edges, bk.x, bk.bias, bk.level, bk.adds], [UInt32(bk.nin), UInt32(bk.nout)], threads: 0, tg: 256, groups: (bk.nout + 31) / 32, height: P) }
    enc(cb, pGauge, [bk.level, bk.gauge, dial ? bk.uOut : zeros, bk.y], [UInt32(P * bk.nout), UInt32(bk.nout)], threads: P * bk.nout, tg: 256)
}
func layer(_ cb: MTLCommandBuffer, depth d: Int, input: MTLBuffer) {                      // one layer, every position: input -> res[d+1]
    let w = wiring[d], bq = banks["qkv.\(w)"]!, bp = banks["proj.\(w)"]!, bf = banks["fc.\(w)"]!, bm = banks["mproj.\(w)"]!, n = UInt32(P * D)
    enc(cb, pCopy, [input, work], [n], threads: P * D, tg: 256)
    enc(cb, pWater, [work, uRes, ln1w[w], ln1b[w], nrm], [], threads: 0, tg: 256, groups: P)
    lapOf(cb, bq, from: nrm, dial: true)
    enc(cb, pAttn, [bq.y, att], [], threads: 12 * P, tg: 12)
    lapOf(cb, bp, from: att, dial: true); enc(cb, pRes, [work, bp.y, uRes], [n], threads: P * D, tg: 256)
    enc(cb, pWater, [work, uRes, ln2w[w], ln2b[w], nrm], [], threads: 0, tg: 256, groups: P)
    lapOf(cb, bf, from: nrm, dial: true); enc(cb, pBend, [bf.y, share], [UInt32(P * bf.nout)], threads: P * bf.nout, tg: 256)
    lapOf(cb, bm, from: bf.y, dial: true); enc(cb, pRes, [work, bm.y, uRes], [n], threads: P * D, tg: 256)
    enc(cb, pCopy, [work, res[d + 1]], [n], threads: P * D, tg: 256)
}
let bh = banks["head"]!
let vocab: [Int: String] = {
    let base = NSString(string: "~/.cache/huggingface/hub/models--gpt2/snapshots").expandingTildeInPath
    let j = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: base + "/" + (try! FileManager.default.contentsOfDirectory(atPath: base)).first! + "/vocab.json"))) as! [String: Int]
    var v = [Int: String](); for (k, i) in j { v[i] = k.replacingOccurrences(of: "Ġ", with: " ").replacingOccurrences(of: "Ċ", with: "\\n") }; return v }()
func text(_ t: [Int]) -> String { return "'" + t.map { vocab[$0] ?? "?" }.joined() + "'" }

// ---- run
var T = ids + [Int](repeating: ids.last!, count: N)             // the future starts as copies of the last word
var rng = Rng(s: 4242)
for k in 1...L {
    let r = res[k].contents().bindMemory(to: Int32.self, capacity: P * D)
    for i in 0..<(P * D) { r[i] = NOISE ? Int32(Int(rng.next() % 2_000_001) - 1_000_000) : 0 }
}
var on = ["each layer reads \(FLOW == "lagged" ? "last step's" : "what is below it now")" + (FLOW == "shuffle" ? ", random order" : "")]
if NOISE { on.append("starts on junk") }; if seed("perm") != nil || ARG["swap"] != nil { on.append("wiring \(wiring)") }
for k in ["channels", "neurons", "heads", "landings", "digits"] where seed(k) != nil { on.append("\(k) shuffled") }
print("the soup on \(dev.name): \(bytes / 1_000_000) MB of 16-bit edges poured in base \(BASE), zeros skipped; \(P) positions x \(L) layers, all at once")
print("'\(prompt)' + \(N) words; \(on.joined(separator: ", "))")
print("model : \(text(want))\n")
var firstRight = [Int?](repeating: nil, count: N), rest: Int? = nil
let t0 = Date(); var step = 0
while step < MAXSTEPS {
    step += 1
    let r0 = res[0].contents().bindMemory(to: Int32.self, capacity: P * D)
    for p in 0..<P { for i in 0..<D { r0[p * D + i] = wteAll[T[p] * D + pc[i]] + wpeAll[p * D + pc[i]] } }
    let cb = queue.makeCommandBuffer()!
    memset(moved.contents(), 0, (L + 1) * 4)
    if FLOW == "lagged" {
        for k in 0...L { enc(cb, pCopy, [res[k], snap[k]], [UInt32(P * D)], threads: P * D, tg: 256) }
        for d in 0..<L { layer(cb, depth: d, input: snap[d]) }
    } else {
        var order = Array(0..<L); if FLOW == "shuffle" { for i in stride(from: L - 1, to: 0, by: -1) { order.swapAt(i, Int(rng.next() % UInt64(i + 1))) } }
        for d in order { layer(cb, depth: d, input: res[d]) }
    }
    for k in 0...L { enc(cb, pCount, [res[k], last[k], moved], [UInt32(k), UInt32(P * D)], threads: P * D, tg: 256) }
    enc(cb, pWater, [res[L], uRes, lnfw, lnfb, nrm], [], threads: 0, tg: 256, groups: P); lapOf(cb, bh, from: nrm, dial: false)
    cb.commit(); cb.waitUntilCompleted()
    for (_, bk) in banks { let a = bk.adds.contents().bindMemory(to: UInt32.self, capacity: 1); adds += Int(a[0]); a[0] = 0 }
    let mv = Array(UnsafeBufferPointer(start: moved.contents().bindMemory(to: Int32.self, capacity: L + 1), count: L + 1))
    let lg = bh.y.contents().bindMemory(to: Float.self, capacity: P * V)
    var newT = ids
    for j in 0..<N { let o = (S - 1 + j) * V; var best = 0; for t in 1..<V where lg[o + t] > lg[o + best] { best = t }; newT.append(best) }
    let changed = zip(newT, T).filter { $0 != $1 }.count; T = newT
    let right = (0..<N).map { T[S + $0] == want[$0] }
    for j in 0..<N { if right[j] { if firstRight[j] == nil { firstRight[j] = step } } else { firstRight[j] = nil } }
    let movingNow = mv[1...].reduce(0) { $0 + Int($1) }
    print(String(format: "  step %3d: %2d/%d right, %2d changed, %6d positions moving   ", step, right.filter { $0 }.count, N, changed, movingNow) + text(Array(T[S...])))
    if movingNow == 0 && changed == 0 { rest = step; break }
}
let wall = Date().timeIntervalSince(t0)
print("\n\(rest != nil ? "came to rest at step \(rest!)" : "still moving after \(MAXSTEPS) steps"); \(Array(T[S...]) == want ? "IDENTICAL" : "DIFFERENT") to the model")
print("each word right from step: \(firstRight.map { $0 == nil ? "-" : String($0!) }.joined(separator: " "))")
print(String(format: "%.2f s (%.0f ms a step), %.1f words/s; %@ adds in base %d", wall, wall / Double(step) * 1000, Double(N) / wall, NumberFormatter.localizedString(from: adds as NSNumber, number: .decimal), BASE))
