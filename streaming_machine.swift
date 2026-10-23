// THE STREAMING MACHINE. GPT-2 on circles, run on the M4's GPU, every layer live every step.
//
//   the state between layers : whole-number positions on per-channel dials, one buffer per boundary. Every step each
//                              layer takes what the layer below passed up the step before. A layer whose input did not
//                              change is settled and sends nothing; it is not rerun (it would give the same again).
//                              The GPU counts the positions that moved at every boundary: the edges, visible.
//   the weights              : 16-bit edges on a 65,536 circle, phase (12 bits) + laps (4 bits), negatives as
//                              complements. Poured in base 2 by default: each of an edge's 15 binary places that is
//                              set adds the input once, shifted by its place. Zero places, zero edges and zero inputs
//                              are skipped and cost nothing. Every pile is an exact whole number.
//   the bend                 : the lap's edge. The share that gets past is looked up from a written table.
//   layer norm, attention    : small float laws on the GPU, results back onto dials (as optical_lm).
//   words                    : when everything is still, the top is read; the word is the next token. Earlier tokens
//                              stand in each layer's keys and values and are never recomputed.
//
// Switches (any together):
//   order=shuffle   layers fire in a random order each step, each reading whatever is below it now
//   init=noise      a new token's layers start holding junk instead of zeros
//   perm=SEED       the 12 layers wired in a random depth order        swap=A,B   two layers swapped
//   channels=SEED   the 768 channels shuffled consistently everywhere
//   neurons=SEED    the 3,072 neurons in every MLP shuffled           heads=SEED  the 12 heads shuffled
//   landings=SEED   the order every pile's arrivals are added in shuffled
//   digits=SEED     the order each edge's digits are poured in shuffled (base 2: its 15 binary places)
//   base=2|8|4096   the circle the edges are poured on, default 2 (all hold the same 16-bit weights)
//   pipe=HOLD       several tokens in flight: the prompt streams in, and once the newest token's top has named the same
//                   word HOLD steps running it is struck at once; if a settled token's final word differs, everything
//                   after it is taken back and the right word struck. Exact by construction; only the steps change.
//   case=0|1|2  words=N  show=N (words printed step by step)
//
//   swiftc -O -o streaming_machine streaming_machine.swift && ./streaming_machine order=shuffle init=noise digits=1
//   Data: ~/tick-machine/gpt2_dial/optical/ (made by tick-machine/optical_export.py), read only.
import Foundation
import Metal

var ARG = [String: String]()
for a in CommandLine.arguments.dropFirst() { let p = a.split(separator: "=", maxSplits: 1).map(String.init); ARG[p[0]] = p.count > 1 ? p[1] : "1" }
func argInt(_ k: String, _ d: Int) -> Int { return Int(ARG[k] ?? "") ?? d }
let SHUFFLE = ARG["order"] == "shuffle", NOISE = ARG["init"] == "noise"
let CASE = argInt("case", 0), WORDS = argInt("words", 8), SHOW = argInt("show", 2)

let src = """
#include <metal_stdlib>
using namespace metal;
#define SUB 8
#define PILES 32
constant int ORD[] = { DIGIT_ORDER };               // the order an edge's digits are poured in
inline long fire(int x, int c, thread uint& n) {   // an edge meets input x; zeros pour nothing and cost nothing
    if (x == 0) return 0;                            // a silent input: skipped
    int v = c < 32768 ? c : c - 65536; int m = v < 0 ? -v : v; long xv = x;
    if (m == 0) return 0;                            // a silent edge: skipped
#if BASEBITS == 12
    long acc = ((xv * (m >> 12)) << 12) + xv * (m & 4095); n += 2;    // base 4096: the laps land at once, then the phase
#elif BASEBITS == 1 && !DIGITS_SHUFFLED
    long acc = 0; uint mm = (uint)m;                 // base 2: visit only the set places; each is one add, shifted
    while (mm) { int j = ctz(mm); acc += xv << j; n++; mm &= mm - 1; }
#else
    long acc = 0;                                    // each digit poured (zero digits skipped), shifted by its place, in ORD
    for (int o = 0; o < NDIGITS; o++) { int j = ORD[o] * BASEBITS; int d = (m >> j) & ((1 << BASEBITS) - 1); if (d == 0) continue;
        long p = 0; for (int k = 0; k < d; k++) p += xv; n += d; acc += p << j; }
#endif
    return v < 0 ? -acc : acc;
}
kernel void lap(device const ushort* edges [[buffer(0)]], device const int* x [[buffer(1)]], device const long* bias [[buffer(2)]], device long* out [[buffer(3)]],
                device atomic_uint* adds [[buffer(4)]], constant uint& nin [[buffer(5)]], constant uint& nout [[buffer(6)]],
                uint t [[thread_position_in_threadgroup]], uint g [[threadgroup_position_in_grid]]) {
    threadgroup long part[SUB][PILES];
    uint pile = g * PILES + (t % PILES), sub = t / PILES;
    long acc = 0; uint n = 0;
    if (pile < nout) for (uint i = sub; i < nin; i += SUB) acc += fire(x[i], (int)edges[i * nout + pile], n);
    atomic_fetch_add_explicit(adds, n, memory_order_relaxed);
    if (g == 0 && t % PILES == 0) { uint z = 0; for (uint i = sub; i < nin; i += SUB) z += x[i] == 0; atomic_fetch_add_explicit(adds + 1, z, memory_order_relaxed); }
    part[sub][t % PILES] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sub == 0 && pile < nout) { long s = bias[pile]; for (uint k = 0; k < SUB; k++) s += part[k][t % PILES]; out[pile] = s; }
}
kernel void gauge_dial(device const long* level [[buffer(0)]], device const float* gauge [[buffer(1)]], device const float* uout [[buffer(2)]], device float* y [[buffer(3)]],
                       constant uint& n [[buffer(4)]], uint j [[thread_position_in_grid]]) {
    if (j >= n) return; long v = level[j]; long hi = v >> 20, lo = v & 0xFFFFF;
    float r = ((float)hi * 1048576.0f + (float)lo) * gauge[j];
    y[j] = uout[j] > 0 ? rint(r / uout[j]) * uout[j] : r;
}
kernel void to_ticks(device const float* y [[buffer(0)]], device const float* uin [[buffer(1)]], device const uint* order [[buffer(2)]], device int* x [[buffer(3)]],
                     constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i < n) x[i] = (int)rint(y[order[i]] / uin[i]); }       // arrivals in the bank's landing order
kernel void edge_bend(device float* y [[buffer(0)]], device const float* share [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return; float v = y[i];                         // where the value arrives against the lap's edge
    float f = rint((v + 16.0f) / 32.0f * 1048575.0f);
    float s = f < 0.0f ? 0.0f : (f > 1048575.0f ? 1.0f : share[(uint)f]);
    y[i] = v * s; }
kernel void resadd(device int* res [[buffer(0)]], device const float* y [[buffer(1)]], device const float* ures [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < 768) res[i] += (int)rint(y[i] / ures[i]); }
kernel void copy_res(device const int* a [[buffer(0)]], device int* b [[buffer(1)]], uint i [[thread_position_in_grid]]) { if (i < 768) b[i] = a[i]; }
kernel void count_moved(device const int* now [[buffer(0)]], device int* last [[buffer(1)]], device atomic_int* moved [[buffer(2)]], constant uint& k [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= 768) return; if (now[i] != last[i]) atomic_fetch_add_explicit(&moved[k], 1, memory_order_relaxed); last[i] = now[i]; }
kernel void water(device const int* res [[buffer(0)]], device const float* ures [[buffer(1)]], device const float* w [[buffer(2)]], device const float* b [[buffer(3)]],
                  device float* y [[buffer(4)]], uint t [[thread_position_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float ps[8], pq[8]; float v[3]; float s = 0, q = 0;
    for (int k = 0; k < 3; k++) { v[k] = (float)res[t + 256 * k] * ures[t + 256 * k]; s += v[k]; }
    s = simd_sum(s); if (lane == 0) ps[sg] = s; threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = (ps[0]+ps[1]+ps[2]+ps[3]+ps[4]+ps[5]+ps[6]+ps[7]) / 768.0f;
    for (int k = 0; k < 3; k++) q += (v[k] - mean) * (v[k] - mean);
    q = simd_sum(q); if (lane == 0) pq[sg] = q; threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rsqrt((pq[0]+pq[1]+pq[2]+pq[3]+pq[4]+pq[5]+pq[6]+pq[7]) / 768.0f + 1e-5f);
    for (int k = 0; k < 3; k++) { uint i = t + 256 * k; y[i] = (v[k] - mean) * inv * w[i] + b[i]; }
}
kernel void store_kv(device const float* qkv [[buffer(0)]], device float* kv [[buffer(1)]], constant uint& pos [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < 2304) kv[pos * 2304 + i] = qkv[i]; }
kernel void attn(device const float* qkv [[buffer(0)]], device const float* kv [[buffer(1)]], device float* o [[buffer(2)]], constant uint& pos [[buffer(3)]], uint h [[thread_position_in_grid]]) {
    if (h >= 12) return; float sc[128]; float mx = -1e30f;
    for (uint s = 0; s <= pos; s++) { float d = 0; for (uint k = 0; k < 64; k++) d += qkv[h*64 + k] * kv[s*2304 + 768 + h*64 + k]; sc[s] = d / 8.0f; mx = max(mx, sc[s]); }
    float z = 0; for (uint s = 0; s <= pos; s++) { sc[s] = exp(sc[s] - mx); z += sc[s]; }
    for (uint k = 0; k < 64; k++) { float a = 0; for (uint s = 0; s <= pos; s++) a += sc[s] / z * kv[s*2304 + 1536 + h*64 + k]; o[h*64 + k] = a; }
}
"""
setvbuf(stdout, nil, _IONBF, 0)
let dir = NSString(string: "~/tick-machine/gpt2_dial/optical/").expandingTildeInPath + "/"
let dev = MTLCreateSystemDefaultDevice()!, queue = dev.makeCommandQueue()!
let BASE = argInt("base", 2), BASEBITS = BASE == 2 ? 1 : (BASE == 8 ? 3 : 12)
let NDIGITS = (15 + BASEBITS - 1) / BASEBITS
let digitOrder = ARG["digits"].flatMap { Int($0) }.map { sd -> [Int] in var p = Array(0..<NDIGITS); var r = sd &+ 7
    for i in stride(from: NDIGITS - 1, to: 0, by: -1) { r = (r &* 1103515245 &+ 12345) & 0x7fffffff; p.swapAt(i, r % (i + 1)) }; return p } ?? Array(0..<NDIGITS)
let copts = MTLCompileOptions()
copts.preprocessorMacros = ["BASEBITS": BASEBITS as NSNumber, "NDIGITS": NDIGITS as NSNumber, "DIGIT_ORDER": digitOrder.map(String.init).joined(separator: ",") as NSString,
                           "DIGITS_SHUFFLED": (ARG["digits"] != nil ? 1 : 0) as NSNumber]
let lib = try! dev.makeLibrary(source: src, options: copts)
func pipe(_ n: String) -> MTLComputePipelineState { return try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
let pLap = pipe("lap"), pGauge = pipe("gauge_dial"), pTicks = pipe("to_ticks"), pBend = pipe("edge_bend"), pRes = pipe("resadd"), pWater = pipe("water")
let pStore = pipe("store_kv"), pAttn = pipe("attn"), pCopy = pipe("copy_res"), pCount = pipe("count_moved")
func load<T>(_ name: String, _ t: T.Type) -> [T] { let d = try! Data(contentsOf: URL(fileURLWithPath: dir + name)); return d.withUnsafeBytes { Array($0.bindMemory(to: T.self)) } }
func buf<T>(_ a: [T]) -> MTLBuffer { return dev.makeBuffer(bytes: a, length: max(a.count, 1) * MemoryLayout<T>.stride, options: .storageModeShared)! }
func zbuf(_ bytes: Int) -> MTLBuffer { return dev.makeBuffer(length: max(bytes, 4), options: .storageModeShared)! }

// ---- shuffles, applied consistently to the loaded weights
struct Rng { var s: UInt64; mutating func next() -> UInt64 { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s } }
func shuffled(_ n: Int, _ seed: Int?) -> [Int] {
    var p = Array(0..<n); guard let sd = seed else { return p }
    var r = Rng(s: UInt64(sd) &+ 0x9E3779B97F4A7C15); for i in stride(from: n - 1, to: 0, by: -1) { p.swapAt(i, Int(r.next() % UInt64(i + 1))) }; return p
}
func seed(_ k: String) -> Int? { return ARG[k].flatMap { Int($0) } }
let L = 12, D = 768, V = 50257, MAXPOS = 128
let pc = shuffled(D, seed("channels")), pn = shuffled(3072, seed("neurons")), ph = shuffled(12, seed("heads"))
let headCols: [Int] = (0..<3).flatMap { b in (0..<12).flatMap { h in (0..<64).map { b * 768 + ph[h] * 64 + $0 } } }   // qkv columns, heads shuffled
let headRows: [Int] = (0..<12).flatMap { h in (0..<64).map { ph[h] * 64 + $0 } }                                          // proj rows, heads shuffled
func pick<T>(_ a: [T], _ idx: [Int]) -> [T] { return idx.map { a[$0] } }
func rows(_ e: [UInt16], _ nout: Int, _ idx: [Int]) -> [UInt16] { var o = [UInt16](); o.reserveCapacity(e.count); for i in idx { o += e[(i * nout)..<((i + 1) * nout)] }; return o }
func cols(_ e: [UInt16], _ nin: Int, _ nout: Int, _ idx: [Int]) -> [UInt16] { var o = [UInt16](repeating: 0, count: e.count); for i in 0..<nin { for (j, c) in idx.enumerated() { o[i * nout + j] = e[i * nout + c] } }; return o }

struct Bank { let nin, nout: Int; let edges, gauge, uIn, uOut, bias, order, x, level, y, adds: MTLBuffer }
var banks = [String: Bank](); var bytes = 0; var bankKind = [ObjectIdentifier: String]()
let man = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: dir + "manifest.json"))) as! [[String: Any]]
for (bi, b) in man.enumerated() {
    let n = b["name"] as! String, nin = b["nin"] as! Int, nout = b["nout"] as! Int, kind = n.split(separator: ".")[0]
    var e = load("\(n).edges.bin", UInt16.self), uIn = load("\(n).u_in.bin", Float.self)
    var gauge = load("\(n).gauge.bin", Float.self), uOut = load("\(n).u_out.bin", Float.self), bias = load("\(n).bias.bin", Int64.self)
    var rIdx = Array(0..<nin), cIdx = Array(0..<nout)
    if nin == D && kind != "proj" && kind != "mproj" { rIdx = pc }               // qkv, fc, head read the 768 channels
    if kind == "proj" { rIdx = headRows }                                        // proj reads the heads' outputs
    if kind == "mproj" { rIdx = pn; cIdx = pc }                                  // mproj reads the neurons, writes the channels
    if kind == "proj" { cIdx = pc }
    if kind == "fc" { cIdx = pn }
    if kind == "qkv" { cIdx = headCols }
    e = cols(rows(e, nout, rIdx), nin, nout, cIdx); uIn = pick(uIn, rIdx)
    gauge = pick(gauge, cIdx); uOut = pick(uOut, cIdx); bias = pick(bias, cIdx)
    let land = shuffled(nin, seed("landings").map { $0 + bi })                  // the order this bank's arrivals are added in
    e = rows(e, nout, land); uIn = pick(uIn, land)
    bytes += e.count * 2
    let eb = buf(e); bankKind[ObjectIdentifier(eb)] = String(kind)
    banks[n] = Bank(nin: nin, nout: nout, edges: eb, gauge: buf(gauge), uIn: buf(uIn), uOut: buf(uOut), bias: buf(bias),
                    order: buf(land.map { UInt32($0) }), x: zbuf(nin * 4), level: zbuf(nout * 8), y: zbuf(nout * 4), adds: zbuf(8))
}
let uRes = buf(pick(load("u_res.bin", Float.self), pc))
let wteAll = load("wte.ticks.bin", Int32.self), wpeAll = load("wpe.ticks.bin", Int32.self)
func lnb(_ n: String) -> MTLBuffer { return buf(pick(load(n, Float.self), pc)) }
let ln1w = (0..<L).map { lnb("ln.\($0).ln_1.weight.bin") }, ln1b = (0..<L).map { lnb("ln.\($0).ln_1.bias.bin") }
let ln2w = (0..<L).map { lnb("ln.\($0).ln_2.weight.bin") }, ln2b = (0..<L).map { lnb("ln.\($0).ln_2.bias.bin") }
let lnfw = lnb("ln_f.weight.bin"), lnfb = lnb("ln_f.bias.bin")
let share = buf((0..<(1 << 20)).map { i -> Float in let p = -16.0 + 32.0 * Double(i) / Double((1 << 20) - 1); return Float(0.5 * (1.0 + erf(p / 2.0.squareRoot()))) })

// ---- wiring
var wiring = Array(0..<L)
if let s = seed("perm") { wiring = shuffled(L, s) }
if let sw = ARG["swap"] { let ab = sw.split(separator: ",").compactMap { Int($0) }; if ab.count == 2 { wiring.swapAt(ab[0], ab[1]) } }

// ---- the machine's state
let res = (0...L).map { _ in zbuf(D * 4) }          // the state at every boundary: 0 = the token, k = what layer k passed up
let snap = (0...L).map { _ in zbuf(D * 4) }         // what each boundary held at the start of the step
let last = (0...L).map { _ in zbuf(D * 4) }         // what each boundary held at the end of the last step (to count edges)
let work = zbuf(D * 4), nrm = zbuf(D * 4), att = zbuf(D * 4), moved = zbuf((L + 1) * 4)
let kv = (0..<L).map { _ in dev.makeBuffer(length: MAXPOS * 2304 * 4, options: .storageModePrivate)! }

func enc(_ cb: MTLCommandBuffer, _ ps: MTLComputePipelineState, _ bufs: [MTLBuffer], _ consts: [UInt32], threads: Int, tg: Int, groups: Bool = false) {
    let e = cb.makeComputeCommandEncoder()!; e.setComputePipelineState(ps)
    for (i, b) in bufs.enumerated() { e.setBuffer(b, offset: 0, index: i) }
    for (k, c) in consts.enumerated() { var v = c; e.setBytes(&v, length: 4, index: bufs.count + k) }
    if groups { e.dispatchThreadgroups(MTLSize(width: threads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)) }
    else { e.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)) }
    e.endEncoding()
}
let zeros = buf([Float](repeating: 0, count: V))
func lapOf(_ cb: MTLCommandBuffer, _ bk: Bank, from y: MTLBuffer, dial: Bool) {
    enc(cb, pTicks, [y, bk.uIn, bk.order, bk.x], [UInt32(bk.nin)], threads: bk.nin, tg: 256)
    enc(cb, pLap, [bk.edges, bk.x, bk.bias, bk.level, bk.adds], [UInt32(bk.nin), UInt32(bk.nout)], threads: (bk.nout + 31) / 32, tg: 256, groups: true)
    possible += bk.nin * bk.nout * PERDIGIT
    allIn[bankKind[ObjectIdentifier(bk.edges)]!, default: 0] += bk.nin
    enc(cb, pGauge, [bk.level, bk.gauge, dial ? bk.uOut : zeros, bk.y], [UInt32(bk.nout)], threads: bk.nout, tg: 256)
}
func layer(_ cb: MTLCommandBuffer, depth d: Int, input: MTLBuffer, pos: Int, out: MTLBuffer? = nil) {    // one layer: input boundary -> output boundary
    let w = wiring[d], bq = banks["qkv.\(w)"]!, bp = banks["proj.\(w)"]!, bf = banks["fc.\(w)"]!, bm = banks["mproj.\(w)"]!
    enc(cb, pCopy, [input, work], [], threads: D, tg: 256)
    enc(cb, pWater, [work, uRes, ln1w[w], ln1b[w], nrm], [], threads: 1, tg: 256, groups: true)
    lapOf(cb, bq, from: nrm, dial: true)
    enc(cb, pStore, [bq.y, kv[d]], [UInt32(pos)], threads: 2304, tg: 256)
    enc(cb, pAttn, [bq.y, kv[d], att], [UInt32(pos)], threads: 12, tg: 12)
    lapOf(cb, bp, from: att, dial: true); enc(cb, pRes, [work, bp.y, uRes], [], threads: D, tg: 256)
    enc(cb, pWater, [work, uRes, ln2w[w], ln2b[w], nrm], [], threads: 1, tg: 256, groups: true)
    lapOf(cb, bf, from: nrm, dial: true); enc(cb, pBend, [bf.y, share], [UInt32(bf.nout)], threads: bf.nout, tg: 256)
    lapOf(cb, bm, from: bf.y, dial: true); enc(cb, pRes, [work, bm.y, uRes], [], threads: D, tg: 256)
    enc(cb, pCopy, [work, out ?? res[d + 1]], [], threads: D, tg: 256)
}
let bh = banks["head"]!
let PERDIGIT = BASEBITS == 12 ? 2 : NDIGITS * ((1 << BASEBITS) - 1)   // the most adds one edge can cost
var possible = 0, done = 0
var zeroIn = [String: Int](), allIn = [String: Int]()
func harvest() { for (n, bk) in banks { let p = bk.adds.contents().bindMemory(to: UInt32.self, capacity: 2); done += Int(p[0])
    let kind = String(n.split(separator: ".")[0]); zeroIn[kind, default: 0] += Int(p[1]); p[0] = 0; p[1] = 0 } }
func readTop(_ top: MTLBuffer? = nil) -> (Int, Float, Float) {
    let cb = queue.makeCommandBuffer()!
    enc(cb, pWater, [top ?? res[L], uRes, lnfw, lnfb, nrm], [], threads: 1, tg: 256, groups: true); lapOf(cb, bh, from: nrm, dial: false)
    cb.commit(); cb.waitUntilCompleted(); harvest()
    let lg = bh.y.contents().bindMemory(to: Float.self, capacity: V); var a = 0, b = 1
    if lg[1] > lg[0] { a = 1; b = 0 }
    for t in 2..<V { if lg[t] > lg[a] { b = a; a = t } else if lg[t] > lg[b] { b = t } }
    return (a, lg[a] - lg[b], lg[a])
}

// ---- reference and token text
let ref = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: dir + "../reference.json"))) as! [[String: Any]]
let ids = ref[CASE]["ids"] as! [Int], want = Array((ref[CASE]["ref"] as! [Int]).prefix(WORDS)), prompt = ref[CASE]["prompt"] as! String
let vocab: [Int: String] = {
    let j = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: NSString(string: "~/.cache/huggingface/hub/models--gpt2/snapshots").expandingTildeInPath + "/" +
        (try! FileManager.default.contentsOfDirectory(atPath: NSString(string: "~/.cache/huggingface/hub/models--gpt2/snapshots").expandingTildeInPath)).first! + "/vocab.json"))) as! [String: Int]
    var v = [Int: String](); for (k, i) in j { v[i] = k.replacingOccurrences(of: "Ġ", with: " ").replacingOccurrences(of: "Ċ", with: "\\n") }; return v }()
func word(_ t: Int) -> String { return "'" + (vocab[t] ?? "?") + "'" }

var on = [String]()
if SHUFFLE { on.append("order shuffled") }; if NOISE { on.append("starts on junk") }
if seed("perm") != nil || ARG["swap"] != nil { on.append("wiring \(wiring)") }
for k in ["channels", "neurons", "heads", "landings"] where seed(k) != nil { on.append("\(k) shuffled") }
if ARG["digits"] != nil { on.append("digits poured in order \(digitOrder)") }
print("streaming machine on \(dev.name): \(bytes / 1_000_000) MB of 16-bit edges poured in base \(BASE) (\(BASE == 4096 ? "laps at once, then the phase" : "\((15 + BASEBITS - 1) / BASEBITS) digits of up to \(BASE - 1) adds, each shifted by its place")); the bend as the lap's edge")
print("'\(prompt)'   \(on.isEmpty ? "as trained" : on.joined(separator: ", "))")


// ---- pipelined: several tokens in flight
let PIPE = argInt("pipe", 0), FLIGHT = argInt("flight", 16)
final class Slot {
    let pos: Int; let tok: Int; let res, snap, last: [MTLBuffer]; let moved: MTLBuffer
    var dirty = [Bool](repeating: true, count: 12), steps = 0, held = 0, lastTop = -1, settled = false
    init(_ pos: Int, _ tok: Int) { self.pos = pos; self.tok = tok
        res = (0...12).map { _ in zbuf(768 * 4) }; snap = (0...12).map { _ in zbuf(768 * 4) }; last = (0...12).map { _ in zbuf(768 * 4) }; moved = zbuf(13 * 4) }
}
if PIPE > 0 {
    var rngP = Rng(s: 777)
    func strike(_ pos: Int, _ tok: Int) -> Slot {
        let sl = Slot(pos, tok); let r0 = sl.res[0].contents().bindMemory(to: Int32.self, capacity: D)
        for i in 0..<D { r0[i] = wteAll[tok * D + pc[i]] + wpeAll[pos * D + pc[i]] }
        let big = (0..<D).map { abs(Int(r0[$0])) }.max()!
        for k in 1...L { let r = sl.res[k].contents().bindMemory(to: Int32.self, capacity: D)
            for i in 0..<D { r[i] = NOISE ? Int32(Int(rngP.next() % UInt64(4 * big + 1)) - 2 * big) : 0 } }
        return sl
    }
    var seqP = [ids[0]], slots = [strike(0, ids[0])], genP = [Int](), stepsP = 0, takebacks = 0, guesses = 0, runs = 0
    let tP = Date()
    print("pipelined: strike a guess after the top holds \(PIPE) steps; up to \(FLIGHT) tokens in flight")
    while genP.count < WORDS {
        stepsP += 1
        if seqP.count < ids.count { seqP.append(ids[seqP.count]); slots.append(strike(seqP.count - 1, seqP.last!)) }   // the prompt streams in
        let cb = queue.makeCommandBuffer()!
        var earlierMoving = false
        for sl in slots where !sl.settled {
            memset(sl.moved.contents(), 0, 13 * 4)
            if SHUFFLE {
                var order = Array(0..<L); for i in stride(from: L - 1, to: 0, by: -1) { order.swapAt(i, Int(rngP.next() % UInt64(i + 1))) }
                for d in order { layer(cb, depth: d, input: sl.res[d], pos: sl.pos, out: sl.res[d + 1]); runs += 1 }
            } else {
                for k in 0...L { enc(cb, pCopy, [sl.res[k], sl.snap[k]], [], threads: D, tg: 256) }
                for d in 0..<L where sl.steps == 0 || earlierMoving || sl.dirty[d] { layer(cb, depth: d, input: sl.snap[d], pos: sl.pos, out: sl.res[d + 1]); runs += 1 }
            }
            for k in 0...L { enc(cb, pCount, [sl.res[k], sl.last[k], sl.moved], [UInt32(k)], threads: D, tg: 256) }
            earlierMoving = true
        }
        cb.commit(); cb.waitUntilCompleted(); harvest()
        var earlierSettled = true, retake: (Int, Int)? = nil
        for sl in slots where !sl.settled {
            let mv = Array(UnsafeBufferPointer(start: sl.moved.contents().bindMemory(to: Int32.self, capacity: L + 1), count: L + 1))
            sl.steps += 1; for d in 0..<L { sl.dirty[d] = mv[d] != 0 }
            if earlierSettled && sl.steps > 1 && mv[1...].allSatisfy({ $0 == 0 }) {
                sl.settled = true
                let next = sl.pos + 1
                if next >= ids.count {                                                       // this token's final word is the next token
                    let (t, _, _) = readTop(sl.res[L])
                    if next < seqP.count && seqP[next] != t { retake = (next, t); takebacks += 1; genP.append(t); break }
                    if next >= seqP.count { seqP.append(t); slots.append(strike(next, t)) }
                    genP.append(t)
                    if genP.count <= SHOW + 40 && genP.count <= WORDS {
                        let ok = genP.count <= want.count && t == want[genP.count - 1] ? "" : "   (the model says \(word(want[genP.count - 1])))"
                        print("  word \(String(format: "%2d", genP.count)): \(word(t).padding(toLength: 12, withPad: " ", startingAt: 0)) settled at step \(stepsP)\(ok)") }
                    if genP.count >= WORDS { break }
                }
            } else { earlierSettled = false }
        }
        if let (p, t) = retake {                                                             // take back everything after the wrong guess
            slots.removeAll { $0.pos >= p }; seqP = Array(seqP[..<p]) + [t]; slots.append(strike(p, t))
            let ok = genP.count <= want.count && t == want[genP.count - 1] ? "" : "   (the model says \(word(want[genP.count - 1])))"
            print("  word \(String(format: "%2d", genP.count)): \(word(t).padding(toLength: 12, withPad: " ", startingAt: 0)) settled at step \(stepsP)  -- guess taken back\(ok)")
        }
        let oldest = slots.first { !$0.settled }?.pos ?? Int.max; slots.removeAll { $0.settled && $0.pos < oldest - 1 }
        if let nw = slots.last, !nw.settled, seqP.count >= ids.count, nw.pos == seqP.count - 1, slots.filter({ !$0.settled }).count < FLIGHT, genP.count + slots.filter({ !$0.settled && $0.pos >= ids.count - 1 }).count < WORDS {
            let (t, _, _) = readTop(nw.res[L])                                               // the newest token's top, so far
            nw.held = t == nw.lastTop ? nw.held + 1 : 1; nw.lastTop = t
            if nw.held >= PIPE { seqP.append(t); slots.append(strike(nw.pos + 1, t)); guesses += 1 }
        }
        if stepsP > 5000 { print("  (stopped: 5000 steps)"); break }
    }
    let wallP = Date().timeIntervalSince(tP), out = Array(genP.prefix(WORDS))
    print("\nstreaming : " + out.map { vocab[$0] ?? "?" }.joined())
    print("the model : " + want.map { vocab[$0] ?? "?" }.joined())
    print(String(format: "%@; %d steps for %d prompt tokens + %d words (one at a time: %d); %d guesses, %d taken back; %d layer runs; %.2f s, %.1f words/s",
                 out == want ? "IDENTICAL" : "DIFFERENT", stepsP, ids.count, out.count, 13 * (ids.count + out.count - 1), guesses, takebacks, runs, wallP, Double(out.count) / wallP))
    exit(0)
}

// ---- run
var rng = Rng(s: 12345); var seq = ids, gen = [Int](), steps = 0, totalEdges = 0, layerRuns = 0
let t0 = Date()
var pos = 0
while gen.count < WORDS {
    let tid = seq[pos], show = pos >= ids.count - 1 && gen.count < SHOW
    let r0 = res[0].contents().bindMemory(to: Int32.self, capacity: D)
    for i in 0..<D { r0[i] = wteAll[tid * D + pc[i]] + wpeAll[pos * D + pc[i]] }            // the token struck at the bottom
    let big = (0..<D).map { abs(Int(r0[$0])) }.max()!
    for k in 1...L {                                                                          // its layers start empty (or on junk)
        let r = res[k].contents().bindMemory(to: Int32.self, capacity: D)
        for i in 0..<D { r[i] = NOISE ? Int32(Int(rng.next() % UInt64(4 * big + 1)) - 2 * big) : 0 }
    }
    var dirty = [Bool](repeating: true, count: L), stepInWord = 0
    if show { print(String(format: "%5@ | edges fired leaving layers 1..12 (positions that moved, of 768)                          | top", "step")) }
    while true {
        steps += 1; stepInWord += 1
        let cb = queue.makeCommandBuffer()!
        memset(moved.contents(), 0, (L + 1) * 4)
        if SHUFFLE {
            var order = Array(0..<L); for i in stride(from: L - 1, to: 0, by: -1) { order.swapAt(i, Int(rng.next() % UInt64(i + 1))) }
            for d in order { layer(cb, depth: d, input: res[d], pos: pos); layerRuns += 1 }       // each reads whatever is below it now
        } else {
            for k in 0...L { enc(cb, pCopy, [res[k], snap[k]], [], threads: D, tg: 256) }       // everyone reads last step's state
            for d in 0..<L where dirty[d] { layer(cb, depth: d, input: snap[d], pos: pos); layerRuns += 1 }
        }
        for k in 0...L { enc(cb, pCount, [res[k], last[k], moved], [UInt32(k)], threads: D, tg: 256) }
        cb.commit(); cb.waitUntilCompleted(); harvest()
        let mv = Array(UnsafeBufferPointer(start: moved.contents().bindMemory(to: Int32.self, capacity: L + 1), count: L + 1))
        let fired = stepInWord == 1 ? Array(mv[1...]) : Array(mv[1...])
        totalEdges += fired.reduce(0) { $0 + Int($1) }
        for d in 0..<L { dirty[d] = stepInWord == 1 ? true : mv[d] != 0 }                     // a layer reruns only if its input moved
        let still = stepInWord > 1 && fired.allSatisfy { $0 == 0 }
        if show {
            let (t, mg, _) = readTop()
            print(String(format: "%5d | ", stepInWord) + fired.map { String(format: "%4d", $0) }.joined(separator: " ") + String(format: " | %-10@ %+.3f", word(t), mg))
        }
        if still || stepInWord > 60 { break }
    }
    pos += 1
    if pos >= seq.count {
        let (t, _, _) = readTop(); gen.append(t); seq.append(t)
        let note = gen.count <= want.count && t == want[gen.count - 1] ? "" : "   (the model says \(word(want[gen.count - 1])))"
        print("  word \(String(format: "%2d", gen.count)): \(word(t).padding(toLength: 12, withPad: " ", startingAt: 0)) settled after \(stepInWord) steps\(note)")
    }
}
let wall = Date().timeIntervalSince(t0)
print("\nstreaming : " + gen.map { vocab[$0] ?? "?" }.joined())
print("the model : " + want.map { vocab[$0] ?? "?" }.joined())
for k in ["qkv", "proj", "fc", "mproj", "head"] { print(String(format: "  inputs arriving as exactly zero, %-5@ bank: %5.1f%%", k, 100.0 * Double(zeroIn[k] ?? 0) / Double(max(allIn[k] ?? 1, 1)))) }
print(String(format: "adds done %@ of %@ possible (%.1f%%): the zeros cost nothing", NumberFormatter.localizedString(from: done as NSNumber, number: .decimal),
             NumberFormatter.localizedString(from: possible as NSNumber, number: .decimal), 100.0 * Double(done) / Double(max(possible, 1))))
print(String(format: "%@; %d steps, %d layer runs (%.1f per token), %d edges; %.2f s, %.1f words/s",
             gen == want ? "IDENTICAL" : "DIFFERENT", steps, layerRuns, Double(layerRuns) / Double(pos), totalEdges, wall, Double(gen.count) / wall))
