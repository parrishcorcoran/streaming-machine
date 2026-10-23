// THE ALL-BASE-2 SOUP. GPT-2 with every number a whole number in base 2, on the M4's GPU (SPEC.md).
// Operations: AND, OR, counting 1s, shift, add, subtract, table lookup. No float, no multiply, no compare between values.
//
//   products and sums : bit planes. A weight digit meets an input digit (AND, 32 inputs a word), the 1s are counted,
//                       and the counts join their places once per pile. One GPU lane per position.
//   other products    : shift-adds over the set digits of the smaller number.
//   the gauge         : log2 by the top 1's position and a table; root = halve, divide = negate; back by a table + shift.
//   the edge          : the share table, read where the value arrives.
//   the arrival       : the arrival table, read at each position's distance from the race's winner; total = the gauge.
//   the race          : top digits first; anyone still in with a 1 knocks out everyone with a 0.
//   the soup          : the prompt and the next N words all in at once; every step every layer fires bottom to top for
//                       every position, each reading what is below it now; every future word is the race at the top
//                       one position back. Runs until nothing moves and no word changes.
//
//   python3 base2_convert.py && python3 base2_planes.py
//   swiftc -O -o base2_soup base2_soup.swift && ./base2_soup case=0 words=16
import Foundation
import Metal

var ARG = [String: String]()
for a in CommandLine.arguments.dropFirst() { let p = a.split(separator: "=", maxSplits: 1).map(String.init); ARG[p[0]] = p.count > 1 ? p[1] : "1" }
func argInt(_ k: String, _ d: Int) -> Int { return Int(ARG[k] ?? "") ?? d }
let CASE = argInt("case", 0), WORDS = argInt("words", 16), MAXSTEPS = argInt("steps", 200)
let dir = NSString(string: "~/streaming-machine/base2_gpu/").expandingTildeInPath + "/"
let man = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: dir + "manifest.json"))) as! [String: Any]
let LOG768 = man["log2_768"] as! Int, EPS40 = man["eps40"] as! Int, ALEN = man["arrival_len"] as! Int

let src = """
#include <metal_stdlib>
using namespace metal;
constant long LOG768 = \(LOG768);
constant long EPS40 = \(EPS40);
constant long ALEN = \(ALEN);

inline long rsh(long x, long s) {                       // x / 2^s, nearest: add half, then shift (s < 0 shifts left)
    if (s <= 0) return x << (-s);
    return (x + (1L << (s - 1))) >> s; }
inline long times(long x, ulong m) {                    // x * m as shift-adds over the set digits of m
    long acc = 0; while (m) { int j = ctz(m); acc += x << j; m &= m - 1; } return acc; }
inline long times_s(long x, long m) { return m < 0 ? -times(x, (ulong)(-m)) : times(x, (ulong)m); }
inline long log2fix(ulong v, device const long* LOG2) { // where the top 1 sits, plus a fine part read from the table
    int t = 63 - clz(v); ulong m = t >= 16 ? (v >> (t - 16)) & 0xFFFF : (v << (16 - t)) & 0xFFFF;
    return (long)t * 65536 + LOG2[m]; }
inline long power(long x, long F, device const long* EXP2) {   // x * 2^(F/65536): a table read of the fine part, a shift
    long I = F >> 16, f = F & 0xFFFF; return rsh(x * EXP2[f], 16 - I); }

kernel void to16(device const long* X [[buffer(0)]], device int* x16 [[buffer(1)]], device int* ex [[buffer(2)]],
                 constant uint& n [[buffer(3)]], constant int& ein [[buffer(4)]], constant uint& stride [[buffer(5)]], constant uint& off [[buffer(6)]],
                 uint t [[thread_position_in_threadgroup]], uint p [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    // one position on 16 bits: the top digit any of its values reaches (OR of the magnitudes) sets its power of two
    threadgroup uint orh[8], orl[8];
    device const long* x = X + p * stride + off;
    ulong o = 0; for (uint i = t; i < n; i += 256) { long v = x[i]; o |= (ulong)(v < 0 ? -v : v); }
    uint hi = simd_or((uint)(o >> 32)), lo = simd_or((uint)o);
    if (lane == 0) { orh[sg] = hi; orl[sg] = lo; } threadgroup_barrier(mem_flags::mem_threadgroup);
    ulong all = 0; for (int k = 0; k < 8; k++) all |= ((ulong)orh[k] << 32) | orl[k];
    int top = all ? 64 - clz(all) : 0; int sh = max(top - 15, 0);
    for (uint i = t; i < n; i += 256) x16[p * n + i] = (int)rsh(x[i], sh);
    if (t == 0) ex[p] = ein + sh;
}
kernel void pack(device const int* x16 [[buffer(0)]], device uint* XP [[buffer(1)]], constant uint& n [[buffer(2)]], constant uint& np [[buffer(3)]],
                 uint id [[thread_position_in_grid]]) {
    // the inputs' binary digits, 32 inputs a word: XP[digit][word][position]
    uint nw = n / 32; if (id >= 16 * nw * 32) return;
    uint lane = id % 32, wi = (id / 32) % nw, b = id / (32 * nw);
    uint w = 0;
    if (lane < np) for (uint t = 0; t < 32; t++) w |= ((((uint)x16[lane * n + wi * 32 + t] & 0xFFFF) >> b) & 1) << t;
    XP[id] = w;
}
kernel void planes(device const uint* WP [[buffer(0)]], device const uint* XP [[buffer(1)]], device long* pile [[buffer(2)]],
                   constant uint& nin [[buffer(3)]], constant uint& nout [[buffer(4)]], constant uint& np [[buffer(5)]],
                   uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint g [[threadgroup_position_in_grid]]) {
    // one pile per simdgroup, one position per lane: a weight digit meets an input digit, the 1s are counted,
    // the count joins its place (the top digit of each is the negative one)
    uint j = g * 8 + sg; if (j >= nout) return;
    uint nw = nin / 32; long acc = 0;
    for (uint a = 0; a < 16; a++) {
        device const uint* wp = WP + (j * 16 + a) * nw;
        for (uint b = 0; b < 16; b++) {
            device const uint* xp = XP + b * nw * 32 + lane;
            uint cnt = 0; for (uint wi = 0; wi < nw; wi++) cnt += popcount(wp[wi] & xp[wi * 32]);
            long c = (long)cnt << (a + b); acc += ((a == 15) != (b == 15)) ? -c : c;
        }
    }
    if (lane < np) pile[lane * nout + j] = acc;
}

kernel void planes2(device const uint* WP [[buffer(0)]], device const uint* XP [[buffer(1)]], device long* pile [[buffer(2)]],
                    constant uint& nin [[buffer(3)]], constant uint& nout [[buffer(4)]], constant uint& np [[buffer(5)]],
                    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint g [[threadgroup_position_in_grid]]) {
    // the same AND-and-count, held in registers: per 32 inputs, the 16 input digits are loaded once, each weight digit
    // once (skipped if all 0), every pair counted into its place a+b; the places join once at the end
    uint j = g * 8 + sg; if (j >= nout) return;
    uint nw = nin / 32; int place[31];
    #pragma unroll
    for (int k = 0; k < 31; k++) place[k] = 0;
    device const uint* wcol = WP + j * 16 * nw;
    for (uint wi = 0; wi < nw; wi++) {
        uint xb[16];
        #pragma unroll
        for (int b = 0; b < 16; b++) xb[b] = XP[(b * nw + wi) * 32 + lane];
        #pragma unroll
        for (int a = 0; a < 16; a++) {
            uint w = wcol[a * nw + wi];
            if (w == 0) continue;
            #pragma unroll
            for (int b = 0; b < 16; b++) { int c = popcount(w & xb[b]); place[a + b] += ((a == 15) != (b == 15)) ? -c : c; }
        }
    }
    long acc = 0;
    #pragma unroll
    for (int k = 0; k < 31; k++) acc += (long)place[k] << k;
    if (lane < np) pile[lane * nout + j] = acc;
}

kernel void tpose(device const int* x16 [[buffer(0)]], device int* xt [[buffer(1)]], constant uint& n [[buffer(2)]], constant uint& np [[buffer(3)]],
                  uint id [[thread_position_in_grid]]) {
    if (id >= n * 32) return; uint i = id / 32, lane = id % 32; xt[id] = lane < np ? x16[lane * n + i] : 0; }
kernel void gates(device const short* W [[buffer(0)]], device const int* x16 [[buffer(1)]], device long* pile [[buffer(2)]],
                  constant uint& nin [[buffer(3)]], constant uint& nout [[buffer(4)]], constant uint& np [[buffer(5)]],
                  uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint g [[threadgroup_position_in_grid]]) {
    // the same product, by the chip's own AND-gate grid and adder tree (the integer multiplier): whole numbers, base 2
    uint j = g * 8 + sg; if (j >= nout) return;
    device const short* w = W + j * nin; device const int* x = x16 + lane;      // x16 here is input-major: [input][32 positions]
    long acc = 0; int part = 0;
    for (uint i = 0; i < nin; i++) { part += (int)w[i] * x[i * 32]; if ((i & 15) == 15) { acc += part; part = 0; } }
    acc += part;
    if (lane < np) pile[lane * nout + j] = acc;
}


inline long shfl_down(long v, ushort o) { uint lo = simd_shuffle_down((uint)(ulong)v, o), hi = simd_shuffle_down((uint)((ulong)v >> 32), o); return (long)(((ulong)hi << 32) | lo); }
inline long bcast0(long v) { uint lo = simd_broadcast((uint)(ulong)v, 0), hi = simd_broadcast((uint)((ulong)v >> 32), 0); return (long)(((ulong)hi << 32) | lo); }
inline long sum64(long v) { for (ushort o = 16; o > 0; o >>= 1) v += shfl_down(v, o); return bcast0(v); }   // every lane gets the total

kernel void gates4(device const short4* W [[buffer(0)]], device const int* xt [[buffer(1)]], device long* pile [[buffer(2)]],
                   constant uint& nin [[buffer(3)]], constant uint& nout [[buffer(4)]], constant uint& np [[buffer(5)]],
                   uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint g [[threadgroup_position_in_grid]]) {
    // the gate grid, 4 weights a fetch, 4 running parts: the pile is only written into while it fills, read once at the end
    uint j = g * 8 + sg; if (j >= nout) return;
    device const short4* w = W + j * (nin / 4); device const int* x = xt + lane;
    int p0 = 0, p1 = 0, p2 = 0, p3 = 0; long acc = 0;
    for (uint q = 0; q < nin / 4; q++) {
        short4 ww = w[q]; uint i = q * 4;
        p0 += (int)ww.x * x[i * 32]; p1 += (int)ww.y * x[(i + 1) * 32]; p2 += (int)ww.z * x[(i + 2) * 32]; p3 += (int)ww.w * x[(i + 3) * 32];
        if ((q & 3) == 3) { acc += (long)p0 + p1 + p2 + p3; p0 = p1 = p2 = p3 = 0; } }
    acc += (long)p0 + p1 + p2 + p3;
    if (lane < np) pile[lane * nout + j] = acc;
}

kernel void gates8(device const short4* W [[buffer(0)]], device const int* xt [[buffer(1)]], device long* pile [[buffer(2)]],
                   constant uint& nin [[buffer(3)]], constant uint& nout [[buffer(4)]], constant uint& np [[buffer(5)]],
                   uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint g [[threadgroup_position_in_grid]]) {
    // one read of each input fills 8 piles: the inputs are fetched once per 8 piles, the 8 weights in one fetch
    uint grp = g * 8 + sg; if (grp * 8 >= nout) return;
    device const short4* w = W + grp * nin * 2; device const int* x = xt + lane;
    int lo[8] = {0, 0, 0, 0, 0, 0, 0, 0}, hi[8] = {0, 0, 0, 0, 0, 0, 0, 0};   // two narrow cells per pile: written, never carried
    for (uint i = 0; i < nin; i++) {
        int xv = x[i * 32]; short4 lw = w[i * 2], hw = w[i * 2 + 1];
        int pr[8] = {lw.x * xv, lw.y * xv, lw.z * xv, lw.w * xv, hw.x * xv, hw.y * xv, hw.z * xv, hw.w * xv};
        for (int k = 0; k < 8; k++) { lo[k] += pr[k] & 0x7FFF; hi[k] += pr[k] >> 15; } }          // bottom 15 digits into one cell, the rest into the other
    long acc[8]; for (int k = 0; k < 8; k++) acc[k] = ((long)hi[k] << 15) + lo[k];                // the one read: high x 2^15 + low
    if (lane < np) for (int k = 0; k < 8; k++) if (grp * 8 + k < nout) pile[lane * nout + grp * 8 + k] = acc[k];
}

kernel void gates8k(device const short4* W [[buffer(0)]], device const int* xt [[buffer(1)]], device long* part [[buffer(2)]],
                    constant uint& nin [[buffer(3)]], constant uint& nout [[buffer(4)]], constant uint& np [[buffer(5)]], constant uint& ksplit [[buffer(6)]],
                    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint2 g2 [[threadgroup_position_in_grid]]) {
    // the inputs split over ksplit groups; each fills its share of 8 piles; the shares are added after
    uint grp = g2.x * 8 + sg; if (grp * 8 >= nout) return;
    uint kk = g2.y, span = nin / ksplit, i0 = kk * span;
    device const short4* w = W + grp * nin * 2; device const int* x = xt + lane;
    int lo[8] = {0, 0, 0, 0, 0, 0, 0, 0}, hi[8] = {0, 0, 0, 0, 0, 0, 0, 0};   // two narrow cells per pile: written, never carried
    for (uint i = i0; i < i0 + span; i++) {
        int xv = x[i * 32]; short4 lw = w[i * 2], hw = w[i * 2 + 1];
        int pr[8] = {lw.x * xv, lw.y * xv, lw.z * xv, lw.w * xv, hw.x * xv, hw.y * xv, hw.z * xv, hw.w * xv};
        for (int k = 0; k < 8; k++) { lo[k] += pr[k] & 0x7FFF; hi[k] += pr[k] >> 15; } }          // bottom 15 digits into one cell, the rest into the other
    long acc[8]; for (int k = 0; k < 8; k++) acc[k] = ((long)hi[k] << 15) + lo[k];                // the one read: high x 2^15 + low
    if (lane < np) for (int k = 0; k < 8; k++) if (grp * 8 + k < nout) part[(kk * 32 + lane) * nout + grp * 8 + k] = acc[k];
}
kernel void addparts(device const long* part [[buffer(0)]], device long* pile [[buffer(1)]], constant uint& nout [[buffer(2)]], constant uint& np [[buffer(3)]],
                     constant uint& ksplit [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= np * nout) return; uint lane = i / nout, j = i % nout; long s = 0;
    for (uint k = 0; k < ksplit; k++) s += part[(k * 32 + lane) * nout + j]; pile[i] = s; }

kernel void gates_tg(device const short4* W [[buffer(0)]], device const int* xt [[buffer(1)]], device long* pile [[buffer(2)]],
                     constant uint& nin [[buffer(3)]], constant uint& nout [[buffer(4)]], constant uint& np [[buffer(5)]],
                     uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint g [[threadgroup_position_in_grid]]) {
    // one core, 64 piles: each chunk of 64 inputs (all 32 positions) is loaded into the core's own memory once and
    // every pile reads it from there; products go into two narrow cells per pile, read once at the end
    threadgroup int xs[64 * 32];
    uint grp = g * 8 + sg; bool live = grp * 8 < nout;
    device const short4* w = W + grp * nin * 2;
    int lo[8] = {0, 0, 0, 0, 0, 0, 0, 0}, hi[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint c = 0; c < nin; c += 64) {
        for (uint k = t; k < 64 * 32; k += 256) xs[k] = xt[c * 32 + k];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (live) for (uint i = 0; i < 64; i++) {
            int xv = xs[i * 32 + lane]; short4 lw = w[(c + i) * 2], hw = w[(c + i) * 2 + 1];
            int pr[8] = {lw.x * xv, lw.y * xv, lw.z * xv, lw.w * xv, hw.x * xv, hw.y * xv, hw.z * xv, hw.w * xv};
            for (int k = 0; k < 8; k++) { lo[k] += pr[k] & 0x7FFF; hi[k] += pr[k] >> 15; } }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (live && lane < np) for (int k = 0; k < 8; k++) if (grp * 8 + k < nout) pile[lane * nout + grp * 8 + k] = ((long)hi[k] << 15) + lo[k];
}

kernel void ones(device const short4* W [[buffer(0)]], device const int* xt [[buffer(1)]], device long* part [[buffer(2)]],
                 constant uint& nin [[buffer(3)]], constant uint& nout [[buffer(4)]], constant uint& np [[buffer(5)]], constant uint& ksplit [[buffer(6)]],
                 uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint2 g2 [[threadgroup_position_in_grid]]) {
    // a weight is only its single ones. Each 1 adds the input into the cell of its place; zeros do nothing.
    // One cell per place (16, the top one negative); the cells are read once: pile = sum of cell x 2^place.
    uint grp = g2.x * 8 + sg; if (grp * 8 >= nout) return;
    uint kk = g2.y, span = nin / ksplit, i0 = kk * span;
    device const short4* w = W + grp * nin * 2; device const int* x = xt + lane;
    for (int k8 = 0; k8 < 8; k8 += 2) {                              // 2 piles at a time: 2 x 16 cells
        int ca[16], cb[16];
        for (int j = 0; j < 16; j++) { ca[j] = 0; cb[j] = 0; }
        for (uint i = i0; i < i0 + span; i++) {
            int xv = x[i * 32]; short4 q = w[i * 2 + (k8 >> 2)];
            uint wa = (uint)(ushort)(k8 % 4 == 0 ? q.x : q.z), wb = (uint)(ushort)(k8 % 4 == 0 ? q.y : q.w);
            while (wa) { int j = ctz(wa); ca[j] += xv; wa &= wa - 1; }      // each single 1: the input lands in its place's cell
            while (wb) { int j = ctz(wb); cb[j] += xv; wb &= wb - 1; } }
        long pa = 0, pb = 0;
        for (int j = 0; j < 15; j++) { pa += (long)ca[j] << j; pb += (long)cb[j] << j; }
        pa -= (long)ca[15] << 15; pb -= (long)cb[15] << 15;              // the top place is the negative one (two's complement)
        if (lane < np) { if (grp * 8 + k8 < nout) part[(kk * 32 + lane) * nout + grp * 8 + k8] = pa;
                         if (grp * 8 + k8 + 1 < nout) part[(kk * 32 + lane) * nout + grp * 8 + k8 + 1] = pb; }
    }
}
kernel void probe8k(device const short4* W [[buffer(0)]], device const int* xt [[buffer(1)]], device long* part [[buffer(2)]],
                    constant uint& nin [[buffer(3)]], constant uint& nout [[buffer(4)]], constant uint& np [[buffer(5)]], constant uint& ksplit [[buffer(6)]],
                    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint2 g2 [[threadgroup_position_in_grid]]) {
    // the inputs split over ksplit groups; each fills its share of 8 piles; the shares are added after
    uint grp = g2.x * 8 + sg; if (grp * 8 >= nout) return;
    uint kk = g2.y, span = nin / ksplit, i0 = kk * span;
    device const short4* w = W + grp * nin * 2; device const int* x = xt + lane;
    int lo[8] = {0, 0, 0, 0, 0, 0, 0, 0}, hi[8] = {0, 0, 0, 0, 0, 0, 0, 0};   // two narrow cells per pile: written, never carried
    for (uint i = i0; i < i0 + span; i++) {
        int xv = x[i * 32]; short4 lw = w[i * 2], hw = w[i * 2 + 1];
        lo[0] += lw.x + xv; lo[1] += lw.y + xv; lo[2] += lw.z + xv; lo[3] += lw.w + xv;                // same reads, one add each (no product, no split)
        lo[4] += hw.x + xv; lo[5] += hw.y + xv; lo[6] += hw.z + xv; lo[7] += hw.w + xv; }
    long acc[8]; for (int k = 0; k < 8; k++) acc[k] = ((long)hi[k] << 15) + lo[k];                // the one read: high x 2^15 + low
    if (lane < np) for (int k = 0; k < 8; k++) if (grp * 8 + k < nout) part[(kk * 32 + lane) * nout + grp * 8 + k] = acc[k];
}

template <int G> inline void gatesG(device const short* W, device const int* xt, device long* part, uint nin, uint nout, uint np, uint ksplit,
                                     uint lane, uint grp, uint kk) {
    // one read of each input fills G piles; products into two narrow cells per pile, read once
    if (grp * G >= nout) return;
    uint span = nin / ksplit, i0 = kk * span;
    device const short* w = W + grp * nin * G; device const int* x = xt + lane;
    int lo[G], hi[G]; for (int k = 0; k < G; k++) { lo[k] = 0; hi[k] = 0; }
    for (uint i = i0; i < i0 + span; i++) { int xv = x[i * 32]; device const short* wi = w + i * G;
        for (int k = 0; k < G; k++) { int pr = wi[k] * xv; lo[k] += pr & 0x7FFF; hi[k] += pr >> 15; } }
    if (lane < np) for (int k = 0; k < G; k++) if (grp * G + k < nout) part[(kk * 32 + lane) * nout + grp * G + k] = ((long)hi[k] << 15) + lo[k];
}
kernel void gates16(device const short* W [[buffer(0)]], device const int* xt [[buffer(1)]], device long* part [[buffer(2)]], constant uint& nin [[buffer(3)]],
                    constant uint& nout [[buffer(4)]], constant uint& np [[buffer(5)]], constant uint& ks [[buffer(6)]],
                    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint2 g2 [[threadgroup_position_in_grid]]) {
    gatesG<16>(W, xt, part, nin, nout, np, ks, lane, g2.x * 8 + sg, g2.y); }
kernel void gates32(device const short* W [[buffer(0)]], device const int* xt [[buffer(1)]], device long* part [[buffer(2)]], constant uint& nin [[buffer(3)]],
                    constant uint& nout [[buffer(4)]], constant uint& np [[buffer(5)]], constant uint& ks [[buffer(6)]],
                    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint2 g2 [[threadgroup_position_in_grid]]) {
    gatesG<32>(W, xt, part, nin, nout, np, ks, lane, g2.x * 8 + sg, g2.y); }
kernel void gauge_log2(device const int* x16 [[buffer(0)]], device const int* ex [[buffer(1)]], device long* LS [[buffer(2)]],
                       device const long* LOG2 [[buffer(3)]], uint lane [[thread_index_in_simdgroup]], uint p [[threadgroup_position_in_grid]]) {
    // the position's size, 32 lanes together: sums of the values and of their squares (the gate grid), then the log
    long s1 = 0, s2 = 0;
    for (uint i = lane; i < 768; i += 32) { long v = x16[p * 768 + i]; s1 += v; s2 += v * v; }
    s1 = sum64(s1); s2 = sum64(s2);
    if (lane == 0) { long V = 768 * s2 - s1 * s1; int sh = 40 + 2 * ex[p]; V += sh >= 0 ? (EPS40 >> sh) : (EPS40 << -sh);
        LS[p] = (log2fix((ulong)V, LOG2) >> 1) - LOG768 + (long)ex[p] * 65536; }
}
kernel void attend2(device const long* QKV [[buffer(0)]], device const int* q16 [[buffer(1)]], device const int* eq [[buffer(2)]], device long* out [[buffer(3)]],
                    device const long* ARR [[buffer(4)]], device const long* LOG2 [[buffer(5)]], device const long* EXP2 [[buffer(6)]],
                    uint lane [[thread_index_in_simdgroup]], uint g [[threadgroup_position_in_grid]]) {
    // one head at one position, 32 lanes together, two of the 64 values each
    uint p = g / 12, h = g % 12; long sc[32];
    for (uint s = 0; s <= p; s++) {
        long dot = (long)q16[p * 2304 + h * 64 + lane] * q16[s * 2304 + 768 + h * 64 + lane]
                 + (long)q16[p * 2304 + h * 64 + lane + 32] * q16[s * 2304 + 768 + h * 64 + lane + 32];
        dot = sum64(dot);
        sc[s] = rsh(dot, -((long)eq[p * 36 + h] + eq[s * 36 + 12 + h] - 3 + 10)); }
    uint best = 0; for (uint s = 1; s <= p; s++) if (sc[s] > sc[best]) best = s;           // the race between the scores
    long start = sc[best], num0 = 0, num1 = 0, Z = 0;
    for (uint s = 0; s <= p; s++) { long dist = start - sc[s]; long a = dist < ALEN ? ARR[dist] : 0; if (a == 0) continue; Z += a;
        num0 += a * QKV[s * 2304 + 1536 + h * 64 + lane]; num1 += a * QKV[s * 2304 + 1536 + h * 64 + lane + 32]; }
    long F = -log2fix((ulong)Z, LOG2); long I = F >> 16, f = F & 0xFFFF;
    out[p * 768 + h * 64 + lane] = rsh(num0 * EXP2[f], 16 - I); out[p * 768 + h * 64 + lane + 32] = rsh(num1 * EXP2[f], 16 - I); }
kernel void race2(device const long* lg [[buffer(0)]], device int* word [[buffer(1)]], constant uint& V [[buffer(2)]],
                  uint t [[thread_position_in_threadgroup]], uint p [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    // the race over 50,257 words: pairs raced side by side, winners raced again, down to one
    threadgroup long bv[32]; threadgroup int bi[32]; device const long* v = lg + p * V;
    long b = LONG_MIN; int bw = 0;
    for (uint i = t; i < V; i += 1024) if (v[i] > b) { b = v[i]; bw = (int)i; }
    for (uint o = 16; o > 0; o >>= 1) { long ob = shfl_down(b, o); int ow = simd_shuffle_down(bw, (ushort)o); if (ob > b || (ob == b && ow < bw)) { b = ob; bw = ow; } }
    if (lane == 0) { bv[sg] = b; bi[sg] = bw; } threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) { long B = bv[0]; int W = bi[0]; for (int k = 1; k < 32; k++) if (bv[k] > B || (bv[k] == B && bi[k] < W)) { B = bv[k]; W = bi[k]; } word[p] = W; }
}
kernel void gauge_log(device const int* x16 [[buffer(0)]], device const int* ex [[buffer(1)]], device long* LS [[buffer(2)]],
                      device const long* LOG2 [[buffer(3)]], uint p [[thread_position_in_grid]]) {
    // the position's size, never computed: 768 x (sum of squares) - (sum)^2, its log halved, minus log2 768
    long s1 = 0, s2 = 0;
    for (uint i = 0; i < 768; i++) { long v = x16[p * 768 + i]; s1 += v; s2 += times(v < 0 ? -v : v, (ulong)(v < 0 ? -v : v)); }
    long V = (s2 << 9) + (s2 << 8) - times(s1 < 0 ? -s1 : s1, (ulong)(s1 < 0 ? -s1 : s1));
    int sh = 40 + 2 * ex[p]; V += sh >= 0 ? (EPS40 >> sh) : (EPS40 << -sh);
    LS[p] = (log2fix((ulong)V, LOG2) >> 1) - LOG768 + (long)ex[p] * 65536;
}
kernel void gauged(device const long* pile [[buffer(0)]], device const int* we [[buffer(1)]], device const int* ex [[buffer(2)]], device const long* LS [[buffer(3)]],
                   device const long* bias [[buffer(4)]], device long* out [[buffer(5)]], device const long* EXP2 [[buffer(6)]],
                   constant uint& nout [[buffer(7)]], constant uint& n [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return; uint p = i / nout, c = i % nout;
    long F = ((long)we[c] + ex[p] + 16) * 65536 - LS[p];
    out[i] = power(pile[i], F, EXP2) + bias[c]; }
kernel void plain(device const long* pile [[buffer(0)]], device const int* we [[buffer(1)]], device const int* ex [[buffer(2)]],
                  device const long* bias [[buffer(3)]], device long* R [[buffer(4)]], constant uint& nout [[buffer(5)]], constant uint& n [[buffer(6)]],
                  uint i [[thread_position_in_grid]]) {
    if (i >= n) return; uint p = i / nout, c = i % nout;
    R[i] += rsh(pile[i], -((long)we[c] + ex[p] + 16)) + bias[c]; }          // shifted onto the stream, added to it
kernel void edge(device long* H [[buffer(0)]], device const long* PHI [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return; long h = H[i]; long idx = rsh(h, 4) + 32768;
    long share = idx < 0 ? 0 : (idx > 65535 ? 65536 : PHI[idx]);
    H[i] = rsh(h * share, 16); }
kernel void heads16(device const long* QKV [[buffer(0)]], device int* q16 [[buffer(1)]], device int* eq [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    // each position's query, key and value of each head on 16 bits, each with its own power of two (OR of magnitudes)
    uint p = id / 36, part = id % 36;                    // 3 parts x 12 heads
    device const long* x = QKV + p * 2304 + part * 64; ulong o = 0;
    for (uint d = 0; d < 64; d++) { long v = x[d]; o |= (ulong)(v < 0 ? -v : v); }
    int top = o ? 64 - clz(o) : 0; int sh = max(top - 15, 0);
    for (uint d = 0; d < 64; d++) q16[p * 2304 + part * 64 + d] = (int)rsh(x[d], sh);
    eq[p * 36 + part] = -16 + sh; }
kernel void attend(device const long* QKV [[buffer(0)]], device const int* q16 [[buffer(1)]], device const int* eq [[buffer(2)]], device long* out [[buffer(3)]],
                   device const long* ARR [[buffer(4)]], device const long* LOG2 [[buffer(5)]], device const long* EXP2 [[buffer(6)]],
                   uint id [[thread_position_in_grid]]) {
    uint p = id / 12, h = id % 12; long sc[32];
    for (uint s = 0; s <= p; s++) {                      // query x key, as shift-adds; the 1/8 is a shift of 3; onto 2^-10
        long dot = 0; for (uint d = 0; d < 64; d++) dot += times_s((long)q16[p * 2304 + h * 64 + d], (long)q16[s * 2304 + 768 + h * 64 + d]);
        sc[s] = rsh(dot, -((long)eq[p * 36 + h] + eq[s * 36 + 12 + h] - 3 + 10)); }
    uint alive = p == 31 ? 0xFFFFFFFFu : ((1u << (p + 1)) - 1);         // the race: top digits first
    for (int bit = 63; bit >= 0; bit--) { uint has = 0;
        for (uint s = 0; s <= p; s++) if ((alive >> s) & 1) if (((((ulong)sc[s]) ^ (1UL << 63)) >> bit) & 1) has |= 1u << s;
        if (has) alive = has; }
    long start = sc[ctz(alive)];
    long num[64]; for (uint d = 0; d < 64; d++) num[d] = 0; long Z = 0;
    for (uint s = 0; s <= p; s++) { long dist = start - sc[s]; long a = dist < ALEN ? ARR[dist] : 0; if (a == 0) continue; Z += a;
        for (uint d = 0; d < 64; d++) num[d] += times(QKV[s * 2304 + 1536 + h * 64 + d], (ulong)a); }
    long LZ = log2fix((ulong)Z, LOG2);
    for (uint d = 0; d < 64; d++) out[p * 768 + h * 64 + d] = power(num[d], -LZ, EXP2); }
kernel void race_words(device const long* lg [[buffer(0)]], device uchar* alive [[buffer(1)]], device int* word [[buffer(2)]],
                       constant uint& V [[buffer(3)]], uint t [[thread_position_in_threadgroup]], uint p [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    // the word: a race over 50,257 piles, top digits first
    threadgroup uint any[32]; device const long* v = lg + p * V; device uchar* al = alive + p * V;
    for (uint i = t; i < V; i += 1024) al[i] = 1;
    for (int bit = 63; bit >= 0; bit--) {
        uint has = 0; for (uint i = t; i < V; i += 1024) if (al[i] && (((((ulong)v[i]) ^ (1UL << 63)) >> bit) & 1)) has = 1;
        has = simd_or(has); if (lane == 0) any[sg] = has; threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        uint a = 0; for (int k = 0; k < 32; k++) a |= any[k];
        if (a) for (uint i = t; i < V; i += 1024) if (al[i] && !(((((ulong)v[i]) ^ (1UL << 63)) >> bit) & 1)) al[i] = 0;
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup); }
    if (t == 0) { int w = -1; for (uint i = 0; i < V && w < 0; i++) if (al[i]) w = (int)i; word[p] = w; }
}
kernel void copyl(device const long* a [[buffer(0)]], device long* b [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) { if (i < n) b[i] = a[i]; }
kernel void moved(device const long* now [[buffer(0)]], device long* last [[buffer(1)]], device atomic_int* cnt [[buffer(2)]], constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return; if (now[i] != last[i]) { atomic_fetch_add_explicit(cnt, 1, memory_order_relaxed); atomic_fetch_add_explicit(cnt + 1 + i / 768, 1, memory_order_relaxed); } last[i] = now[i]; }
"""
setvbuf(stdout, nil, _IONBF, 0)
let dev = MTLCreateSystemDefaultDevice()!, queue = dev.makeCommandQueue()!
let lib = try! dev.makeLibrary(source: src, options: MTLCompileOptions())
func pipe(_ n: String) -> MTLComputePipelineState { return try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
let pTo16 = pipe("to16"), pPack = pipe("pack"), pPlanes = pipe("planes"), pGates = pipe(ARG["kernel"] == "gates1" ? "gates" : "gates4"), pGates8 = pipe("gates8"), pGates8k = pipe("gates8k"), pGatesTG = pipe("gates_tg"), pOnes = pipe("ones"), pProbe = pipe("probe8k"), pG16 = pipe("gates16"), pG32 = pipe("gates32"), PROBE = ARG["kernel"] == "probe", ONES = ARG["kernel"] == "ones", TG = ARG["kernel"] == "tg", ONE = ARG["one"] == "1", pAddParts = pipe("addparts"), G8 = ARG["kernel"] == nil || ARG["kernel"] == "gates8" || ARG["kernel"] == "probe", pTpose = pipe("tpose"), pGaugeLog2 = pipe("gauge_log2"), pAttend2 = pipe("attend2"), pRace2 = pipe("race2"), GATES = ARG["kernel"] != "planes", pGaugeLog = pipe("gauge_log"), pGauged = pipe("gauged"), pPlain = pipe("plain")
let pEdge = pipe("edge"), pHeads = pipe("heads16"), pAttend = pipe("attend"), pRace = pipe("race_words"), pCopy = pipe("copyl"), pMoved = pipe("moved")
func loadData(_ name: String) -> Data { return try! Data(contentsOf: URL(fileURLWithPath: dir + name)) }
func bufData(_ d: Data) -> MTLBuffer { return d.withUnsafeBytes { dev.makeBuffer(bytes: $0.baseAddress!, length: d.count, options: .storageModeShared)! } }
func zbuf(_ bytes: Int) -> MTLBuffer { return dev.makeBuffer(length: max(bytes, 8), options: .storageModeShared)! }
func i64(_ name: String) -> [Int64] { let d = loadData(name); return d.withUnsafeBytes { Array($0.bindMemory(to: Int64.self)) } }

let GP = argInt("piles", 8)
struct Bank { let nin, nout: Int; let planes, e, b, w16, w8, wg: MTLBuffer }
var banks = [String: Bank]()
for b in man["banks"] as! [[String: Any]] {
    let n = b["name"] as! String
    banks[n] = Bank(nin: b["nin"] as! Int, nout: b["nout"] as! Int, planes: bufData(loadData("\(n).planes")), e: bufData(loadData("\(n).e")), b: bufData(loadData("\(n).b")), w16: bufData(loadData("\(n).w16")), w8: bufData(loadData("\(n).w8")), wg: bufData(loadData("\(n).w\(GP)")))
}
let PHI = bufData(loadData("phi.bin")), ARR = bufData(loadData("arrival.bin")), LOG2 = bufData(loadData("log2.bin")), EXP2 = bufData(loadData("exp2.bin"))
let wte = i64("wte.bin"), wpe = i64("wpe.bin")

let ref = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: NSString(string: ARG["refs"] ?? "~/tick-machine/gpt2_dial/reference.json").expandingTildeInPath))) as! [[String: Any]]
let L = 12, D = 768, V = 50257, PMAX = 32
let vocab: [Int: String] = {
    let base = NSString(string: "~/.cache/huggingface/hub/models--gpt2/snapshots").expandingTildeInPath
    let j = try! JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: base + "/" + (try! FileManager.default.contentsOfDirectory(atPath: base)).first! + "/vocab.json"))) as! [String: Int]
    var v = [Int: String](); for (k, i) in j { v[i] = k.replacingOccurrences(of: "Ġ", with: " ").replacingOccurrences(of: "Ċ", with: "\\n") }; return v }()
func text(_ t: [Int]) -> String { return "'" + t.map { vocab[$0] ?? "?" }.joined() + "'" }


let R = (0...L).map { _ in zbuf(PMAX * D * 8) }, last = (0...L).map { _ in zbuf(PMAX * D * 8) }
let x16 = zbuf(PMAX * 3072 * 4), ex = zbuf(PMAX * 4), XP = zbuf(16 * 96 * 32 * 4), pile = zbuf(PMAX * V * 8), LS = zbuf(PMAX * 8)
let qkv = zbuf(PMAX * 2304 * 8), q16 = zbuf(PMAX * 2304 * 4), eq = zbuf(PMAX * 36 * 4), att = zbuf(PMAX * D * 8), hid = zbuf(PMAX * 3072 * 8), lg = zbuf(PMAX * V * 8)
let xt = zbuf(32 * 3072 * 4), parts = zbuf(64 * 32 * 3072 * 8), partsV = zbuf(8 * 32 * 50264 * 8), alive = zbuf(PMAX * V), words = zbuf(PMAX * 4), cnt = zbuf((PMAX + 1) * 4)
var P = 0

var shared: MTLComputeCommandEncoder? = nil, sharedCB: MTLCommandBuffer? = nil
func encoder(_ cb: MTLCommandBuffer) -> MTLComputeCommandEncoder {            // one encoder for a whole step: dispatches still run in order
    if ONE, let e = shared, sharedCB === cb { return e }
    let e = cb.makeComputeCommandEncoder(dispatchType: .serial)!; if ONE { shared = e; sharedCB = cb }; return e }
func endShared() { shared?.endEncoding(); shared = nil; sharedCB = nil }
func enc(_ cb: MTLCommandBuffer, _ ps: MTLComputePipelineState, _ bufs: [MTLBuffer], _ consts: [Int32], threads: Int, tg: Int, groups: Int = 0) {
    let e = encoder(cb); e.setComputePipelineState(ps)
    for (i, b) in bufs.enumerated() { e.setBuffer(b, offset: 0, index: i) }
    for (k, c) in consts.enumerated() { var v = c; e.setBytes(&v, length: 4, index: bufs.count + k) }
    if groups > 0 { e.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)) }
    else { e.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(tg, threads), height: 1, depth: 1)) }
    if !ONE { e.endEncoding() }
}
func onto16(_ cb: MTLCommandBuffer, _ X: MTLBuffer, n: Int, stride: Int) {           // every position on 16 bits, then into digit planes
    enc(cb, pTo16, [X, x16, ex], [Int32(n), -16, Int32(stride), 0], threads: 0, tg: 256, groups: P)
    if !GATES { enc(cb, pPack, [x16, XP], [Int32(n), Int32(P)], threads: 16 * (n / 32) * 32, tg: 256) }
}
func planesOf(_ cb: MTLCommandBuffer, _ bk: Bank) {
    if GATES && TG { enc(cb, pTpose, [x16, xt], [Int32(bk.nin), Int32(P)], threads: bk.nin * 32, tg: 256)
        enc(cb, pGatesTG, [bk.w8, xt, pile], [Int32(bk.nin), Int32(bk.nout), Int32(P)], threads: 0, tg: 256, groups: (bk.nout + 63) / 64); return }
    if GATES && (G8 || ONES) && (bk.nout <= 3072 || ONES) {
        let ks = bk.nout > 3072 ? 2 : bk.nin / argInt("span", 192)                          // short chunks of inputs, every chunk its own group of lanes
        enc(cb, pTpose, [x16, xt], [Int32(bk.nin), Int32(P)], threads: bk.nin * 32, tg: 256)
        let e = encoder(cb); e.setComputePipelineState(GP == 16 ? pG16 : (GP == 32 ? pG32 : (PROBE ? pProbe : (ONES ? pOnes : pGates8k))))
        for (i, b) in [GP == 8 ? bk.w8 : bk.wg, xt, bk.nout > 3072 ? partsV : parts].enumerated() { e.setBuffer(b, offset: 0, index: i) }
        for (k, c) in [Int32(bk.nin), Int32(bk.nout), Int32(P), Int32(ks)].enumerated() { var v = c; e.setBytes(&v, length: 4, index: 3 + k) }
        e.dispatchThreadgroups(MTLSize(width: (bk.nout + 8 * GP - 1) / (8 * GP), height: ks, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)); if !ONE { e.endEncoding() }
        enc(cb, pAddParts, [bk.nout > 3072 ? partsV : parts, pile], [Int32(bk.nout), Int32(P), Int32(ks)], threads: P * bk.nout, tg: 256); return }
    if GATES && G8 { enc(cb, pTpose, [x16, xt], [Int32(bk.nin), Int32(P)], threads: bk.nin * 32, tg: 256)
        enc(cb, pGates8, [bk.w8, xt, pile], [Int32(bk.nin), Int32(bk.nout), Int32(P)], threads: 0, tg: 256, groups: (bk.nout + 63) / 64); return }
    if GATES { enc(cb, pTpose, [x16, xt], [Int32(bk.nin), Int32(P)], threads: bk.nin * 32, tg: 256); enc(cb, pGates, [bk.w16, xt, pile], [Int32(bk.nin), Int32(bk.nout), Int32(P)], threads: 0, tg: 256, groups: (bk.nout + 7) / 8); return }
    enc(cb, pPlanes, [bk.planes, XP, pile], [Int32(bk.nin), Int32(bk.nout), Int32(P)], threads: 0, tg: 256, groups: (bk.nout + 7) / 8)
}
func layer(_ cb: MTLCommandBuffer, _ d: Int) {                                       // layer d: R[d] -> R[d+1], every position
    let bq = banks["qkv.\(d)"]!, bp = banks["proj.\(d)"]!, bf = banks["fc.\(d)"]!, bm = banks["mproj.\(d)"]!
    let nD = Int32(P * D)
    onto16(cb, R[d], n: D, stride: D); enc(cb, pGaugeLog2, [x16, ex, LS, LOG2], [], threads: 0, tg: 32, groups: P)
    planesOf(cb, bq); enc(cb, pGauged, [pile, bq.e, ex, LS, bq.b, qkv, EXP2], [Int32(2304), Int32(P * 2304)], threads: P * 2304, tg: 256)
    enc(cb, pHeads, [qkv, q16, eq], [], threads: P * 36, tg: 36)
    enc(cb, pAttend2, [qkv, q16, eq, att, ARR, LOG2, EXP2], [], threads: 0, tg: 32, groups: P * 12)
    enc(cb, pCopy, [R[d], R[d + 1]], [nD], threads: P * D, tg: 256)
    onto16(cb, att, n: D, stride: D); planesOf(cb, bp)
    enc(cb, pPlain, [pile, bp.e, ex, bp.b, R[d + 1]], [Int32(D), nD], threads: P * D, tg: 256)
    onto16(cb, R[d + 1], n: D, stride: D); enc(cb, pGaugeLog2, [x16, ex, LS, LOG2], [], threads: 0, tg: 32, groups: P)
    planesOf(cb, bf); enc(cb, pGauged, [pile, bf.e, ex, LS, bf.b, hid, EXP2], [Int32(3072), Int32(P * 3072)], threads: P * 3072, tg: 256)
    enc(cb, pEdge, [hid, PHI], [Int32(P * 3072)], threads: P * 3072, tg: 256)
    onto16(cb, hid, n: 3072, stride: 3072); planesOf(cb, bm)
    enc(cb, pPlain, [pile, bm.e, ex, bm.b, R[d + 1]], [Int32(D), nD], threads: P * D, tg: 256)
}
func readTop(_ cb: MTLCommandBuffer) {
    let bh = banks["head"]!
    onto16(cb, R[L], n: D, stride: D); enc(cb, pGaugeLog2, [x16, ex, LS, LOG2], [], threads: 0, tg: 32, groups: P)
    planesOf(cb, bh); enc(cb, pGauged, [pile, bh.e, ex, LS, bh.b, lg, EXP2], [Int32(V), Int32(P * V)], threads: P * V, tg: 256)
    enc(cb, pRace2, [lg, words], [Int32(V)], threads: 0, tg: 1024, groups: P)
}



if ARG["profile"] != nil {                                   // time each part of one step on the GPU
    let ids0 = ref[CASE]["ids"] as! [Int]; P = min(ids0.count + WORDS, 32)
    var T0 = ids0 + [Int](repeating: ids0.last!, count: P - ids0.count)
    let r0 = R[0].contents().bindMemory(to: Int64.self, capacity: P * D)
    for p in 0..<P { for i in 0..<D { r0[p * D + i] = wte[T0[p] * D + i] + wpe[p * D + i] } }
    func timed(_ name: String, _ body: (MTLCommandBuffer) -> Void) {
        for rep in 0..<2 { let cb = queue.makeCommandBuffer()!; body(cb); endShared(); cb.commit(); cb.waitUntilCompleted()
            if rep == 1 { print(String(format: "  %-44@ %8.2f ms", name, (cb.gpuEndTime - cb.gpuStartTime) * 1000)) } }
    }
    let bq = banks["qkv.0"]!, bf = banks["fc.0"]!, bm = banks["mproj.0"]!, bh = banks["head"]!
    print("one step, \(P) positions:")
    timed("whole step (12 layers + the top)") { cb in for d in 0..<L { layer(cb, d) }; readTop(cb) }
    timed("one layer") { cb in layer(cb, 0) }
    timed("25 do-nothing operations in a row (one layer's count)") { cb in for _ in 0..<25 { enc(cb, pCopy, [R[0], R[1]], [1], threads: 1, tg: 1) } }
    timed("300 do-nothing operations in a row (one step's count)") { cb in for _ in 0..<300 { enc(cb, pCopy, [R[0], R[1]], [1], threads: 1, tg: 1) } }
    timed("300 do-nothing, one encoder") { cb in let e = cb.makeComputeCommandEncoder(dispatchType: .serial)!; e.setComputePipelineState(pCopy); e.setBuffer(R[0], offset: 0, index: 0); e.setBuffer(R[1], offset: 0, index: 1); var one: Int32 = 1; e.setBytes(&one, length: 4, index: 2)
        for _ in 0..<300 { e.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)) }; e.endEncoding() }
    timed("  onto 16 bits + digit planes (768)") { cb in onto16(cb, R[0], n: D, stride: D) }
    timed("  bit planes, qkv bank (768 -> 2304)") { cb in planesOf(cb, bq) }
    timed("  bit planes, fc bank (768 -> 3072)") { cb in planesOf(cb, bf) }
    timed("  bit planes, mproj bank (3072 -> 768)") { cb in onto16(cb, hid, n: 3072, stride: 3072); planesOf(cb, bm) }
    timed("  the gauge (log of the size)") { cb in enc(cb, pGaugeLog2, [x16, ex, LS, LOG2], [], threads: 0, tg: 32, groups: P) }
    timed("  attention (heads16 + attend)") { cb in enc(cb, pHeads, [qkv, q16, eq], [], threads: P * 36, tg: 36); enc(cb, pAttend2, [qkv, q16, eq, att, ARR, LOG2, EXP2], [], threads: 0, tg: 32, groups: P * 12) }
    timed("the top: bit planes over 50,257 words") { cb in onto16(cb, R[L], n: D, stride: D); planesOf(cb, bh) }
    timed("the top: gauge on 50,257 words") { cb in enc(cb, pGauged, [pile, bh.e, ex, LS, bh.b, lg, EXP2], [Int32(V), Int32(P * V)], threads: P * V, tg: 256) }
    timed("the top: the race over 50,257 words") { cb in enc(cb, pRace2, [lg, words], [Int32(V)], threads: 0, tg: 1024, groups: P) }
    exit(0)
}
let CASES = ARG["all"] != nil ? Array(0..<ref.count) : [CASE]
var nIdentical = 0, report = [String]()
for CASE in CASES {
let ids = ref[CASE]["ids"] as! [Int], prompt = ref[CASE]["prompt"] as! String
let N = min(WORDS, (ref[CASE]["ref"] as! [Int]).count, 32 - ids.count), want = Array((ref[CASE]["ref"] as! [Int]).prefix(N)), S = ids.count; P = S + N
for k in 0...L { memset(R[k].contents(), 0, PMAX * D * 8); memset(last[k].contents(), 0, PMAX * D * 8) }
print("the all-base-2 soup on \(dev.name): whole numbers only; \(P) positions x \(L) layers, all at once")
    print("'\(prompt)' + \(N) words\nmodel : \(text(want))")
print("")
    var T = ids + [Int](repeating: ids.last!, count: N), firstRight = [Int?](repeating: nil, count: N), rest: Int? = nil, step = 0
    let t0 = Date()
    while step < MAXSTEPS {
        step += 1
        let r0 = R[0].contents().bindMemory(to: Int64.self, capacity: P * D)
        for p in 0..<P { for i in 0..<D { r0[p * D + i] = wte[T[p] * D + i] + wpe[p * D + i] } }
        memset(cnt.contents(), 0, (PMAX + 1) * 4)
        let cb = queue.makeCommandBuffer()!
        for d in 0..<L { layer(cb, d) }                                                   // bottom to top, each reading what is below it now
        for k in 1...L { enc(cb, pMoved, [R[k], last[k], cnt], [Int32(P * D)], threads: P * D, tg: 256) }
        readTop(cb)
        cb.commit(); cb.waitUntilCompleted()
        let wd = words.contents().bindMemory(to: Int32.self, capacity: P)
        let newT = ids + (0..<N).map { Int(wd[S - 1 + $0]) }
        let changed = zip(newT, T).filter { $0 != $1 }.count; T = newT
        let movedNow = Int(cnt.contents().bindMemory(to: Int32.self, capacity: 1)[0])
        let right = (0..<N).map { T[S + $0] == want[$0] }
        for j in 0..<N { if right[j] { if firstRight[j] == nil { firstRight[j] = step } } else { firstRight[j] = nil } }
        print(String(format: "  step %3d: %2d/%d right, %2d changed, %6d values moving   ", step, right.filter { $0 }.count, N, changed, movedNow) + text(Array(T[S...])))
        if movedNow == 0 && changed == 0 { rest = step; break }
    }
    let wall = Date().timeIntervalSince(t0)
    print("\n\(rest != nil ? "came to rest at step \(rest!)" : "still moving after \(MAXSTEPS) steps"); \(Array(T[S...]) == want ? "IDENTICAL" : "DIFFERENT") to the model")
    print("each word right from step: " + firstRight.map { $0 == nil ? "-" : String($0!) }.joined(separator: " "))
    print(String(format: "%.2f s, %.0f ms a step, %.1f words/s", wall, wall / Double(step) * 1000, Double(N) / wall))
    
    if Array(T[S...]) == want { nIdentical += 1 } else {
        let j = (0..<N).first { T[S + $0] != want[$0] }!
        let mg = (ref[CASE]["margins"] as? [Double]).map { String(format: "%.4f", $0[j]) } ?? "?"
        report.append("  '\(prompt)': first differs at word \(j + 1): \(text([T[S + j]])) where the model says \(text([want[j]])) (the model's own lead there: \(mg))") }
}
if CASES.count > 1 { print("\n\(nIdentical) of \(CASES.count) prompts identical to the model"); report.forEach { print($0) } }
