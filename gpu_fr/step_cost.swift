// The GPU time of one dependent step that does almost nothing (each copies one number from the last), 2,000 in a row.
//   swiftc -O -o gpu_fr/step_cost gpu_fr/step_cost.swift && ./gpu_fr/step_cost
import Foundation
import Metal
let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
let lib = try! dev.makeLibrary(source: "#include <metal_stdlib>\nusing namespace metal;\nkernel void step(device long* a [[buffer(0)]], uint i [[thread_position_in_grid]]) { if (i == 0) a[1] = a[0] + 1; a[0] = a[1]; }", options: nil)
let ps = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "step")!)
let b = dev.makeBuffer(length: 64, options: .storageModeShared)!
func run(_ n: Int, one: Bool, threads: Int) -> Double {
    let cb = q.makeCommandBuffer()!; var e: MTLComputeCommandEncoder? = one ? cb.makeComputeCommandEncoder(dispatchType: .serial) : nil
    for _ in 0..<n { let x = e ?? cb.makeComputeCommandEncoder()!; x.setComputePipelineState(ps); x.setBuffer(b, offset: 0, index: 0)
        x.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(threads, 256), height: 1, depth: 1)); if e == nil { x.endEncoding() } }
    e?.endEncoding(); cb.commit(); cb.waitUntilCompleted(); return (cb.gpuEndTime - cb.gpuStartTime) / Double(n) * 1e6 }
_ = run(200, one: false, threads: 1)
for t in [1, 2304] { for _ in 0..<3 { print(String(format: "threads %4d: own encoder %.2f us a step; one serial encoder %.2f us a step", t, run(2000, one: false, threads: t), run(2000, one: true, threads: t))) } }
