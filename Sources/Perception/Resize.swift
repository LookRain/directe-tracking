import Foundation

// cv2.resize for 8-bit 3-channel images, ported from OpenCV 4.x imgproc/src/resize.cpp so the network
// inputs match the lab's: INTER_LINEAR (fixed point, 11-bit coefficients; exact 2x downscales go
// through INTER_AREA as in OpenCV) and INTER_AREA (pixel-area averaging for downscales).
// The lab's images are BGR (3 channels); OpenCV's vertical pass rounds the first 16*floor(3w/16)
// interleaved values with its SIMD formula and the rest with the scalar one, reproduced here.

enum Interp { case linear, area }

/// Runs `body(lo, hi)` over row bands covering [0, rows): on several cores when the output is big enough
/// (`pixels`) to pay for it, else in one go. Every output row is computed from the source alone, so the
/// result is the same either way. Eight bands, so cores of different speeds (the phone's 2 performance +
/// 4 efficiency cores) share the work.
@inline(__always) func forBands(_ rows: Int, pixels: Int, _ body: (Int, Int) -> Void) {
    let bands = pixels >= 1 << 16 ? min(8, rows) : 1
    if bands <= 1 { body(0, rows); return }
    DispatchQueue.concurrentPerform(iterations: bands) { b in body(rows * b / bands, rows * (b + 1) / bands) }
}

/// How the pixel work runs, process-wide (one job at a time; set from JobOptions.multicorePixels at the job's start):
/// `multicore` = the SIMD resizes in row bands on several cores, and each person's masks, colour features and re-ID
/// crop on their own core; off = the original single-threaded code. Same results either way.
public enum PixelWork {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var _multicore = true
    public static var multicore: Bool {
        get { lock.withLock { _multicore } }
        set { lock.withLock { _multicore = newValue } }
    }
}

/// cv2.resize(src, (dst.width, dst.height), interpolation=interp): writes B, G, R (and alpha 255) into dst.
func cvResize(_ src: PixelView, _ dst: PixelView, _ interp: Interp) {
    guard PixelWork.multicore else { return cvResizeScalar(src, dst, interp) }
    let sw = src.width, sh = src.height, dw = dst.width, dh = dst.height
    if sw == dw && sh == dh {
        for y in 0..<sh {
            let s = src.row(y), d = dst.row(y)
            for x in 0..<sw { d[4 * x] = s[4 * x]; d[4 * x + 1] = s[4 * x + 1]; d[4 * x + 2] = s[4 * x + 2]; d[4 * x + 3] = 255 }
        }
        return
    }
    let invX = Double(dw) / Double(sw), invY = Double(dh) / Double(sh)
    let scaleX = 1 / invX, scaleY = 1 / invY
    let iscaleX = Int(scaleX.rounded(.toNearestOrEven)), iscaleY = Int(scaleY.rounded(.toNearestOrEven))
    let areaFast = abs(scaleX - Double(iscaleX)) < Double.ulpOfOne && abs(scaleY - Double(iscaleY)) < Double.ulpOfOne
    var mode = interp
    if mode == .linear && areaFast && iscaleX == 2 && iscaleY == 2 { mode = .area }
    if mode == .area && scaleX >= 1 && scaleY >= 1 {
        if areaFast { resizeAreaFast(src, dst, iscaleX, iscaleY) } else { resizeArea(src, dst, scaleX, scaleY) }
        return
    }
    resizeLinear(src, dst, scaleX: scaleX, scaleY: scaleY, invX: invX, invY: invY, areaMode: mode == .area)
}

private func resizeLinear(_ src: PixelView, _ dst: PixelView, scaleX: Double, scaleY: Double,
                          invX: Double, invY: Double, areaMode: Bool) {
    let sw = src.width, sh = src.height, dw = dst.width, dh = dst.height
    // horizontal taps
    var xofs = [Int](repeating: 0, count: dw)
    var ialpha = [Int32](repeating: 0, count: dw * 2)
    for dx in 0..<dw {
        var fx: Float, sx: Int
        if !areaMode {
            fx = Float((Double(dx) + 0.5) * scaleX - 0.5)
            sx = Int(fx.rounded(.down))
            fx -= Float(sx)
        } else {
            sx = Int((Double(dx) * scaleX).rounded(.down))
            fx = Float(Double(dx + 1) - Double(sx + 1) * invX)
            fx = fx <= 0 ? 0 : fx - fx.rounded(.down)
        }
        if sx < 0 { fx = 0; sx = 0 }
        if sx >= sw - 1 { fx = 0; sx = sw - 1 }
        xofs[dx] = sx
        ialpha[2 * dx] = Int32(((1 - fx) * 2048).rounded(.toNearestOrEven))
        ialpha[2 * dx + 1] = Int32((fx * 2048).rounded(.toNearestOrEven))
    }
    // OpenCV's vertical pass rounds the first 16*floor(3w/16) interleaved B G R values with its SIMD formula and the
    // rest with the scalar one: pixels below `simdPixels` are all SIMD, the few after it are done value by value.
    let width3 = dw * 3
    let simdEnd = (width3 / 16) * 16
    let simdPixels = min(simdEnd / 3, dw)
    // byte offsets of each output pixel's two source pixels
    var off0 = [Int](repeating: 0, count: dw), off1 = off0
    for dx in 0..<dw { off0[dx] = 4 * xofs[dx]; off1[dx] = 4 * min(xofs[dx] + 1, sw - 1) }

    off0.withUnsafeBufferPointer { o0 in
        off1.withUnsafeBufferPointer { o1 in
            ialpha.withUnsafeBufferPointer { al in
                forBands(dh, pixels: dw * dh) { lo, hi in
                    // each band keeps its own two horizontally resized source rows (B G R x, the 4th lane unused)
                    let rowA = UnsafeMutablePointer<SIMD4<Int32>>.allocate(capacity: dw)
                    let rowB = UnsafeMutablePointer<SIMD4<Int32>>.allocate(capacity: dw)
                    defer { rowA.deallocate(); rowB.deallocate() }
                    var bufA = rowA, bufB = rowB
                    var cachedA = -1, cachedB = -1
                    func hresize(_ sy: Int, into buf: UnsafeMutablePointer<SIMD4<Int32>>) {
                        let s = UnsafeRawPointer(src.row(sy))
                        for dx in 0..<dw {
                            // a1 == 0 at the edges: the second tap adds nothing, as in OpenCV
                            let p0 = SIMD4<Int32>(truncatingIfNeeded: s.loadUnaligned(fromByteOffset: o0[dx], as: SIMD4<UInt8>.self))
                            let p1 = SIMD4<Int32>(truncatingIfNeeded: s.loadUnaligned(fromByteOffset: o1[dx], as: SIMD4<UInt8>.self))
                            let t0: SIMD4<Int32> = p0 &* al[2 * dx], t1: SIMD4<Int32> = p1 &* al[2 * dx + 1]
                            buf[dx] = t0 &+ t1
                        }
                    }
                    let lo255 = SIMD4<Int32>(repeating: 0), hi255 = SIMD4<Int32>(repeating: 255)
                    for dy in lo..<hi {
                        var fy: Float, sy: Int
                        if !areaMode {
                            fy = Float((Double(dy) + 0.5) * scaleY - 0.5)
                            sy = Int(fy.rounded(.down))
                            fy -= Float(sy)
                        } else {
                            sy = Int((Double(dy) * scaleY).rounded(.down))
                            fy = Float(Double(dy + 1) - Double(sy + 1) * invY)
                            fy = fy <= 0 ? 0 : fy - fy.rounded(.down)
                        }
                        let b0 = Int32(((1 - fy) * 2048).rounded(.toNearestOrEven)), b1 = Int32((fy * 2048).rounded(.toNearestOrEven))
                        let r0 = min(max(sy, 0), sh - 1), r1 = min(max(sy + 1, 0), sh - 1)
                        if cachedA != r0 {
                            if cachedB == r0 { swap(&bufA, &bufB); swap(&cachedA, &cachedB) } else { hresize(r0, into: bufA); cachedA = r0 }
                        }
                        if cachedB != r1 { hresize(r1, into: bufB); cachedB = r1 }
                        let d = UnsafeMutableRawPointer(dst.row(dy))
                        for x in 0..<simdPixels {
                            // v_rshr_pack_u<2>(v_mul_hi(v0 >> 4, b0) + v_mul_hi(v1 >> 4, b1))
                            let va: SIMD4<Int32> = ((bufA[x] &>> 4) &* b0) &>> 16
                            let vb: SIMD4<Int32> = ((bufB[x] &>> 4) &* b1) &>> 16
                            let v: SIMD4<Int32> = (va &+ vb &+ 2) &>> 2
                            var o = SIMD4<UInt8>(truncatingIfNeeded: v.clamped(lowerBound: lo255, upperBound: hi255))
                            o[3] = 255
                            d.storeBytes(of: o, toByteOffset: 4 * x, as: SIMD4<UInt8>.self)
                        }
                        let o = d.assumingMemoryBound(to: UInt8.self)
                        for x in simdPixels..<dw {
                            for c in 0..<3 {
                                let v0 = bufA[x][c], v1 = bufB[x][c]
                                let out: Int32
                                if 3 * x + c < simdEnd {
                                    out = ((((v0 >> 4) * b0) >> 16) + (((v1 >> 4) * b1) >> 16) + 2) >> 2
                                } else {
                                    out = (v0 * b0 + v1 * b1 + (1 << 21)) >> 22
                                }
                                o[4 * x + c] = UInt8(truncatingIfNeeded: min(max(out, 0), 255))
                            }
                            o[4 * x + 3] = 255
                        }
                    }
                }
            }
        }
    }
}

/// INTER_AREA with integer scale factors (block average): the same integer block sums, gathered a source row at a
/// time (sequential reads rather than a block at a time per channel), in row bands on several cores.
private func resizeAreaFast(_ src: PixelView, _ dst: PixelView, _ kx: Int, _ ky: Int) {
    let area = kx * ky
    let inv = 1 / Float(area)
    let dw = dst.width
    forBands(dst.height, pixels: dw * dst.height) { lo, hi in
        let sums = UnsafeMutablePointer<Int>.allocate(capacity: 3 * dw)
        defer { sums.deallocate() }
        for dy in lo..<hi {
            sums.initialize(repeating: 0, count: 3 * dw)
            for j in 0..<ky {
                let s = src.row(dy * ky + j)
                for dx in 0..<dw {
                    let p = s + 4 * dx * kx
                    var b = 0, g = 0, r = 0
                    for i in 0..<kx { b += Int(p[4 * i]); g += Int(p[4 * i + 1]); r += Int(p[4 * i + 2]) }
                    sums[3 * dx] += b; sums[3 * dx + 1] += g; sums[3 * dx + 2] += r
                }
            }
            let d = dst.row(dy)
            for dx in 0..<dw {
                for c in 0..<3 {
                    let sum = sums[3 * dx + c]
                    if kx == 2 && ky == 2 {
                        d[4 * dx + c] = UInt8((sum + 2) >> 2)
                    } else {
                        d[4 * dx + c] = UInt8(clamping: Int((Float(sum) * inv).rounded(.toNearestOrEven)))
                    }
                }
                d[4 * dx + 3] = 255
            }
        }
    }
}

private struct AreaTap { var di: Int; var si: Int; var alpha: Float }

private func areaTab(_ ssize: Int, _ dsize: Int, _ scale: Double) -> [AreaTap] {
    var tab: [AreaTap] = []
    for dx in 0..<dsize {
        let fsx1 = Double(dx) * scale, fsx2 = fsx1 + scale
        let cellWidth = min(scale, Double(ssize) - fsx1)
        var sx1 = Int(fsx1.rounded(.up)), sx2 = Int(fsx2.rounded(.down))
        sx2 = min(sx2, ssize - 1)
        sx1 = min(sx1, sx2)
        if Double(sx1) - fsx1 > 1e-3 { tab.append(AreaTap(di: dx, si: sx1 - 1, alpha: Float((Double(sx1) - fsx1) / cellWidth))) }
        if sx1 < sx2 { for sx in sx1..<sx2 { tab.append(AreaTap(di: dx, si: sx, alpha: Float(1.0 / cellWidth))) } }
        if fsx2 - Double(sx2) > 1e-3 {
            tab.append(AreaTap(di: dx, si: sx2, alpha: Float(min(min(fsx2 - Double(sx2), 1), cellWidth) / cellWidth)))
        }
    }
    return tab
}

/// INTER_AREA for non-integer downscales (ResizeArea_Invoker, float accumulation in OpenCV's order).
func resizeArea(_ src: PixelView, _ dst: PixelView, _ scaleX: Double, _ scaleY: Double) {
    let xtab = areaTab(src.width, dst.width, scaleX), ytab = areaTab(src.height, dst.height, scaleY)
    let w3 = dst.width * 3
    let buf = UnsafeMutablePointer<Float>.allocate(capacity: w3), sum = UnsafeMutablePointer<Float>.allocate(capacity: w3)
    defer { buf.deallocate(); sum.deallocate() }
    sum.initialize(repeating: 0, count: w3)
    var prevDy = ytab.first?.di ?? 0

    func emit(_ dy: Int) {
        let d = dst.row(dy)
        for x in 0..<dst.width {
            for c in 0..<3 { d[4 * x + c] = UInt8(clamping: Int(sum[3 * x + c].rounded(.toNearestOrEven))) }
            d[4 * x + 3] = 255
        }
    }

    xtab.withUnsafeBufferPointer { xt in
        for t in ytab {
            let s = src.row(t.si)
            buf.initialize(repeating: 0, count: w3)
            for k in xt {
                let sp = s + 4 * k.si, dp = buf + 3 * k.di
                dp[0] += Float(sp[0]) * k.alpha
                dp[1] += Float(sp[1]) * k.alpha
                dp[2] += Float(sp[2]) * k.alpha
            }
            if t.di != prevDy {
                emit(prevDy)
                for e in 0..<w3 { sum[e] = t.alpha * buf[e] }
                prevDy = t.di
            } else {
                for e in 0..<w3 { sum[e] += t.alpha * buf[e] }
            }
        }
    }
    emit(prevDy)
}

/// cv2.cvtColor(BGR2GRAY) for 8-bit: (1868 B + 9617 G + 4899 R + 2^13) >> 14.
@inline(__always) func cvGrey(_ b: UInt8, _ g: UInt8, _ r: UInt8) -> UInt8 {
    let y: Int = Int(b) * 1868 + Int(g) * 9617 + Int(r) * 4899
    return UInt8((y + 8192) >> 14)
}
