import Foundation

// The single-threaded scalar cv2.resize port as it was before the SIMD / multi-core one in Resize.swift (23 Oct 2026),
// for A/B tests on the phone (JobOptions.multicorePixels = false). Same bytes out; INTER_AREA for non-integer
// downscales is shared (Resize.swift `resizeArea`, which didn't change).

/// The original single-threaded scalar port of cv2.resize (same output as `cvResize`, byte for byte), kept so
/// `PixelWork.multicore = false` can compare against it.
func cvResizeScalar(_ src: PixelView, _ dst: PixelView, _ interp: Interp) {
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
        if areaFast { resizeAreaFastScalar(src, dst, iscaleX, iscaleY) } else { resizeArea(src, dst, scaleX, scaleY) }
        return
    }
    resizeLinearScalar(src, dst, scaleX: scaleX, scaleY: scaleY, invX: invX, invY: invY, areaMode: mode == .area)
}

private func resizeLinearScalar(_ src: PixelView, _ dst: PixelView, scaleX: Double, scaleY: Double,
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
    let width3 = dw * 3
    let simdEnd = (width3 / 16) * 16
    let rowA = UnsafeMutablePointer<Int32>.allocate(capacity: width3), rowB = UnsafeMutablePointer<Int32>.allocate(capacity: width3)
    defer { rowA.deallocate(); rowB.deallocate() }
    var bufA = rowA, bufB = rowB
    var cachedA = -1, cachedB = -1

    xofs.withUnsafeBufferPointer { xo in
        ialpha.withUnsafeBufferPointer { al in
            func hresize(_ sy: Int, into buf: UnsafeMutablePointer<Int32>) {
                let s = src.row(sy)
                for dx in 0..<dw {
                    let sx = xo[dx], a0 = al[2 * dx], a1 = al[2 * dx + 1]
                    let p0 = s + 4 * sx, p1 = s + 4 * min(sx + 1, sw - 1)
                    let o = buf + 3 * dx
                    if a1 == 0 {
                        o[0] = Int32(p0[0]) * a0; o[1] = Int32(p0[1]) * a0; o[2] = Int32(p0[2]) * a0
                    } else {
                        o[0] = Int32(p0[0]) * a0 + Int32(p1[0]) * a1
                        o[1] = Int32(p0[1]) * a0 + Int32(p1[1]) * a1
                        o[2] = Int32(p0[2]) * a0 + Int32(p1[2]) * a1
                    }
                }
            }
            for dy in 0..<dh {
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
                let d = dst.row(dy)
                var e = 0
                for x in 0..<dw {
                    let o = d + 4 * x
                    for c in 0..<3 {
                        let v0 = bufA[e], v1 = bufB[e]
                        let out: Int32
                        if e < simdEnd {
                            // v_rshr_pack_u<2>(v_mul_hi(v0 >> 4, b0) + v_mul_hi(v1 >> 4, b1))
                            out = ((((v0 >> 4) * b0) >> 16) + (((v1 >> 4) * b1) >> 16) + 2) >> 2
                        } else {
                            out = (v0 * b0 + v1 * b1 + (1 << 21)) >> 22
                        }
                        o[c] = UInt8(clamping: out)
                        e += 1
                    }
                    o[3] = 255
                }
            }
        }
    }
}

/// INTER_AREA with integer scale factors (block average).
private func resizeAreaFastScalar(_ src: PixelView, _ dst: PixelView, _ kx: Int, _ ky: Int) {
    let area = kx * ky
    let inv = 1 / Float(area)
    for dy in 0..<dst.height {
        let d = dst.row(dy)
        for dx in 0..<dst.width {
            for c in 0..<3 {
                var sum = 0
                for j in 0..<ky {
                    let s = src.row(dy * ky + j)
                    for i in 0..<kx { sum += Int(s[4 * (dx * kx + i) + c]) }
                }
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
