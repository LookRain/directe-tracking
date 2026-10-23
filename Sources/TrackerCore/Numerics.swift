import Accelerate
import Foundation

// Bit-for-bit numpy, where it matters. The lab tracker mixes float32 (candidate boxes, colour
// features, re-ID embeddings) and float64 (its own state), and numpy (>= 2, NEP 50) decides per
// expression which precision an operation runs in: float32 op float32 stays float32, a Python
// float next to a float32 array is rounded to float32 first, a numpy float64 scalar promotes to
// float64. A comparison near a threshold can flip on the last bit, so the port mirrors that:
// every value is stored as Double, and an operation numpy runs in float32 is rounded with `f32`.
//
// For +, -, *, / and sqrt of float32 inputs, rounding the exact-ish double result to float32 gives
// the correctly rounded float32 result (53 >= 2 * 24 + 2: no double-rounding error), so
// `f32(a * b)` is numpy's float32 `a * b`. Dot products go through Accelerate's BLAS with the
// exact calls numpy's matmul makes (numpy on macOS links Accelerate), and reductions use numpy's
// summation order (pairwise along a contiguous axis, sequential along an outer one).

/// numpy float32 rounding of a value computed in Double.
@inline(__always) func f32(_ x: Double) -> Double { Double(Float(x)) }

/// A Python float next to a float32 array: rounded to float32 before the operation.
@inline(__always) func weak32(_ x: Double) -> Double { Double(Float(x)) }

/// Values of a numpy array that is either float32 or float64 (`v` holds the float32 values exactly).
struct Scores {
    var v: [Double]
    var f32: Bool

    /// `self[i] >= thr` with `thr` a Python float (weakly typed: rounded to the array's precision).
    @inline(__always) func ge(_ i: Int, _ thr: Double) -> Bool { v[i] >= (f32 ? weak32(thr) : thr) }
    @inline(__always) func lt(_ i: Int, _ thr: Double) -> Bool { v[i] < (f32 ? weak32(thr) : thr) }
}

/// `value > thr` for a numpy value of the given precision against a Python float.
@inline(__always) func gt(_ value: Double, _ thr: Double, f32 isF32: Bool) -> Bool { value > (isF32 ? weak32(thr) : thr) }

@inline(__always) func argmax(_ a: [Double]) -> Int {
    var bi = 0
    for i in 1..<max(a.count, 1) where a[i] > a[bi] { bi = i }
    return bi
}

@inline(__always) func argmin(_ a: [Double]) -> Int {
    var bi = 0
    for i in 1..<max(a.count, 1) where a[i] < a[bi] { bi = i }
    return bi
}

/// np.median of Python floats.
func median(_ a: [Double]) -> Double {
    let s = a.sorted(), n = s.count
    return n % 2 == 1 ? s[n / 2] : (0 + s[n / 2 - 1] + s[n / 2]) / 2
}

// MARK: numpy float32 reductions

/// `np.add.reduce` of float32 along a contiguous axis: numpy's pairwise sum (8 accumulators in
/// blocks of up to 128), added to the identity 0.
func npSum(_ a: UnsafeBufferPointer<Float>) -> Float {
    guard let p = a.baseAddress, a.count > 0 else { return 0 }
    return 0 + pairwise(p, a.count)
}

private func pairwise(_ a: UnsafePointer<Float>, _ n: Int) -> Float {
    if n < 8 {
        var res: Float = 0
        for i in 0..<n { res += a[i] }
        return res
    } else if n <= 128 {
        var r0 = a[0], r1 = a[1], r2 = a[2], r3 = a[3], r4 = a[4], r5 = a[5], r6 = a[6], r7 = a[7]
        var i = 8
        while i < n - (n % 8) {
            r0 += a[i]; r1 += a[i + 1]; r2 += a[i + 2]; r3 += a[i + 3]
            r4 += a[i + 4]; r5 += a[i + 5]; r6 += a[i + 6]; r7 += a[i + 7]
            i += 8
        }
        var res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n { res += a[i]; i += 1 }
        return res
    } else {
        var n2 = n / 2
        n2 -= n2 % 8
        return pairwise(a, n2) + pairwise(a + n2, n - n2)
    }
}

// MARK: BLAS, exactly as numpy's matmul / dot / linalg.norm call it

// The classic CBLAS entry points are deprecated in favour of the ILP64 interface (which Swift can't
// select without a C define); they run the same kernels, and calling them through a protocol keeps
// the build free of deprecation warnings.
protocol BLASCalls: Sendable {
    func sdot(_ n: Int, _ x: UnsafePointer<Float>, _ y: UnsafePointer<Float>) -> Float
    func ddot(_ n: Int, _ x: UnsafePointer<Double>, _ y: UnsafePointer<Double>) -> Double
    func sgemvT(_ e: Int, _ rows: Int, _ a: UnsafePointer<Float>, _ x: UnsafePointer<Float>, _ y: UnsafeMutablePointer<Float>)
    func sgemmNT(_ m: Int, _ n: Int, _ k: Int, _ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ c: UnsafeMutablePointer<Float>)
}

struct AccelerateBLAS: BLASCalls {
    @available(macOS, deprecated: 13.3) @available(iOS, deprecated: 16.4)
    func sdot(_ n: Int, _ x: UnsafePointer<Float>, _ y: UnsafePointer<Float>) -> Float {
        cblas_sdot(Int32(n), x, 1, y, 1)
    }

    @available(macOS, deprecated: 13.3) @available(iOS, deprecated: 16.4)
    func ddot(_ n: Int, _ x: UnsafePointer<Double>, _ y: UnsafePointer<Double>) -> Double {
        cblas_ddot(Int32(n), x, 1, y, 1)
    }

    /// numpy `_gemv` for a C-contiguous [rows, e] matrix: column-major, transposed.
    @available(macOS, deprecated: 13.3) @available(iOS, deprecated: 16.4)
    func sgemvT(_ e: Int, _ rows: Int, _ a: UnsafePointer<Float>, _ x: UnsafePointer<Float>, _ y: UnsafeMutablePointer<Float>) {
        cblas_sgemv(CblasColMajor, CblasTrans, Int32(e), Int32(rows), 1, a, Int32(e), x, 1, 0, y, 1)
    }

    /// numpy `_matmul_matrixmatrix` for A [m, k] @ B[n, k].T: row-major, B transposed.
    @available(macOS, deprecated: 13.3) @available(iOS, deprecated: 16.4)
    func sgemmNT(_ m: Int, _ n: Int, _ k: Int, _ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ c: UnsafeMutablePointer<Float>) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(m), Int32(n), Int32(k), 1, a, Int32(k), b, Int32(k), 0, c, Int32(n))
    }
}

let blas: any BLASCalls = AccelerateBLAS()

/// numpy `x @ y` for two float32 vectors (FLOAT_dot: sdot accumulated into a double, then float).
func dot(_ x: ArraySlice<Float>, _ y: [Float]) -> Float {
    precondition(x.count == y.count, "embedding sizes differ")
    return x.withUnsafeBufferPointer { xp in y.withUnsafeBufferPointer { yp in
        Float(0.0 + Double(blas.sdot(x.count, xp.baseAddress!, yp.baseAddress!)))
    } }
}

/// numpy `A @ x`, A a C-contiguous [rows, e] float32 matrix (flat), x [e]: a dot product for one row,
/// gemv otherwise.
func matvec(_ a: [Float], rows: Int, _ x: [Float]) -> [Float] {
    if rows == 0 { return [] }
    let e = x.count
    precondition(a.count == rows * e, "embedding sizes differ")
    if rows == 1 { return [dot(a[...], x)] }
    var y = [Float](repeating: 0, count: rows)
    a.withUnsafeBufferPointer { ap in x.withUnsafeBufferPointer { xp in y.withUnsafeMutableBufferPointer { yp in
        blas.sgemvT(e, rows, ap.baseAddress!, xp.baseAddress!, yp.baseAddress!)
    } } }
    return y
}

/// numpy `A @ B.T` for C-contiguous float32 A [m, e] and B [n, e] (flat), as numpy's matmul
/// dispatches it: dot for 1x1, gemv when either side is a single row, gemm otherwise. [m, n] flat.
func matmulT(_ a: [Float], m: Int, _ b: [Float], n: Int, e: Int) -> [Float] {
    if m == 0 || n == 0 { return [Float](repeating: 0, count: m * n) }
    precondition(a.count == m * e && b.count == n * e, "embedding sizes differ")
    if m == 1 && n == 1 { return [dot(a[...], b)] }
    if m == 1 { return matvec(b, rows: n, a) }
    if n == 1 { return matvec(a, rows: m, b) }
    var c = [Float](repeating: 0, count: m * n)
    a.withUnsafeBufferPointer { ap in b.withUnsafeBufferPointer { bp in c.withUnsafeMutableBufferPointer { cp in
        blas.sgemmNT(m, n, e, ap.baseAddress!, bp.baseAddress!, cp.baseAddress!)
    } } }
    return c
}

/// np.linalg.norm of a float64 2-vector (sqrt of ddot).
func norm(_ v: SIMD2<Double>) -> Double {
    let a = [v.x, v.y]
    return a.withUnsafeBufferPointer { (0.0 + blas.ddot(2, $0.baseAddress!, $0.baseAddress!)).squareRoot() }
}

/// np.linalg.norm of a float32 vector (sqrt of sdot, in float32).
func norm32(_ a: [Float]) -> Float {
    a.withUnsafeBufferPointer { Float(0.0 + Double(blas.sdot(a.count, $0.baseAddress!, $0.baseAddress!))).squareRoot() }
}

// MARK: geometry in numpy's precision

/// A box's own area as numpy computes it: in float32 for a float32 box.
@inline(__always) func npArea(_ b: Box, f32 isF32: Bool) -> Double {
    isF32 ? f32(f32(b.x1 - b.x0) * f32(b.y1 - b.y0)) : (b.x1 - b.x0) * (b.y1 - b.y0)
}

/// lab `iou(a, b)` in numpy's precision: float32 when both are float32 arrays, otherwise float64
/// (a float32 box's own area still in float32). Returns the values and whether they are float32.
func npIoU(_ a: Box, _ aF32: Bool, _ bs: [Box], _ bF32: Bool) -> Scores {
    let p = aF32 && bF32
    @inline(__always) func r(_ x: Double) -> Double { p ? f32(x) : x }
    let areaA = npArea(a, f32: aF32)
    let eps = p ? weak32(1e-6) : 1e-6
    var out = [Double]()
    out.reserveCapacity(bs.count)
    for b in bs {
        let iw = max(r(min(a.x1, b.x1) - max(a.x0, b.x0)), 0)
        let ih = max(r(min(a.y1, b.y1) - max(a.y0, b.y0)), 0)
        let inter = r(iw * ih)
        out.append(r(inter / max(r(r(areaA + npArea(b, f32: bF32)) - inter), eps)))
    }
    return Scores(v: out, f32: p)
}

/// lab `same_person(a, b, thr)` for float32 boxes (float32 throughout).
func npSamePerson32(_ a: Box, _ bs: [Box], _ thr: Double) -> [Bool] {
    let ious = npIoU(a, true, bs, true)
    let eps = weak32(1e-6)
    let areaA = max(npArea(a, f32: true), eps), ha = f32(a.y1 - a.y0)
    return bs.indices.map { i in
        let b = bs[i]
        let iw = max(f32(min(a.x1, b.x1) - max(a.x0, b.x0)), 0)
        let ih = max(f32(min(a.y1, b.y1) - max(a.y0, b.y0)), 0)
        let inter = f32(iw * ih)
        let contained = f32(inter / min(areaA, max(npArea(b, f32: true), eps)))
        let hb = f32(b.y1 - b.y0)
        let similar = min(ha, hb) >= f32(0.25 * max(ha, hb))
        return ious.v[i] >= weak32(thr) || (contained >= weak32(0.7) && similar)
    }
}

extension Box {
    /// The box with float32 coordinates (how the lab stores candidate boxes).
    var float32: Box { Box(f32(x0), f32(y0), f32(x1), f32(y1)) }
}
