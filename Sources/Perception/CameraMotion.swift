import Accelerate
import CoreVideo
import Foundation

/// Whole-image shift between consecutive ticks: port of directe_tracking/online.py CameraMotion, i.e.
/// cv2.phaseCorrelate(prev, cur, hanning) on a 360-px-wide grey copy (cv2 INTER_LINEAR resize,
/// BGR2GRAY), scaled back to source pixels. Returns (shift, response); the first frame gives
/// (0, 1). The tracker treats response < 0.3 as "camera motion unknown". The DFT runs as matrix
/// products (Accelerate) at OpenCV's optimal DFT size (zero padded), like cv2.dft.
public final class CameraMotion {
    public static let width = 360

    private var prev: (re: [Float], im: [Float])?
    private var shape = SIMD2<Int>(0, 0)       // grey rows, cols
    private var window: [Float] = []           // padded M x N, Hanning (sqrt, as cv2.createHanningWindow)
    private var dft: DFT2D?

    public init() {}

    public func reset() { prev = nil }

    /// (dx, dy) in source pixels: how far the image content moved since the previous call, and the
    /// phase-correlation response (peak energy, ~1 for a clean shift). Frames must be 32BGRA.
    public func shift(_ frame: CVPixelBuffer) -> (SIMD2<Double>, Double) {
        (try? withPixels(frame) { shift($0) }) ?? (SIMD2(0, 0), 0)
    }

    func shift(_ view: PixelView) -> (SIMD2<Double>, Double) {
        let s = Double(view.width) / Double(Self.width)
        let rows = pyRound(Double(view.height) / s), cols = Self.width
        let small = BGRAImage(width: cols, height: rows)
        cvResize(view, small.view, .linear)
        if shape != SIMD2(rows, cols) {
            shape = SIMD2(rows, cols)
            prev = nil
            let M = optimalDFTSize(rows), N = optimalDFTSize(cols)
            dft = DFT2D(M: M, N: N)
            window = [Float](repeating: 0, count: M * N)
            let c0 = 2 * Double.pi / Double(cols - 1), c1 = 2 * Double.pi / Double(rows - 1)
            for i in 0..<rows {
                let wr = 0.5 * (1 - cos(c1 * Double(i)))
                for j in 0..<cols {
                    let wc = 0.5 * (1 - cos(c0 * Double(j)))
                    window[i * N + j] = Float(wr * wc).squareRoot()
                }
            }
        }
        let d = dft!
        let N = d.N
        var x = [Float](repeating: 0, count: d.M * N)
        let v = small.view
        x.withUnsafeMutableBufferPointer { xp in
            window.withUnsafeBufferPointer { wp in
                for i in 0..<rows {
                    let r = v.row(i)
                    for j in 0..<cols {
                        xp[i * N + j] = Float(cvGrey(r[4 * j], r[4 * j + 1], r[4 * j + 2])) * wp[i * N + j]
                    }
                }
            }
        }
        let cur = d.forward(x)
        defer { prev = cur }
        guard let p = prev else { return (SIMD2(0, 0), 1) }
        // P = F1 * conj(F2), C = P / |P|
        let n = d.M * d.N, nh = d.M * d.H
        var cr = [Float](repeating: 0, count: nh), ci = [Float](repeating: 0, count: nh)
        cr.withUnsafeMutableBufferPointer { crp in
            ci.withUnsafeMutableBufferPointer { cip in
                p.re.withUnsafeBufferPointer { pr in
                    p.im.withUnsafeBufferPointer { pi in
                        cur.re.withUnsafeBufferPointer { qr in
                            cur.im.withUnsafeBufferPointer { qi in
                                for k in 0..<nh {
                                    let a = pr[k], b = pi[k], c = qr[k], e = qi[k]
                                    let re = a * c + b * e, im = b * c - a * e
                                    let m = (re * re + im * im).squareRoot()
                                    let den = m * m + Float.ulpOfOne
                                    crp[k] = re * m / den
                                    cip[k] = im * m / den
                                }
                            }
                        }
                    }
                }
            }
        }
        var corr = d.inverseReal(cr, ci)
        fftShift(&corr, d.M, d.N)
        var peak = 0
        corr.withUnsafeBufferPointer { c in
            var best = c[0]
            for k in 1..<n where c[k] > best { peak = k; best = c[k] }
        }
        let py = peak / d.N, px = peak % d.N
        var sx = 0.0, sy = 0.0, sum = 0.0
        for yy in max(py - 2, 0)...min(py + 2, d.M - 1) {
            for xx in max(px - 2, 0)...min(px + 2, d.N - 1) {
                let w = Double(corr[yy * d.N + xx])
                sx += Double(xx) * w; sy += Double(yy) * w; sum += w
            }
        }
        let response = sum / Double(n)
        sum += Double.ulpOfOne
        let t = SIMD2(sx / sum, sy / sum)
        let shift = SIMD2(Double(d.N) / 2, Double(d.M) / 2) - t
        return (shift * s, response)
    }
}

/// cv::getOptimalDFTSize: the smallest 2^a 3^b 5^c >= n.
func optimalDFTSize(_ n: Int) -> Int {
    var m = max(n, 1)
    while true {
        var k = m
        for f in [2, 3, 5] { while k % f == 0 { k /= f } }
        if k == 1 { return m }
        m += 1
    }
}

/// OpenCV's fftShift in phasecorr.cpp: swaps quadrants (q0 <-> q3, q1 <-> q2) of size cols/2 x rows/2.
func fftShift(_ a: inout [Float], _ rows: Int, _ cols: Int) {
    let xm = cols >> 1, ym = rows >> 1
    if xm == 0 || ym == 0 { return }
    a.withUnsafeMutableBufferPointer { a in
        for y in 0..<ym {
            for x in 0..<xm {
                a.swapAt(y * cols + x, (y + ym) * cols + x + xm)
                a.swapAt(y * cols + x + xm, (y + ym) * cols + x)
            }
        }
    }
}

/// 2-D DFT of an M x N real image and the unscaled inverse back to a real image, as dense matrix
/// products (BLAS) over the non-redundant half spectrum (columns 0...N/2; the rest is the
/// conjugate mirror). Arbitrary sizes, so OpenCV's optimal DFT sizes need no power of two.
final class DFT2D {
    let M: Int, N: Int, H: Int
    private let cm: [Float], sm: [Float]      // M x M
    private let cn: [Float], sn: [Float]      // N x H (forward, columns 0..<H)
    private let icn: [Float], isn: [Float]    // H x N (inverse rows, with the half-spectrum weights)

    init(M: Int, N: Int) {
        self.M = M; self.N = N; H = N / 2 + 1
        func angle(_ i: Int, _ j: Int, _ n: Int) -> Double { 2 * Double.pi * Double((i * j) % n) / Double(n) }
        var c = [Float](repeating: 0, count: M * M), s = c
        for i in 0..<M { for j in 0..<M { c[i * M + j] = Float(cos(angle(i, j, M))); s[i * M + j] = Float(sin(angle(i, j, M))) } }
        cm = c; sm = s
        let h = N / 2 + 1
        var fc = [Float](repeating: 0, count: N * h), fs = fc, ic = [Float](repeating: 0, count: h * N), isv = ic
        for n in 0..<N {
            for k in 0..<h {
                fc[n * h + k] = Float(cos(angle(n, k, N))); fs[n * h + k] = Float(sin(angle(n, k, N)))
                // x[n] = sum_k w_k Re(C[k] e^{+i a}) with w = 1 for k = 0 (and N/2 when N is even), else 2
                let w: Double = (k == 0 || (N % 2 == 0 && k == N / 2)) ? 1 : 2
                ic[k * N + n] = Float(w * cos(angle(n, k, N))); isv[k * N + n] = Float(w * sin(angle(n, k, N)))
            }
        }
        cn = fc; sn = fs; icn = ic; isn = isv
    }

    /// C (m x n) = alpha A (m x k) B (k x n) + beta C
    private func gemm(_ a: [Float], _ b: [Float], _ c: inout [Float], _ m: Int, _ n: Int, _ k: Int,
                      alpha: Float = 1, beta: Float = 0) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(m), Int32(n), Int32(k),
                    alpha, a, Int32(k), b, Int32(n), beta, &c, Int32(n))
    }

    /// Half spectrum (M x H) of Z = F_M X F_N, F = exp(-2 pi i jk / n).
    func forward(_ x: [Float]) -> (re: [Float], im: [Float]) {
        var yr = [Float](repeating: 0, count: M * H), yi = yr
        gemm(x, cn, &yr, M, H, N)
        gemm(x, sn, &yi, M, H, N, alpha: -1)
        var zr = [Float](repeating: 0, count: M * H), zi = zr
        gemm(cm, yr, &zr, M, H, M); gemm(sm, yi, &zr, M, H, M, beta: 1)
        gemm(cm, yi, &zi, M, H, M); gemm(sm, yr, &zi, M, H, M, alpha: -1, beta: 1)
        return (zr, zi)
    }

    /// Real M x N image whose (Hermitian) spectrum has this half (M x H): conj(F_M) C conj(F_N),
    /// no 1/(MN) scaling (cv2.idft without DFT_SCALE).
    func inverseReal(_ cr: [Float], _ ci: [Float]) -> [Float] {
        var ur = [Float](repeating: 0, count: M * H), ui = ur
        gemm(cm, cr, &ur, M, H, M); gemm(sm, ci, &ur, M, H, M, alpha: -1, beta: 1)
        gemm(cm, ci, &ui, M, H, M); gemm(sm, cr, &ui, M, H, M, beta: 1)
        var x = [Float](repeating: 0, count: M * N)
        gemm(ur, icn, &x, M, N, H); gemm(ui, isn, &x, M, N, H, alpha: -1, beta: 1)
        return x
    }
}
