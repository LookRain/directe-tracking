import CoreVideo
import Foundation

// Pixels as the lab sees them: 8-bit BGR (here BGRA, alpha ignored), row-major, origin top-left.
// Frames come in as kCVPixelFormatType_32BGRA pixel buffers (AVAssetReader's native output).

public enum PerceptionError: Error, CustomStringConvertible {
    case pixelFormat(OSType)
    case pixelBuffer(CVReturn)
    case modelMissing(String)
    case modelOutput(String)

    public var description: String {
        switch self {
        case .pixelFormat(let f): return "frames must be kCVPixelFormatType_32BGRA, got \(f)"
        case .pixelBuffer(let r): return "CVPixelBufferCreate failed (\(r))"
        case .modelMissing(let n): return "Core ML model \(n) not found"
        case .modelOutput(let n): return "unexpected Core ML output: \(n)"
        }
    }
}

/// A BGRA pixel region with a row stride: a locked pixel buffer, an owned image, or a crop of either.
/// Valid only while its owner is alive (and locked).
struct PixelView {
    var base: UnsafeMutablePointer<UInt8>
    var width: Int
    var height: Int
    var rowBytes: Int

    @inline(__always) func row(_ y: Int) -> UnsafeMutablePointer<UInt8> { base + y * rowBytes }

    /// The sub-rectangle [x, x + w) x [y, y + h) as its own image (resizing it clamps at its edges,
    /// exactly like resizing `frame[y:y+h, x:x+w]` in numpy/cv2).
    func crop(_ x: Int, _ y: Int, _ w: Int, _ h: Int) -> PixelView {
        PixelView(base: base + y * rowBytes + x * 4, width: w, height: h, rowBytes: rowBytes)
    }

    func fill(_ b: UInt8, _ g: UInt8, _ r: UInt8) {
        for y in 0..<height {
            let p = row(y)
            for x in 0..<width {
                p[4 * x] = b; p[4 * x + 1] = g; p[4 * x + 2] = r; p[4 * x + 3] = 255
            }
        }
    }

    func copy(to dst: PixelView) {
        precondition(dst.width == width && dst.height == height)
        for y in 0..<height { dst.row(y).update(from: row(y), count: width * 4) }
    }
}

/// An owned BGRA image (crops, masked crops, small working copies).
final class BGRAImage {
    let width: Int
    let height: Int
    let base: UnsafeMutablePointer<UInt8>

    init(width: Int, height: Int) {
        self.width = width
        self.height = height
        base = .allocate(capacity: max(width * height * 4, 4))
        base.initialize(repeating: 0, count: max(width * height * 4, 4))
    }

    deinit { base.deallocate() }

    var view: PixelView { PixelView(base: base, width: width, height: height, rowBytes: width * 4) }
}

/// Locks a BGRA pixel buffer and hands out a view of it.
func withPixels<R>(_ pb: CVPixelBuffer, readOnly: Bool = true, _ body: (PixelView) throws -> R) throws -> R {
    let fmt = CVPixelBufferGetPixelFormatType(pb)
    guard fmt == kCVPixelFormatType_32BGRA else { throw PerceptionError.pixelFormat(fmt) }
    let flags: CVPixelBufferLockFlags = readOnly ? .readOnly : []
    CVPixelBufferLockBaseAddress(pb, flags)
    defer { CVPixelBufferUnlockBaseAddress(pb, flags) }
    let view = PixelView(base: CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self),
                         width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb),
                         rowBytes: CVPixelBufferGetBytesPerRow(pb))
    return try body(view)
}

/// IOSurface-backed BGRA buffers for model inputs, pooled per size. Safe to use from the passes that run at the
/// same time.
final class BufferPool: @unchecked Sendable {
    private var pools: [SIMD2<Int>: CVPixelBufferPool] = [:]
    private let lock = NSLock()

    func make(width: Int, height: Int) throws -> CVPixelBuffer {
        let key = SIMD2(width, height)
        lock.lock()
        defer { lock.unlock() }
        if pools[key] == nil {
            let attrs: [CFString: Any] = [
                kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey: width, kCVPixelBufferHeightKey: height,
                kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
            ]
            var pool: CVPixelBufferPool?
            let r = CVPixelBufferPoolCreate(nil, nil, attrs as CFDictionary, &pool)
            guard r == kCVReturnSuccess, let pool else { throw PerceptionError.pixelBuffer(r) }
            pools[key] = pool
        }
        var pb: CVPixelBuffer?
        let r = CVPixelBufferPoolCreatePixelBuffer(nil, pools[key]!, &pb)
        guard r == kCVReturnSuccess, let pb else { throw PerceptionError.pixelBuffer(r) }
        return pb
    }
}

/// Python's round() (half to even), as the lab's letterbox and scale maths use it.
@inline(__always) func pyRound(_ x: Double) -> Int { Int(x.rounded(.toNearestOrEven)) }
