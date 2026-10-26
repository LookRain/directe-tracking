import CoreVideo
import Foundation
import Testing
import TrackerCore
@testable import Perception

/// `MotionFollow`: a small figure the detector stops seeing is followed while it moves, never onto someone the
/// detector does see, and not past the coast time.
@Suite struct FollowTests {
    /// A textured wall (fixed pattern) with a 14 x 40 px figure (a two-tone body) at (x, y).
    static func frame(figureAt p: SIMD2<Int>?, width W: Int = 320, height H: Int = 480) -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, W, H, kCVPixelFormatType_32BGRA, nil, &pb)
        CVPixelBufferLockBaseAddress(pb!, [])
        let base = CVPixelBufferGetBaseAddress(pb!)!.assumingMemoryBound(to: UInt8.self)
        let rb = CVPixelBufferGetBytesPerRow(pb!)
        for y in 0..<H {
            for x in 0..<W {
                var v = UInt8(150 + (((x / 7) * 31 + (y / 5) * 17) % 40))  // a fixed, busy wall
                var c = SIMD3<UInt8>(v, v, v)
                if let p, x >= p.x, x < p.x + 14, y >= p.y, y < p.y + 40 {
                    v = y < p.y + 18 ? 40 : 200
                    c = y < p.y + 18 ? SIMD3(30, 40, 200) : SIMD3(v, 60, 30)  // red top, blue bottom (BGR)
                }
                let o = y * rb + 4 * x
                base[o] = c.x; base[o + 1] = c.y; base[o + 2] = c.z; base[o + 3] = 255
            }
        }
        CVPixelBufferUnlockBaseAddress(pb!, [])
        return pb!
    }

    static func box(_ p: SIMD2<Int>) -> Box { Box(Double(p.x), Double(p.y), Double(p.x + 14), Double(p.y + 40)) }

    @Test func followsAMovingFigureTheDetectorMisses() throws {
        let f = MotionFollow()
        var p = SIMD2(150, 200)
        var t = 0.0
        _ = try f.propose(t: t, frame: Self.frame(figureAt: p), offset: .zero, shiftKnown: true, predicted: Self.box(p), boxes: [Self.box(p)])
        try f.after(t: t, frame: Self.frame(figureAt: p), state: .locked, chosen: Self.box(p))
        for _ in 0..<8 {  // the detector sees nobody; the figure climbs 4 px a look
            t += 0.1
            p.y -= 4
            let fr = Self.frame(figureAt: p)
            let nb = try f.propose(t: t, frame: fr, offset: .zero, shiftKnown: true, predicted: Self.box(p &+ SIMD2(0, 4)), boxes: [])
            let got = try #require(nb)
            #expect(got.iou(Self.box(p)) > 0.7)
            try f.after(t: t, frame: fr, state: .locked, chosen: got)
        }
        #expect(f.accepted == 8)
    }

    @Test func notOntoSomeoneTheDetectorSees() throws {
        let f = MotionFollow()
        let p = SIMD2(150, 200)
        _ = try f.propose(t: 0, frame: Self.frame(figureAt: p), offset: .zero, shiftKnown: true, predicted: Self.box(p), boxes: [Self.box(p)])
        try f.after(t: 0, frame: Self.frame(figureAt: p), state: .locked, chosen: Self.box(p))
        let q = SIMD2(150, 196)
        // a detection at the prediction: nothing to do
        #expect(try f.propose(t: 0.1, frame: Self.frame(figureAt: q), offset: .zero, shiftKnown: true,
                              predicted: Self.box(p), boxes: [Self.box(q)]) == nil)
    }

    @Test func stopsAfterTheCoastTime() throws {
        let f = MotionFollow()
        var p = SIMD2(150, 200)
        _ = try f.propose(t: 0, frame: Self.frame(figureAt: p), offset: .zero, shiftKnown: true, predicted: Self.box(p), boxes: [Self.box(p)])
        try f.after(t: 0, frame: Self.frame(figureAt: p), state: .locked, chosen: Self.box(p))
        var last: Box? = Self.box(p)
        var t = 0.0
        while t < f.maxCoastS + 1 {
            t += 0.5
            p.y -= 2
            last = try f.propose(t: t, frame: Self.frame(figureAt: p), offset: .zero, shiftKnown: true, predicted: Self.box(p), boxes: [])
            if let b = last { try f.after(t: t, frame: Self.frame(figureAt: p), state: .locked, chosen: b) }
        }
        #expect(last == nil)
    }

    @Test func aWallWithNobodyOnItIsNotFollowed() throws {
        let f = MotionFollow()
        let p = SIMD2(150, 200)
        _ = try f.propose(t: 0, frame: Self.frame(figureAt: p), offset: .zero, shiftKnown: true, predicted: Self.box(p), boxes: [Self.box(p)])
        try f.after(t: 0, frame: Self.frame(figureAt: p), state: .locked, chosen: Self.box(p))
        // the figure is gone (out of view): the wall alone doesn't match the figure's patch
        #expect(try f.propose(t: 0.1, frame: Self.frame(figureAt: nil), offset: .zero, shiftKnown: true, predicted: Self.box(p), boxes: []) == nil)
    }
}
