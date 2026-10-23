import Foundation
import Testing
@testable import Perception

// OpenCV / Ultralytics building blocks against perception_units.json
// (tools/ios_golden/perception_golden.py units()). These are exact ports: pixel-exact expected.

@Suite struct OpenCVPorts {
    let u: [String: Any]
    init() throws { u = try Golden.json("perception_units.json") }

    /// H and V exact; S may differ by 1 in a handful of pixels (this cv2 build converts with
    /// KleidiCV, whose S rounding differs from OpenCV's table maths at x.5; features only threshold S).
    @Test func hsvMatchesCv2() {
        let h = u["hsv"] as! [String: Any]
        let bgr = Golden.bytes(h["bgr"]), hsv = Golden.bytes(h["hsv"])
        var bad = 0, sOff = 0
        for i in 0..<(bgr.count / 3) {
            let r = cvHSV(Int(bgr[3 * i]), Int(bgr[3 * i + 1]), Int(bgr[3 * i + 2]))
            if r.h != Int(hsv[3 * i]) || r.v != Int(hsv[3 * i + 2]) || abs(r.s - Int(hsv[3 * i + 1])) > 1 { bad += 1 }
            if r.s != Int(hsv[3 * i + 1]) { sOff += 1 }
        }
        print("HSV: \(sOff) of \(bgr.count / 3) pixels with S off by 1, \(bad) worse")
        #expect(bad == 0)
        #expect(sOff * 1000 <= bgr.count / 3)
    }

    /// Exact where cv2 runs OpenCV's own resize (upscales, one-axis scales, integer factors other
    /// than 2); within 1 where this arm64 cv2 build hands the job to its HAL (bilinear shrinking both
    /// axes, exact 2x): measured ~20% of values off by 1 there, never more.
    @Test func resizeMatchesCv2() {
        for case let c as [String: Any] in u["resize"] as! [Any] {
            let sw = c["src_w"] as! Int, sh = c["src_h"] as! Int, w = c["w"] as! Int, h = c["h"] as! Int
            let area = c["interp"] as! String == "area"
            let src = Golden.image(bgr: Golden.bytes(c["src"]), width: sw, height: sh)
            let want = Golden.bytes(c["dst"])
            let dst = BGRAImage(width: w, height: h)
            cvResize(src.view, dst.view, area ? .area : .linear)
            var bad = 0, worst = 0
            for i in 0..<(w * h) {
                for k in 0..<3 {
                    let d = abs(Int(dst.base[4 * i + k]) - Int(want[3 * i + k]))
                    if d != 0 { bad += 1; worst = max(worst, d) }
                }
            }
            let hal = !area && w < sw && h < sh && !(sw % w == 0 && sh % h == 0 && sw / w != 2)
            if hal {
                print("resize linear \(sw)x\(sh) -> \(w)x\(h): \(bad) of \(3 * w * h) values off by \(worst)")
                #expect(worst <= 1)
            } else {
                #expect(bad == 0, "\(c["interp"]!) \(sw)x\(sh) -> \(w)x\(h): \(bad) values differ (max \(worst))")
            }
        }
    }

    private func pts(_ a: Any?) -> [IPoint] { (a as! [[Int]]).map { IPoint($0[0], $0[1]) } }

    @Test func fillConvexPolyMatchesCv2() {
        var failures = 0
        for case let c as [String: Any] in u["fill_convex"] as! [Any] {
            var m = Mask(width: 40, height: 40)
            fillConvexPoly(&m, pts(c["pts"]))
            let want = Golden.bytes(c["mask"])
            let diff = zip(m.data, want).filter { $0 != $1 }.count
            if diff != 0 { failures += 1; print("fillConvexPoly \(c["pts"]!): \(diff) px differ") }
        }
        #expect(failures == 0)
    }

    @Test func fillPolyMatchesCv2() {
        var failures = 0
        for case let c as [String: Any] in u["fill_poly"] as! [Any] {
            var m = Mask(width: 48, height: 48)
            fillPoly(&m, pts(c["pts"]))
            let want = Golden.bytes(c["mask"])
            let diff = zip(m.data, want).filter { $0 != $1 }.count
            if diff != 0 {
                failures += 1
                if failures <= 3 {
                    var cells: [String] = []
                    for i in 0..<want.count where m.data[i] != want[i] { cells.append("(\(i % 48),\(i / 48)) got \(m.data[i])") }
                    print("fillPoly #\(failures) pts=\((c["pts"] as! [[Int]]).map { "\($0[0]),\($0[1])" }.joined(separator: ";")) diff=\(cells.joined(separator: " "))")
                }
            }
        }
        #expect(failures == 0)
    }

    @Test func contoursMatchCv2AndUltralytics() {
        var failures = 0
        for case let c as [String: Any] in u["contours"] as! [Any] {
            let w = c["w"] as! Int, h = c["h"] as! Int
            var m = Mask(width: w, height: h)
            m.data = Golden.bytes(c["mask"])
            let got = findExternalContours(m)
            let want = (c["contours"] as! [Any]).map { pts($0) }
            if got != want { failures += 1; print("contours differ: \(got.count) vs \(want.count)") }
            if maskSegment(m) != pts(c["segment"]) { failures += 1; print("segment differs") }
        }
        #expect(failures == 0)
    }

    @Test func optimalDFTSizes() {
        #expect(optimalDFTSize(360) == 360)
        #expect(optimalDFTSize(640) == 640)
        #expect(optimalDFTSize(202) == 216)
        #expect(optimalDFTSize(7) == 8)
    }

    @Test func windowMatchesLab() {
        // lab _window(cx, cy, side, W, H)
        #expect(Perceiver.window(cx: 100, cy: 1800, side: 960, width: 1080, height: 1920) == Box(0, 960, 960, 1920))
        #expect(Perceiver.window(cx: 540, cy: 960, side: 2000, width: 1080, height: 1920) == Box(0, 420, 1080, 1500))
    }

    @Test func letterboxMatchesUltralytics() {
        let lb = Letterbox(srcW: 1080, srcH: 1920, auto: true)
        #expect(lb.canvasW == 384 && lb.canvasH == 640 && lb.newW == 360 && lb.left == 12 && lb.top == 0)
        let l2 = Letterbox(srcW: 1920, srcH: 1080, auto: true)
        #expect(l2.canvasW == 640 && l2.canvasH == 384 && l2.top == 12)
        let sq = Letterbox(srcW: 1000, srcH: 1000, auto: true)
        #expect(sq.canvasW == 640 && sq.canvasH == 640 && sq.left == 0)
    }
}
