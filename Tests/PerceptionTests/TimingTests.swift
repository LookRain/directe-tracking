import CoreML
import CoreVideo
import Foundation
import Testing
@testable import Perception

// Speed on this Mac (Neural Engine). Meaningful only in a release build:
//   swift test -c release -Xswiftc -enable-testing --filter PerceptionTiming
// (a debug build runs the pixel loops 10-50x slower).

func ms(_ body: () throws -> Void) rethrows -> Double {
    let t0 = DispatchTime.now().uptimeNanoseconds
    try body()
    return Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
}

func median(_ v: [Double]) -> Double { v.sorted()[v.count / 2] }

#if DEBUG
let releaseBuild = false
#else
let releaseBuild = true
#endif

@Suite(.serialized, .enabled(if: releaseBuild && Golden.haveFrames && Golden.haveModels))
struct PerceptionTiming {
    @Test(arguments: ["lite", "mid"])
    func timings(_ name: String) throws {
        let set = name == "lite" ? ModelSet.lite : ModelSet.mid
        var models: PerceptionModels!
        let load = try ms { models = try PerceptionModels(set, directory: Golden.models, computeUnits: .all) }
        let p = Perceiver(models: models)
        let fx = try Golden.json("perception_lite.json")
        var lines: [String] = [String(format: "=== %@ (%@): load %.0f ms", name, set.focusPose, load)]
        for case let c as [String: Any] in (fx["cases"] as! [Any]) where ["IMG_4783_000360.png", "IMG_4102_000300.png"].contains(c["frame"] as! String) {
            let pb = try Golden.pixelBuffer(png: Golden.frames.appendingPathComponent(c["frame"] as! String))
            let f = c["focus"] as! [Int]
            let focus = Box(Double(f[0]), Double(f[1]), Double(f[2]), Double(f[3]))
            let track = toBox(box4((c["second"] as! [String: Any])["pb"]))
            for _ in 0..<3 { _ = try p.perceive(pb, focus: focus) }  // warm up (ANE program load)
            var tPerceive: [Double] = [], tWide: [Double] = [], tFocusPose: [Double] = [], tSeg: [Double] = []
            var tDescribe: [Double] = [], tSecond: [Double] = [], tCam: [Double] = [], tPeople: [Double] = []
            var nCands = 0
            let cam = CameraMotion()
            _ = cam.shift(pb)
            for _ in 0..<10 {
                tPerceive.append(try ms { nCands = try p.perceive(pb, focus: focus).count })
                tSecond.append(try ms { _ = try p.secondChance(pb, predicted: track, candidates: []) })
                tCam.append(ms { _ = cam.shift(pb) })
                try withPixels(pb) { v in
                    let crop = v.crop(f[0], f[1], f[2] - f[0], f[3] - f[1])
                    tWide.append(try ms { _ = try models.widePose.detect(v, pool: p.pool) })
                    tFocusPose.append(try ms { _ = try models.focusPose.detect(crop, pool: p.pool) })
                    var segs: [SegDet] = []
                    tSeg.append(try ms { segs = try models.focusSeg!.detect(crop, pool: p.pool) })
                    let people = segs.map { Person(box: $0.box, kpts: [], poly: $0.poly.map { SIMD2(Double($0.x), Double($0.y)) }) }
                    tDescribe.append(try ms { _ = try p.describe(crop, people) })
                }
            }
            for _ in 0..<3 { tPeople.append(try ms { _ = try p.people(in: pb) }) }
            lines.append(String(format: "  %@ %dx%d, focus %d px, %d candidates: perceive %.1f ms (wide %.1f, focus pose %.1f, masks %.1f, describing the masked people %.1f); secondChance %.1f ms; camera shift %.1f ms; people(in:) tiled %.1f ms",
                                c["frame"] as! String, CVPixelBufferGetWidth(pb), CVPixelBufferGetHeight(pb), f[2] - f[0], nCands,
                                median(tPerceive), median(tWide), median(tFocusPose), median(tSeg), median(tDescribe),
                                median(tSecond), median(tCam), median(tPeople)))
        }
        print(lines.joined(separator: "\n"))
    }
}
