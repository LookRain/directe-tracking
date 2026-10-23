import Foundation
import TrackerCore

// Colour signature of one person: exact port of directe_tracking/appearance.py features() (VERSION 4).
// Regions come only from confident keypoints; with a body mask, only pixels inside it count.

private let hsvShift = 12
private let sdivTable: [Int] = (0..<256).map { $0 == 0 ? 0 : Int((Double(255 << hsvShift) / Double($0)).rounded(.toNearestOrEven)) }
private let hdivTable: [Int] = (0..<256).map { $0 == 0 ? 0 : Int((Double(180 << hsvShift) / (6.0 * Double($0))).rounded(.toNearestOrEven)) }

/// cv2.cvtColor(BGR2HSV) for one 8-bit pixel: H in 0...179, S and V in 0...255.
@inline(__always) func cvHSV(_ b: Int, _ g: Int, _ r: Int) -> (h: Int, s: Int, v: Int) {
    let v = max(b, g, r), vmin = min(b, g, r)
    let diff = v - vmin
    let vr = v == r ? -1 : 0, vg = v == g ? -1 : 0
    let s = (diff * sdivTable[v] + (1 << (hsvShift - 1))) >> hsvShift
    var h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + (~vg & (r - g + 4 * diff))))
    h = (h * hdivTable[diff] + (1 << (hsvShift - 1))) >> hsvShift
    if h < 0 { h += 180 }
    return (h, s, v)
}

enum Appearance {
    static let regions = AppearanceSpec.regions
    static let dim = AppearanceSpec.dim
    static let kpConf = AppearanceSpec.kpConf
    static let lSh = 5, rSh = 6, lHip = 11, rHip = 12, lKnee = 13, rKnee = 14
    static let head = [0, 1, 2, 3, 4]

    typealias P = SIMD2<Float>

    private static func norm(_ p: P) -> Float { (p.x * p.x + p.y * p.y).squareRoot() }

    private static func quad(_ p: P, _ q: P, _ halfWidth: Float) -> [SIMD2<Double>] {
        let d = q - p
        var n = P(-d.y, d.x)
        n /= max(norm(n), 1e-6)
        n *= halfWidth
        return [p + n, q + n, q - n, p - n].map { SIMD2<Double>(Double($0.x), Double($0.y)) }
    }

    private static func area(_ poly: [P]) -> Float {
        var a: Float = 0, b: Float = 0
        let n = poly.count
        for i in 0..<n {
            let j = (i + n - 1) % n
            a += poly[i].x * poly[j].y
            b += poly[i].y * poly[j].x
        }
        return 0.5 * abs(a - b)
    }

    /// lab region_polygons(): body regions placed by keypoints, indexed like AppearanceSpec.regions.
    static func regionPolygons(box: SIMD4<Float>, kpts: [Keypoint]) -> [[[SIMD2<Double>]]] {
        var polys = [[[SIMD2<Double>]]](repeating: [], count: regions.count)
        guard kpts.count == 17 else { return polys }
        let ok = kpts.map { $0.conf >= kpConf }
        let xy = kpts.map { P($0.x, $0.y) }
        let boxArea = max(Double((box[2] - box[0]) * (box[3] - box[1])), 1.0)
        guard ok[lSh] && ok[rSh] && ok[lHip] && ok[rHip] else { return polys }
        let sm = (xy[lSh] + xy[rSh]) / 2, hm = (xy[lHip] + xy[rHip]) / 2
        let up = sm - hm
        let length = Double(norm(up))
        let hipVec = xy[rHip] - xy[lHip]
        let hipW = max(Double(norm(hipVec)), 0.25 * length, 4.0)
        let torso = [xy[lSh], xy[rSh], xy[rHip], xy[lHip]]
        if length > 6 && Double(area(torso)) >= 0.03 * boxArea {
            let c = (torso[0] + torso[1] + torso[2] + torso[3]) / 4
            polys[1] = [torso.map { let p = c + 0.75 * ($0 - c); return SIMD2<Double>(Double(p.x), Double(p.y)) }]
            let along = hipVec / max(norm(hipVec), 1e-6) * Float(0.65 * hipW)
            let across = up * 0.15
            polys[2] = [[hm - along + across, hm + along + across, hm + along - across, hm - along - across]
                .map { SIMD2<Double>(Double($0.x), Double($0.y)) }]
        }
        let headOK = head.filter { ok[$0] }
        if !headOK.isEmpty && length > 6 {
            var hc = P(0, 0)
            for i in headOK { hc += xy[i] }
            hc /= Float(headOK.count)
            let r = 0.15 * length
            polys[0] = [(0..<12).map { k in
                let a = Double(k) * (2 * Double.pi / 12)  // np.linspace(0, 2 pi, 12, endpoint=False)
                return SIMD2<Double>(Double(hc.x) + r * cos(a), Double(hc.y) + r * sin(a))
            }]
        }
        var legs: [[SIMD2<Double>]] = []
        for (hip, knee) in [(lHip, lKnee), (rHip, rKnee)] where ok[knee] && norm(xy[knee] - xy[hip]) > 4 {
            let mid = xy[hip] + 0.45 * (xy[knee] - xy[hip])
            legs.append(quad(xy[hip], mid, Float(max(0.15 * hipW, 2.0))))
        }
        polys[3] = legs
        return polys
    }

    /// lab _hist(): 12 smoothed hue bins for coloured pixels + 5 smoothed brightness bins for the rest.
    static func hist(hue: [Int], grey: [Int]) -> [Float] {
        var out = [Float](repeating: 0, count: dim)
        let hb = AppearanceSpec.hueBins
        if hue.contains(where: { $0 > 0 }) {
            for i in 0..<hb {
                out[i] = 0.25 * Float(hue[(i + hb - 1) % hb]) + 0.5 * Float(hue[i]) + 0.25 * Float(hue[(i + 1) % hb])
            }
        }
        let ng = grey.count
        for i in 0..<ng {
            let next: Float = i + 1 < ng ? Float(grey[i + 1]) : 0, prev: Float = i > 0 ? Float(grey[i - 1]) : 0
            out[hb + i] = 0.15 * next + 0.7 * Float(grey[i]) + 0.15 * prev
        }
        let total = out.reduce(Float(0), +)
        let d = max(total, 1)
        return out.map { $0 / d }
    }

    /// Rounded integer vertices of a polygon relative to a crop origin (np.round(poly - offset)).
    static func vertices(_ poly: [SIMD2<Double>], _ ox: Int, _ oy: Int) -> [IPoint] {
        poly.map { IPoint(Int(($0.x - Double(ox)).rounded(.toNearestOrEven)), Int(($0.y - Double(oy)).rounded(.toNearestOrEven))) }
    }

    /// lab features(frame, box, kpts, body): (feat[4][DIM], valid[4]).
    static func features(_ frame: PixelView, box: SIMD4<Float>, kpts: [Keypoint], body: [SIMD2<Double>]?) -> ([[Float]], [Bool]) {
        let W = frame.width, H = frame.height
        let x0 = max(Int(box[0]) - 2, 0), y0 = max(Int(box[1]) - 2, 0)
        let x1 = min(Int(box[2]) + 3, W), y1 = min(Int(box[3]) + 3, H)
        var feat = [[Float]](repeating: [Float](repeating: 0, count: dim), count: regions.count)
        var valid = [Bool](repeating: false, count: regions.count)
        if x1 - x0 < 3 || y1 - y0 < 3 { return (feat, valid) }
        let cw = x1 - x0, ch = y1 - y0
        var bodyMask: Mask?
        if let body, body.count >= 3 {
            var m = Mask(width: cw, height: ch)
            fillPoly(&m, vertices(body, x0, y0))
            bodyMask = m
        }
        let polys = regionPolygons(box: box, kpts: kpts)
        for i in 0..<regions.count where !polys[i].isEmpty {
            // Rasterise in the region's bounding box (clamped to the crop): the same pixels as a
            // crop-sized mask, since nothing is drawn outside the vertices' box.
            let verts = polys[i].map { vertices($0, x0, y0) }
            var bx0 = Int.max, by0 = Int.max, bx1 = Int.min, by1 = Int.min
            for v in verts { for p in v { bx0 = min(bx0, p.x); by0 = min(by0, p.y); bx1 = max(bx1, p.x); by1 = max(by1, p.y) } }
            bx0 = max(bx0, 0); by0 = max(by0, 0); bx1 = min(bx1, cw - 1); by1 = min(by1, ch - 1)
            if bx1 < bx0 || by1 < by0 { continue }
            let mw = bx1 - bx0 + 1, mh = by1 - by0 + 1
            var mask = Mask(width: mw, height: mh)
            let o = IPoint(bx0, by0)
            for v in verts { fillConvexPoly(&mask, v.map { $0 &- o }) }
            var hue = [Int](repeating: 0, count: AppearanceSpec.hueBins)
            var grey = [Int](repeating: 0, count: AppearanceSpec.greyEdges.count + 1)
            var n = 0
            mask.data.withUnsafeBufferPointer { md in
                for y in 0..<mh {
                    let row = frame.row(y0 + by0 + y)
                    for x in 0..<mw where md[y * mw + x] != 0 {
                        if let bodyMask, bodyMask.data[(by0 + y) * cw + bx0 + x] == 0 { continue }
                        let p = row + 4 * (x0 + bx0 + x)
                        let (h, s, v) = cvHSV(Int(p[0]), Int(p[1]), Int(p[2]))
                        n += 1
                        if s >= 50 && v >= 40 {
                            hue[min(h * AppearanceSpec.hueBins / 180, AppearanceSpec.hueBins - 1)] += 1
                        } else {
                            grey[(v >= 45 ? 1 : 0) + (v >= 90 ? 1 : 0) + (v >= 150 ? 1 : 0) + (v >= 200 ? 1 : 0)] += 1
                        }
                    }
                }
            }
            if n >= AppearanceSpec.minPixels {
                feat[i] = hist(hue: hue, grey: grey)
                valid[i] = true
            }
        }
        return (feat, valid)
    }
}
