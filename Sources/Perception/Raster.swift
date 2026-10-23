import Foundation

// Polygon rasterisation and contour tracing, ported from OpenCV 4.x (imgproc/src/drawing.cpp,
// contours.cpp) and Ultralytics (ops.masks2segments, converter.merge_multi_segment), so body masks
// and region masks cover exactly the pixels the lab's cv2.fillPoly / cv2.fillConvexPoly cover.

/// A binary mask (0/1), row-major.
struct Mask {
    let width: Int
    let height: Int
    var data: [UInt8]

    init(width: Int, height: Int) {
        self.width = width
        self.height = height
        data = [UInt8](repeating: 0, count: width * height)
    }

    @inline(__always) subscript(x: Int, y: Int) -> UInt8 {
        get { data[y * width + x] }
        set { data[y * width + x] = newValue }
    }

    mutating func hline(_ y: Int, _ x1: Int, _ x2: Int) {
        if x1 > x2 { return }
        let o = y * width
        for x in x1...x2 { data[o + x] = 1 }
    }

    /// self &= other
    mutating func and(_ o: Mask) {
        for i in 0..<data.count { data[i] &= o.data[i] }
    }
}

typealias IPoint = SIMD2<Int>

private let xyShift = 16
private let xyOne = 1 << 16

/// cv2.clipLine for an image of size (w, h).
private func clipLine(_ w: Int, _ h: Int, _ p1: inout IPoint, _ p2: inout IPoint) -> Bool {
    let right = w - 1, bottom = h - 1
    if w <= 0 || h <= 0 { return false }
    var x1 = p1.x, y1 = p1.y, x2 = p2.x, y2 = p2.y
    func code(_ x: Int, _ y: Int) -> Int { (x < 0 ? 1 : 0) + (x > right ? 2 : 0) + (y < 0 ? 4 : 0) + (y > bottom ? 8 : 0) }
    var c1 = code(x1, y1), c2 = code(x2, y2)
    if (c1 & c2) == 0 && (c1 | c2) != 0 {
        if c1 & 12 != 0 {
            let a = c1 < 8 ? 0 : bottom
            x1 += Int(Double(a - y1) * Double(x2 - x1) / Double(y2 - y1))
            y1 = a
            c1 = (x1 < 0 ? 1 : 0) + (x1 > right ? 2 : 0)
        }
        if c2 & 12 != 0 {
            let a = c2 < 8 ? 0 : bottom
            x2 += Int(Double(a - y2) * Double(x2 - x1) / Double(y2 - y1))
            y2 = a
            c2 = (x2 < 0 ? 1 : 0) + (x2 > right ? 2 : 0)
        }
        if (c1 & c2) == 0 && (c1 | c2) != 0 {
            if c1 != 0 {
                let a = c1 == 1 ? 0 : right
                y1 += Int(Double(a - x1) * Double(y2 - y1) / Double(x2 - x1))
                x1 = a
                c1 = 0
            }
            if c2 != 0 {
                let a = c2 == 1 ? 0 : right
                y2 += Int(Double(a - x2) * Double(y2 - y1) / Double(x2 - x1))
                x2 = a
                c2 = 0
            }
        }
    }
    p1 = IPoint(x1, y1); p2 = IPoint(x2, y2)
    return (c1 | c2) == 0
}

/// cv2.line(img, p1, p2, 1, LINE_8) (LineIterator, left to right).
func drawLine(_ m: inout Mask, _ a: IPoint, _ b: IPoint) {
    var p1 = a, p2 = b
    if p1.x < 0 || p1.x >= m.width || p2.x < 0 || p2.x >= m.width || p1.y < 0 || p1.y >= m.height || p2.y < 0 || p2.y >= m.height {
        if !clipLine(m.width, m.height, &p1, &p2) { return }
    }
    var dx = p2.x - p1.x, dy = p2.y - p1.y
    if dx < 0 { dx = -dx; dy = -dy; swap(&p1, &p2) }
    var sy = 1
    if dy < 0 { dy = -dy; sy = -1 }
    let vert = dy > dx
    if vert { swap(&dx, &dy) }
    var err = dx - (dy + dy)
    let plusDelta = dx + dx, minusDelta = -(dy + dy)
    var x = p1.x, y = p1.y
    for _ in 0..<(dx + 1) {
        m[x, y] = 1
        let minor = err < 0
        err += minusDelta + (minor ? plusDelta : 0)
        if vert { y += sy; if minor { x += 1 } } else { x += 1; if minor { y += sy } }
    }
}

/// cv2.fillConvexPoly(img, pts, 1) with LINE_8 and shift 0.
func fillConvexPoly(_ m: inout Mask, _ v: [IPoint]) {
    let npts = v.count
    if npts == 0 { return }
    var p0 = v[npts - 1]
    var xmin = v[0].x, xmax = v[0].x, ymin = v[0].y, ymax = v[0].y, imin = 0
    for i in 0..<npts {
        let p = v[i]
        if p.y < ymin { ymin = p.y; imin = i }
        ymax = max(ymax, p.y); xmax = max(xmax, p.x); xmin = min(xmin, p.x)
        drawLine(&m, p0, p)
        p0 = p
    }
    if npts < 3 || xmax < 0 || ymax < 0 || xmin >= m.width || ymin >= m.height { return }
    ymax = min(ymax, m.height - 1)
    struct Edge { var idx: Int; var di: Int; var x: Int; var dx: Int; var ye: Int }
    var edge = [Edge(idx: imin, di: 1, x: -xyOne, dx: 0, ye: ymin), Edge(idx: imin, di: npts - 1, x: -xyOne, dx: 0, ye: ymin)]
    var y = ymin
    var edges = npts
    let half = xyOne >> 1
    repeat {
        for i in 0..<2 where y >= edge[i].ye {
            var idx0 = edge[i].idx
            let di = edge[i].di
            var idx = idx0 + di
            if idx >= npts { idx -= npts }
            while true {
                let more = edges > 0
                edges -= 1
                if !more { break }
                let ty = v[idx].y
                if ty > y {
                    let xs = v[idx0].x << xyShift, xe = v[idx].x << xyShift
                    edge[i].ye = ty
                    edge[i].dx = ((xe - xs) * 2 + (ty - y)) / (2 * (ty - y))
                    edge[i].x = xs
                    edge[i].idx = idx
                    break
                }
                idx0 = idx
                idx += di
                if idx >= npts { idx -= npts }
            }
        }
        if edges < 0 { break }
        if y >= 0 {
            let (l, r) = edge[0].x > edge[1].x ? (1, 0) : (0, 1)
            var x1 = (edge[l].x + half) >> xyShift, x2 = (edge[r].x + half) >> xyShift
            if x2 >= 0 && x1 < m.width {
                x1 = max(x1, 0); x2 = min(x2, m.width - 1)
                m.hline(y, x1, x2)
            }
        }
        edge[0].x += edge[0].dx
        edge[1].x += edge[1].dx
        y += 1
    } while y <= ymax
}

/// cv2.fillPoly(img, [pts], 1) with LINE_8 and shift 0 (CollectPolyEdges + FillEdgeCollection).
func fillPoly(_ m: inout Mask, _ v: [IPoint]) {
    let count = v.count
    if count == 0 { return }
    struct PolyEdge { var y0: Int; var y1: Int; var x: Int; var dx: Int }
    var edges: [PolyEdge] = []
    var pt0 = IPoint(v[count - 1].x << xyShift, v[count - 1].y)
    for i in 0..<count {
        let pt1 = IPoint(v[i].x << xyShift, v[i].y)
        var t0 = IPoint((pt0.x + (xyOne >> 1)) >> xyShift, pt0.y)
        var t1 = IPoint((pt1.x + (xyOne >> 1)) >> xyShift, pt1.y)
        drawLine(&m, t0, t1)
        var pt0c = pt0, pt1c = pt1
        if t0.x < 0 || t0.x >= m.width || t1.x < 0 || t1.x >= m.width || t0.y < 0 || t0.y >= m.height || t1.y < 0 || t1.y >= m.height {
            _ = clipLine(m.width, m.height, &t0, &t1)
            if t0.y != t1.y { pt0c.y = t0.y; pt1c.y = t1.y }
        }
        pt0c.x = t0.x << xyShift
        pt1c.x = t1.x << xyShift
        defer { pt0 = pt1 }
        if pt0.y == pt1.y { continue }
        let dx = (pt1c.x - pt0c.x) / (pt1c.y - pt0c.y)
        if pt0.y < pt1.y {
            edges.append(PolyEdge(y0: pt0.y, y1: pt1.y, x: pt0c.x + (pt0.y - pt0c.y) * dx, dx: dx))
        } else {
            edges.append(PolyEdge(y0: pt1.y, y1: pt0.y, x: pt1c.x + (pt1.y - pt1c.y) * dx, dx: dx))
        }
    }
    if edges.count < 2 { return }
    var yMin = Int.max, yMax = Int.min, xMin = Int.max, xMax = Int.min
    for e in edges {
        let xEnd = e.x + (e.y1 - e.y0) * e.dx
        yMin = min(yMin, e.y0); yMax = max(yMax, e.y1)
        xMin = min(xMin, e.x, xEnd); xMax = max(xMax, e.x, xEnd)
    }
    if yMax < 0 || yMin >= m.height || xMax < 0 || xMin >= (m.width << xyShift) { return }
    yMax = min(yMax, m.height)
    var xs: [Int] = []
    for y in yMin..<max(yMin, yMax) {
        xs.removeAll(keepingCapacity: true)
        for e in edges where e.y0 <= y && y < e.y1 { xs.append(e.x + (y - e.y0) * e.dx) }
        if y < 0 || xs.count < 2 { continue }
        xs.sort()
        var k = 0
        while k + 1 < xs.count {
            var x1 = (xs[k] + xyOne - 1) >> xyShift, x2 = xs[k + 1] >> xyShift
            if x1 < m.width && x2 >= 0 {
                x1 = max(x1, 0); x2 = min(x2, m.width - 1)
                m.hline(y, x1, x2)
            }
            k += 2
        }
    }
}

// MARK: - contours

/// cv2.findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)[0]: outer borders of the 8-connected
/// components that are not inside another one, in OpenCV's order (last found first).
func findExternalContours(_ mask: Mask) -> [[IPoint]] {
    let w = mask.width + 2, h = mask.height + 2
    var img = [Int8](repeating: 0, count: w * h)
    for y in 0..<mask.height {
        for x in 0..<mask.width where mask[x, y] != 0 { img[(y + 1) * w + x + 1] = 1 }
    }
    let deltas: [Int] = { let d = [1, -w + 1, -w, -w - 1, -1, w - 1, w, w + 1]; return d + d }()
    let codeDeltas: [IPoint] = [IPoint(1, 0), IPoint(1, -1), IPoint(0, -1), IPoint(-1, -1),
                                IPoint(-1, 0), IPoint(-1, 1), IPoint(0, 1), IPoint(1, 1)]
    let nbd: Int8 = 2, rightBound: Int8 = 2 | -128

    func fetch(_ i0: Int, _ origin: IPoint) -> [IPoint] {
        var pts: [IPoint] = []
        var pt = origin
        var s = 4, sEnd = 4
        var i1 = 0
        repeat {
            s = (s - 1) & 7
            i1 = i0 + deltas[s]
        } while img[i1] == 0 && s != sEnd
        if s == sEnd {
            img[i0] = rightBound
            return [pt]
        }
        var i3 = i0, i4 = 0
        var prevS = s ^ 4
        while true {
            sEnd = s
            s = min(s, 15)
            while s < 15 {
                s += 1
                i4 = i3 + deltas[s]
                if img[i4] != 0 { break }
            }
            s &= 7
            if UInt(bitPattern: s - 1) < UInt(sEnd) {
                img[i3] = rightBound
            } else if img[i3] == 1 {
                img[i3] = nbd
            }
            if s != prevS {
                pts.append(pt)
                prevS = s
            }
            pt &+= codeDeltas[s]
            if i4 == i0 && i3 == i1 { break }
            i3 = i4
            s = (s + 4) & 7
        }
        return pts
    }

    var found: [[IPoint]] = []
    for y in 1..<(h - 1) {
        var lnbdX = 0
        var prev = img[y * w]
        var x = 1
        while x < w - 1 {
            let p = img[y * w + x]
            if p != prev {
                var isHole = false
                var skip = false
                if !(prev == 0 && p == 1) {
                    if p != 0 || prev < 1 {
                        skip = true
                    } else {
                        if prev & -2 != 0 { lnbdX = x - 1 }
                        isHole = true
                    }
                }
                if !skip && (isHole || img[y * w + lnbdX] > 0) { skip = true }
                if !skip {
                    let ox = x - (isHole ? 1 : 0)
                    found.append(fetch(y * w + ox, IPoint(ox - 1, y - 1)))
                    lnbdX = ox
                    prev = img[y * w + x]
                    x += 1
                    continue
                }
            }
            prev = p
            if prev & -2 != 0 { lnbdX = x }
            x += 1
        }
    }
    return found.reversed()
}

/// Ultralytics converter.merge_multi_segment: joins segments at their closest points with thin
/// (doubled-back) bridges, so the polygon covers every component.
func mergeMultiSegment(_ input: [[IPoint]]) -> [IPoint] {
    var segments = input
    let n = segments.count
    var idxList = [[Int]](repeating: [], count: n)
    for i in 1..<n {
        let a = segments[i - 1], b = segments[i]
        var best = Int.max, bi = 0, bj = 0
        for (ii, p) in a.enumerated() {
            for (jj, q) in b.enumerated() {
                let d = (p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y)
                if d < best { best = d; bi = ii; bj = jj }
            }
        }
        idxList[i - 1].append(bi)
        idxList[i].append(bj)
    }
    var s: [[IPoint]] = []
    for i in 0..<n {
        var idx = idxList[i]
        if idx.count == 2 && idx[0] > idx[1] {
            idx.reverse()
            segments[i].reverse()
        }
        let k = ((idx[0] % segments[i].count) + segments[i].count) % segments[i].count
        segments[i] = Array(segments[i][k...] + segments[i][..<k])
        segments[i].append(segments[i][0])
        if i == 0 || i == n - 1 {
            s.append(segments[i])
        } else {
            let hi = min(idx[1] - idx[0], segments[i].count - 1)
            s.append(Array(segments[i][0...hi]))
        }
    }
    for i in stride(from: n - 1, through: 0, by: -1) where i != 0 && i != n - 1 {
        let idx = idxList[i]
        let nidx = abs(idx[1] - idx[0])
        if nidx < segments[i].count { s.append(Array(segments[i][nidx...])) }
    }
    return s.flatMap { $0 }
}

/// ops.masks2segments(mask[None])[0] with the default strategy "all".
func maskSegment(_ mask: Mask) -> [IPoint] {
    let cs = findExternalContours(mask)
    if cs.isEmpty { return [] }
    return cs.count > 1 ? mergeMultiSegment(cs) : cs[0]
}
