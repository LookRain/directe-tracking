import Testing
@testable import TrackerCore

@Test func boxIoU() {
    let a = Box(0, 0, 10, 10), b = Box(5, 0, 15, 10)
    #expect(abs(a.iou(b) - 1.0 / 3.0) < 1e-9)
    #expect(AppearanceSpec.dim == 17)
}

@Test func aDoubtDropsOnlyItsReach() {
    // TrackerConfig.doubtReach: a doubt at 10 s with a 2 s reach keeps what was learnt before 8 s
    let R = AppearanceSpec.regions.count, D = AppearanceSpec.dim
    var p = Profile(feat: [Float](repeating: 1 / Float(D), count: R * D), valid: [Bool](repeating: true, count: R), emb: [1, 0])
    p.pending = stride(from: 0.0, through: 10, by: 1).map { ($0, .gallery([1, 0])) }
    var q = p
    q.doubt(since: 8)
    #expect(q.pending.map(\.t) == [0, 1, 2, 3, 4, 5, 6, 7])
    p.doubt()
    #expect(p.pending.isEmpty, "without a reach: everything")
}
