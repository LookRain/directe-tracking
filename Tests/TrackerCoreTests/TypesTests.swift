import Testing
@testable import TrackerCore

@Test func boxIoU() {
    let a = Box(0, 0, 10, 10), b = Box(5, 0, 15, 10)
    #expect(abs(a.iou(b) - 1.0 / 3.0) < 1e-9)
    #expect(AppearanceSpec.dim == 17)
}
