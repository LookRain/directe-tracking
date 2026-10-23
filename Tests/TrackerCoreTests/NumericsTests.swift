import Testing
@testable import TrackerCore

@Test func appearanceConstantsAreNumpyFloat32() {
    // numpy rounds the Python (double) constants to float32; Swift's Float literals must agree
    #expect(AppearanceSpec.weights == [0.15, 0.35, 0.2, 0.3].map { Float($0 as Double) })
    #expect(AppearanceSpec.vetoBelow == Float(0.55 as Double))
    #expect(AppearanceSpec.vetoGain == Float(0.8 as Double))
    #expect(AppearanceSpec.minEvidence == Float(0.3 as Double))
    #expect(AppearanceSpec.unknown == -1)
}

@Test func pairwiseSumMatchesNumpyOrder() {
    // 17 values: 8 accumulators over the first 16, then the 17th (numpy pairwise_sum)
    let a: [Float] = (0..<17).map { 1 / Float($0 + 3) }
    let r = (0..<8).map { a[$0] + a[$0 + 8] }
    let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7])) + a[16]
    #expect(a.withUnsafeBufferPointer { npSum($0) } == res)
    let small: [Float] = [0.1, 0.2, 0.3]
    #expect(small.withUnsafeBufferPointer { npSum($0) } == (0.1 as Float) + 0.2 + 0.3)
}

@Test func iouPrecision() {
    let a = Box(10, 10, 110, 210), b = Box(60, 10, 160, 210)
    let s32 = npIoU(a, true, [b], true), s64 = npIoU(a, false, [b], false)
    #expect(s32.f32 && !s64.f32)
    #expect(abs(s32.v[0] - 1.0 / 3.0) < 1e-6 && abs(s64.v[0] - 1.0 / 3.0) < 1e-12)
    #expect(s32.v[0] == Double(Float(s32.v[0])))
    #expect(npSamePerson32(a, [b, Box(20, 20, 100, 120), Box(30, 30, 40, 40)], 0.3) == [true, true, false])
}
