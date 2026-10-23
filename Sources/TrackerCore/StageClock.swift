import Foundation

/// Wall time and calls per named stage, summed over a job, for the app's diagnostics report. A class so the
/// Perceiver and the job loop can share one; stages that run at the same time (on several threads) each count
/// their own wall time, so the stages can add up to more than the job.
public final class StageClock: @unchecked Sendable {
    private let lock = NSLock()
    private var _seconds: [String: Double] = [:]
    private var _counts: [String: Int] = [:]

    public var seconds: [String: Double] { lock.withLock { _seconds } }
    public var counts: [String: Int] { lock.withLock { _counts } }

    public init() {}

    @discardableResult
    public func time<R>(_ stage: String, _ body: () throws -> R) rethrows -> R {
        let t0 = DispatchTime.now().uptimeNanoseconds
        defer { add(stage, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9) }
        return try body()
    }

    public func add(_ stage: String, _ s: Double) {
        lock.withLock {
            _seconds[stage, default: 0] += s
            _counts[stage, default: 0] += 1
        }
    }
}
