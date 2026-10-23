import Foundation

/// Colour-signature similarity (directe_tracking/appearance.py `similarity_many`), float32 like the lab.
/// Features are flat: `AppearanceSpec.regions.count` x `AppearanceSpec.dim` per person.
public enum Appearance {
    static let R = AppearanceSpec.regions.count
    static let D = AppearanceSpec.dim
    static let W = AppearanceSpec.weights

    /// Bhattacharyya coefficient of region `r` of `feats` (person at `offset`) and `profile`.
    @inline(__always)
    static func bc(_ feats: [Float], _ offset: Int, _ profile: [Float], _ r: Int) -> Float {
        withUnsafeTemporaryAllocation(of: Float.self, capacity: D) { tmp in
            for k in 0..<D { tmp[k] = (feats[offset + r * D + k] * profile[r * D + k]).squareRoot() }
            return npSum(UnsafeBufferPointer(tmp))
        }
    }

    /// Similarity per candidate in [0, 1], or `AppearanceSpec.unknown` when too little of the body was
    /// seen in both. `feats`: n x regions x dim (flat), `valids`: n x regions.
    public static func similarityMany(profile: [Float], pvalid: [Bool], feats: [Float], valids: [Bool]) -> [Float] {
        let n = valids.count / R
        var out = [Float](repeating: 0, count: n)
        for i in 0..<n {
            var denom: Float = 0, num: Float = 0, worst: Float = 0
            for r in 0..<R {
                let both = valids[i * R + r] && pvalid[r]
                let b = bc(feats, i * R * D, profile, r)
                let w: Float = both ? W[r] : 0
                denom = denom + w
                num = num + b * w
                let c: Float = both && W[r] >= Float(0.2 as Double) ? b : 1
                worst = r == 0 ? c : min(worst, c)
            }
            let mean = num / max(denom, Float(1e-6 as Double))
            let sim = min(max(mean - AppearanceSpec.vetoGain * max(0, AppearanceSpec.vetoBelow - worst), 0), 1)
            out[i] = denom >= AppearanceSpec.minEvidence ? sim : AppearanceSpec.unknown
        }
        return out
    }

    /// One person against one profile (lab `similarity`).
    public static func similarity(_ fa: [Float], _ va: [Bool], _ fb: [Float], _ vb: [Bool]) -> Float {
        similarityMany(profile: fa, pvalid: va, feats: fb, valids: vb)[0]
    }

    /// lab tracker `_sim_weighted`: similarity_many with per-region weight multipliers (float64
    /// weights, so the result is float64).
    static func simWeighted(profile: [Float], pvalid: [Bool], feats: [Float], valids: [Bool], wmul: [Double]) -> [Double] {
        let n = valids.count / R
        var out = [Double](repeating: 0, count: n)
        for i in 0..<n {
            var evidence: Float = 0, wsum = 0.0, num = 0.0, worst: Float = 0
            for r in 0..<R {
                let both = valids[i * R + r] && pvalid[r]
                let b = bc(feats, i * R * D, profile, r)
                let w = both ? Double(W[r]) * wmul[r] : 0
                num = num + Double(b) * w
                wsum = wsum + w
                evidence = evidence + (both ? W[r] : 0)
                let c: Float = both && W[r] >= Float(0.2 as Double) && wmul[r] >= 0.5 ? b : 1
                worst = r == 0 ? c : min(worst, c)
            }
            let mean = num / max(wsum, 1e-6)
            let pen = AppearanceSpec.vetoGain * max(0, AppearanceSpec.vetoBelow - worst)
            let s = min(max(mean - Double(pen), 0), 1)
            out[i] = evidence >= AppearanceSpec.minEvidence ? s : Double(AppearanceSpec.unknown)
        }
        return out
    }
}
