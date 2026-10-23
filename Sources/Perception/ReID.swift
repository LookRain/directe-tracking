import CoreML
import CoreVideo
import Foundation

// Re-ID embeddings as boxmot's ReID runs them in the lab (directe_tracking/identity.py Embedder): the crop is
// resized to 128x256 with cv2 INTER_LINEAR, BGR->RGB, /255, ImageNet mean/std (the last three are
// inside the exported model), OSNet, L2-normalised. Masked crops grey out (124) everything outside
// the person's mask (lab identity.masked_crop).

final class Embedder {
    let model: MLModel
    static let inputW = 128, inputH = 256

    init(model: MLModel) { self.model = model }

    /// lab masked_crop(frame, box, poly) -> (plain, masked); returns the one asked for.
    static func crop(_ frame: PixelView, box: SIMD4<Float>, poly: [SIMD2<Double>]?, masked: Bool) -> BGRAImage {
        let W = frame.width, H = frame.height
        let x0 = max(Int(box[0]), 0), y0 = max(Int(box[1]), 0)
        var x1 = min(Int(box[2].rounded(.up)), W), y1 = min(Int(box[3].rounded(.up)), H)
        x1 = min(max(x1, x0 + 2), W); y1 = min(max(y1, y0 + 2), H)  // numpy slicing clamps at the frame edge
        let w = max(x1 - x0, 0), h = max(y1 - y0, 0)
        let img = BGRAImage(width: w, height: h)
        if w == 0 || h == 0 { return img }
        let out = img.view
        frame.crop(x0, y0, w, h).copy(to: out)
        if masked, let poly, poly.count >= 3 {
            var m = Mask(width: w, height: h)
            fillPoly(&m, Appearance.vertices(poly, x0, y0))
            for y in 0..<h {
                let row = out.row(y)
                for x in 0..<w where m.data[y * w + x] == 0 {
                    row[4 * x] = 124; row[4 * x + 1] = 124; row[4 * x + 2] = 124
                }
            }
        }
        return img
    }

    /// L2-normalised embeddings of these crops (boxmot: an empty crop becomes a black 256x128 image).
    func embed(_ crops: [BGRAImage], pool: BufferPool) throws -> [[Float]] {
        if crops.isEmpty { return [] }
        var inputs: [MLFeatureProvider] = []
        for c in crops {
            let pb = try pool.make(width: Self.inputW, height: Self.inputH)
            try withPixels(pb, readOnly: false) { dst in
                if c.width == 0 || c.height == 0 { dst.fill(0, 0, 0) } else { cvResize(c.view, dst, .linear) }
            }
            inputs.append(try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: pb)]))
        }
        let out = try model.predictions(from: MLArrayBatchProvider(array: inputs), options: MLPredictionOptions())
        var embs: [[Float]] = []
        for i in 0..<out.count {
            guard let a = out.features(at: i).featureValue(for: "embedding")?.multiArrayValue else {
                throw PerceptionError.modelOutput("embedding")
            }
            var e = denseFloats(a)
            var n = e.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
            if n == 0 { n = 1 }
            for k in 0..<e.count { e[k] /= n }
            embs.append(e)
        }
        return embs
    }

    /// Whether `model` gives a finite, non-zero embedding for a plain test crop.
    static func works(_ model: MLModel) -> Bool {
        let img = BGRAImage(width: 64, height: 128)
        let v = img.view
        for y in 0..<v.height {
            let row = v.row(y)
            for x in 0..<v.width { row[4 * x] = UInt8(2 * x); row[4 * x + 1] = UInt8(y); row[4 * x + 2] = UInt8(255 - y); row[4 * x + 3] = 255 }
        }
        guard let e = try? Embedder(model: model).embed([img], pool: BufferPool()).first else { return false }
        return e.allSatisfy(\.isFinite) && e.contains { $0 != 0 }
    }
}
