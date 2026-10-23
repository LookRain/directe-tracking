import Foundation

/// The candidates of one frame, flat (lab tracker `Cands`). Boxes are float32-valued like the lab's.
struct Cands {
    static let R = AppearanceSpec.regions.count, D = AppearanceSpec.dim, FD = R * D

    var boxes: [Box]
    var feats: [Float]     // count x R x D
    var valids: [Bool]     // count x R
    var embs: [Float]?     // count x e, nil when any candidate has no embedding
    var e: Int

    var count: Int { boxes.count }

    init(boxes: [Box], feats: [Float], valids: [Bool], embs: [Float]?, e: Int) {
        self.boxes = boxes; self.feats = feats; self.valids = valids; self.embs = embs; self.e = e
    }

    init(_ cands: [Candidate]) {
        boxes = cands.map { $0.box.float32 }
        feats = []; feats.reserveCapacity(cands.count * Self.FD)
        valids = []; valids.reserveCapacity(cands.count * Self.R)
        for c in cands {
            precondition(c.feats.count == Self.R && c.feats.allSatisfy { $0.count == Self.D } && c.valids.count == Self.R,
                         "Candidate.feats must be \(Self.R) x \(Self.D), valids \(Self.R)")
            for f in c.feats { feats += f }
            valids += c.valids
        }
        let e = cands.first?.emb?.count ?? 0
        if cands.allSatisfy({ $0.emb?.count == e && e > 0 }) {  // (an empty frame: [], like the lab's [0, E])
            embs = cands.flatMap { $0.emb! }
            self.e = e
        } else {
            embs = nil
            self.e = 0
        }
    }

    func feat(_ i: Int) -> [Float] { Array(feats[i * Self.FD ..< (i + 1) * Self.FD]) }
    func valid(_ i: Int) -> [Bool] { Array(valids[i * Self.R ..< (i + 1) * Self.R]) }
    func emb(_ i: Int) -> ArraySlice<Float> { embs![i * e ..< (i + 1) * e] }

    func subset(_ k: Int) -> Cands {
        Cands(boxes: [boxes[k]], feats: feat(k), valids: valid(k), embs: embs.map { _ in Array(emb(k)) }, e: e)
    }

    func concat(_ o: Cands) -> Cands {
        var embs: [Float]? = nil
        if let a = self.embs, let b = o.embs { embs = a + b }
        return Cands(boxes: boxes + o.boxes, feats: feats + o.feats, valids: valids + o.valids, embs: embs,
                     e: max(e, o.e))
    }

    /// Centres (float32) and heights (float32, at least 1) as the lab's `_geometry`.
    func geometry() -> (cents: [SIMD2<Double>], hs: [Double]) {
        (boxes.map { SIMD2(f32($0.x0 + $0.x1) / 2, f32($0.y0 + $0.y1) / 2) },
         boxes.map { max(f32($0.y1 - $0.y0), 1) })
    }
}

/// Who the climber is (lab tracker `Profile`): colour signatures from the tag moment plus a copy
/// that slowly follows the light, a re-ID anchor plus a gallery of other viewpoints, and (with
/// `TrackerConfig.negatives`) a bank of people who aren't the climber. With `transact`, updates
/// wait in `pending` until they survive `commitS` without a doubt.
public struct Profile: Sendable {
    /// A profile update waiting to be committed.
    public enum Update: Sendable {
        case colour(feat: [Float], valid: [Bool])
        case gallery([Float])
        case negative(emb: [Float], feat: [Float], valid: [Bool])
    }

    /// Colour, from the tag moment; never changes. regions x dim, flat.
    public var feat: [Float]
    public var valid: [Bool]
    /// Re-ID anchor from the tag moment.
    public var emb: [Float]?
    /// Colour copy that slowly follows lighting/viewpoint.
    public var adapt: [Float]
    public var avalid: [Bool]
    /// Re-ID embeddings of other viewpoints.
    public var gallery: [[Float]] = []
    public var lastGalleryT = -1e9
    /// Re-ID embeddings of people who aren't the climber, and their colour.
    public var negs: [[Float]] = []
    public var negF: [[Float]] = []
    public var negV: [[Bool]] = []
    public var pending: [(t: Double, update: Update)] = []
    public var lastNegT = -1e9

    public init(feat: [Float], valid: [Bool], emb: [Float]? = nil) {
        precondition(feat.count == Cands.FD && valid.count == Cands.R)
        self.feat = feat; self.valid = valid; self.emb = emb
        adapt = feat; avalid = valid
    }

    /// `feat` as regions x dim.
    public init(feats: [[Float]], valid: [Bool], emb: [Float]? = nil) {
        self.init(feat: feats.flatMap { $0 }, valid: valid, emb: emb)
    }

    // MARK: similarity

    func colour(_ c: Cands, _ cfg: TrackerConfig) -> Scores {
        if cfg.negatives && cfg.negColour && !negF.isEmpty {
            // A region a known non-climber shares with the profile (the belayer's blue shorts too)
            // stops counting much.
            var wm = [Double](repeating: 1, count: Cands.R)
            for (prof, pv) in [(feat, valid), (adapt, avalid)] {
                for r in 0..<Cands.R {
                    var mx: Float = 0
                    for k in negF.indices {
                        let b: Float = negV[k][r] && pv[r] ? Appearance.bc(negF[k], 0, prof, r) : 0
                        mx = k == 0 ? b : max(mx, b)
                    }
                    let q = min(max((Float(0.95 as Double) - mx) / Float(0.35 as Double), Float(0.15 as Double)), 1)
                    wm[r] = min(wm[r], Double(q))
                }
            }
            let a = Appearance.simWeighted(profile: feat, pvalid: valid, feats: c.feats, valids: c.valids, wmul: wm)
            let b = Appearance.simWeighted(profile: adapt, pvalid: avalid, feats: c.feats, valids: c.valids, wmul: wm)
            return Scores(v: zip(a, b).map { max($0, $1) }, f32: false)
        }
        let a = Appearance.similarityMany(profile: feat, pvalid: valid, feats: c.feats, valids: c.valids)
        let b = Appearance.similarityMany(profile: adapt, pvalid: avalid, feats: c.feats, valids: c.valids)
        return Scores(v: zip(a, b).map { Double(max($0, $1)) }, f32: true)
    }

    func posCos(_ c: Cands) -> [Float]? {
        guard let emb, let embs = c.embs, c.count > 0 else { return nil }
        var cos = matvec(embs, rows: c.count, emb)
        if !gallery.isEmpty {
            let g = matmulT(embs, m: c.count, gallery.flatMap { $0 }, n: gallery.count, e: c.e)
            for i in 0..<c.count {
                var m = g[i * gallery.count]
                for j in 1..<gallery.count { m = max(m, g[i * gallery.count + j]) }
                cos[i] = max(cos[i], m - Float(0.03 as Double))  // gallery views count, slightly less than the anchor
            }
        }
        return cos
    }

    func negCos(_ c: Cands) -> [Float]? {
        guard !negs.isEmpty, let embs = c.embs, c.count > 0 else { return nil }
        let g = matmulT(embs, m: c.count, negs.flatMap { $0 }, n: negs.count, e: c.e)
        return (0..<c.count).map { i in
            var m = g[i * negs.count]
            for j in 1..<negs.count { m = max(m, g[i * negs.count + j]) }
            return m
        }
    }

    func reidCos(_ c: Cands, _ cfg: TrackerConfig) -> [Float]? {
        guard let pos = posCos(c) else { return nil }
        guard cfg.negatives, let neg = negCos(c) else { return pos }
        let lam = Float(cfg.negLam), margin = Float(cfg.negMargin)
        return zip(pos, neg).map { p, n in p - lam * max(0, n - p + margin) }  // nearer a non-climber: penalise
    }

    /// Appearance per candidate in [0, 1], or unknown (-1).
    func sim(_ c: Cands, _ cfg: TrackerConfig) -> Scores {
        let colour = cfg.useColour ? self.colour(c, cfg)
            : Scores(v: [Double](repeating: Double(AppearanceSpec.unknown), count: c.count), f32: false)
        guard cfg.useReid, let cos = reidCos(c, cfg) else { return colour }
        let lo = Float(cfg.reidLo), span = Float(cfg.reidHi - cfg.reidLo)
        let r = cos.map { min(max(($0 - lo) / span, 0), 1) }
        if cfg.reidMode == .fallback {
            return Scores(v: c.boxes.indices.map { colour.v[$0] >= 0 ? colour.v[$0] : Double(r[$0]) }, f32: colour.f32)
        }
        let w = cfg.colourWeight
        if colour.f32 {
            let w32 = Float(w), u32 = Float(1 - w)
            return Scores(v: c.boxes.indices.map { i in
                colour.v[i] >= 0 ? Double(w32 * Float(colour.v[i]) + u32 * r[i]) : Double(r[i])
            }, f32: true)
        }
        let u32 = Float(1 - w)
        return Scores(v: c.boxes.indices.map { i in
            colour.v[i] >= 0 ? w * colour.v[i] + Double(u32 * r[i]) : Double(r[i])
        }, f32: false)
    }

    /// Public similarity of each candidate to this profile (lab `Profile.sim`).
    public func similarity(_ candidates: [Candidate], config: TrackerConfig) -> [Double] {
        sim(Cands(candidates), config).v
    }

    // MARK: learning

    mutating func learn(_ c: Cands, _ k: Int, _ t: Double, _ cfg: TrackerConfig) {
        let f = c.feat(k), v = c.valid(k)
        if cfg.transact {
            pending.append((t, .colour(feat: f, valid: v)))
        } else {
            adaptColour(f, v, cfg)
        }
        guard c.embs != nil, let anchor = emb, !(t - lastGalleryT < cfg.galleryEveryS) else { return }
        let e = c.emb(k)
        if Double(dot(e, anchor)) < cfg.reidLo + cfg.galleryAdmit { return }  // too far from the anchor to trust as the same person
        if !gallery.isEmpty, let m = matvec(gallery.flatMap { $0 }, rows: gallery.count, Array(e)).max(), Double(m) > cfg.galleryNew {
            return  // nothing new
        }
        lastGalleryT = t
        if cfg.transact {
            pending.append((t, .gallery(Array(e))))
        } else {
            addGallery(Array(e), cfg)
        }
    }

    mutating func adaptColour(_ f: [Float], _ v: [Bool], _ cfg: TrackerConfig) {
        let keep = Float(1 - cfg.adaptRate), rate = Float(cfg.adaptRate)
        for r in 0..<Cands.R where v[r] {
            if avalid[r] {
                for k in r * Cands.D ..< (r + 1) * Cands.D { adapt[k] = keep * adapt[k] + rate * f[k] }
            } else {
                for k in r * Cands.D ..< (r + 1) * Cands.D { adapt[k] = f[k] }
                avalid[r] = true
            }
        }
    }

    mutating func addGallery(_ e: [Float], _ cfg: TrackerConfig) {
        gallery.append(e)
        if gallery.count > cfg.gallerySize { gallery.removeFirst() }
    }

    mutating func addNegative(_ e: [Float], _ f: [Float], _ v: [Bool], _ t: Double, _ cfg: TrackerConfig) {
        if !negs.isEmpty, let m = matvec(negs.flatMap { $0 }, rows: negs.count, e).max(), Double(m) > cfg.negDedupe {
            return  // nothing new
        }
        if cfg.transact {
            pending.append((t, .negative(emb: e, feat: f, valid: v)))
        } else {
            commitNegative(e, f, v, cfg)
        }
    }

    mutating func commitNegative(_ e: [Float], _ f: [Float], _ v: [Bool], _ cfg: TrackerConfig) {
        negs.append(e); negF.append(f); negV.append(v)
        if negs.count > cfg.negSize {
            negs.removeFirst(); negF.removeFirst(); negV.removeFirst()
        }
    }

    /// Updates that survived `commitS` without a doubt become part of the profile.
    mutating func commitDue(_ t: Double, _ cfg: TrackerConfig) {
        var keep: [(t: Double, update: Update)] = []
        for item in pending {
            if t - item.t < cfg.commitS {
                keep.append(item)
                continue
            }
            switch item.update {
            case let .colour(f, v): adaptColour(f, v, cfg)
            case let .gallery(e): addGallery(e, cfg)
            case let .negative(e, f, v): commitNegative(e, f, v, cfg)
            }
        }
        pending = keep
    }

    /// Something suggests the lock may have been wrong lately: forget what it taught us.
    mutating func doubt() { pending = [] }
}

// MARK: - Appearance state (export / import)

/// A target's appearance, as a value that can be saved and given to a later tracker: the colour signature by body
/// region (the tag-time one and the adapted copy), the re-ID anchor and gallery, and the bank of people who aren't
/// the target. Updates still waiting to be committed aren't part of it (they hadn't survived their wait yet).
public struct AppearanceState: Codable, Sendable, Hashable {
    /// regions x dim, flat (`AppearanceSpec`).
    public var feat: [Float]
    public var valid: [Bool]
    public var adapt: [Float]
    public var avalid: [Bool]
    public var emb: [Float]?
    public var gallery: [[Float]]
    public var negatives: [[Float]]
    public var negativeFeats: [[Float]]
    public var negativeValid: [[Bool]]

    public init(_ p: Profile) {
        feat = p.feat; valid = p.valid; adapt = p.adapt; avalid = p.avalid; emb = p.emb
        gallery = p.gallery; negatives = p.negs; negativeFeats = p.negF; negativeValid = p.negV
    }

    /// The profile again (no pending updates; the gallery and negative timers start over).
    public var profile: Profile {
        var p = Profile(feat: feat, valid: valid, emb: emb)
        p.adapt = adapt; p.avalid = avalid
        p.gallery = gallery
        p.negs = negatives; p.negF = negativeFeats; p.negV = negativeValid
        return p
    }
}

extension Profile {
    /// This profile (fresh from a tag) seeded with an earlier one of the same target, for tracking again from a
    /// later moment: the fresh tag stays the anchor (the target may have changed clothes or light), and the earlier
    /// one adds what it knows that the fresh one agrees with. Only if the earlier anchor passes for the fresh one
    /// (re-ID cosine at least `reidLo + galleryAdmit`, the gallery's own admission bar) does it add anything:
    ///  - its gallery views and its anchor that pass the same bar, de-duplicated (`galleryNew`), oldest first, so the
    ///    fresh tag's views are the newest and last longest;
    ///  - its bank of other people, except anyone who passes for the fresh anchor (`reidLo` or more);
    ///  - its adapted colour, blended in at `priorColour` where both saw the region (the fresh one keeps the rest).
    /// Without re-ID embeddings on both sides, or when they disagree, the fresh profile is returned unchanged.
    public func reusing(_ prior: Profile, config cfg: TrackerConfig, priorColour: Float = 0.3) -> Profile {
        guard let anchor = emb, let old = prior.emb, old.count == anchor.count else { return self }
        let admit = cfg.reidLo + cfg.galleryAdmit
        guard Double(dot(old[...], anchor)) >= admit else { return self }
        var out = self
        var views: [[Float]] = []
        for e in prior.gallery + [old] where e.count == anchor.count && Double(dot(e[...], anchor)) >= admit {
            if views.contains(where: { Double(dot($0[...], e)) > cfg.galleryNew }) { continue }
            views.append(e)
        }
        let room = max(cfg.gallerySize - gallery.count, 0)
        out.gallery = Array(views.suffix(room)) + gallery
        for i in prior.negs.indices where prior.negs[i].count == anchor.count && Double(dot(prior.negs[i][...], anchor)) < cfg.reidLo {
            out.negs.append(prior.negs[i]); out.negF.append(prior.negF[i]); out.negV.append(prior.negV[i])
        }
        if out.negs.count > cfg.negSize {
            let drop = out.negs.count - cfg.negSize
            out.negs.removeFirst(drop); out.negF.removeFirst(drop); out.negV.removeFirst(drop)
        }
        let w = min(max(priorColour, 0), 1)
        for r in 0..<Cands.R where out.avalid[r] && prior.avalid[r] {
            for k in r * Cands.D ..< (r + 1) * Cands.D { out.adapt[k] = (1 - w) * out.adapt[k] + w * prior.adapt[k] }
        }
        return out
    }
}
