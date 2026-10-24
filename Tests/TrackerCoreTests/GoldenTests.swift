import Foundation
import Testing
@testable import TrackerCore

// Golden replay of the lab tracker (tools/ios_golden/tracker_golden.py): tag() on the tag window,
// then camera -> predictedBox -> [referee actions] -> step per tick, on cached FULL detections of
// four real segments (four configs each) and on twelve synthetic scenes for the rare paths (four
// configs each, six for the stacked look-alike). The fixture inputs are exact (float32 boxes, float16 features and embeddings), so
// the port must reproduce the lab: the same state, chosen candidate, source, flags and profile sizes
// every tick, boxes within 1e-3 px. Measured: bit-exact, every number on every tick.

struct GoldenCase {
    struct NamedConfig: Decodable {
        let name: String
        let config: TrackerConfig
    }

    struct Decoded: Decodable {
        let configs: [NamedConfig]
        let trustS: Double
    }

    let name: String
    let meta: [String: Any]
    let configs: [NamedConfig]
    let trustS: Double
    let data: Data

    /// The fixtures here (synthetic scenes), then any in $DIRECTE_GOLDEN: cases recorded on real footage, which stay
    /// with whoever filmed it (tools/golden/tracker_golden.py writes both).
    static var dirs: [URL] {
        let here = Bundle.module.url(forResource: "Fixtures", withExtension: nil)!
        guard let extra = ProcessInfo.processInfo.environment["DIRECTE_GOLDEN"], !extra.isEmpty else { return [here] }
        return [here, URL(fileURLWithPath: extra)]
    }

    /// Where a fixture file is.
    static func url(_ file: String) -> URL? {
        dirs.map { $0.appendingPathComponent(file) }.first { FileManager.default.fileExists(atPath: $0.path) }
    }

    /// The cases among `names` that are here.
    static func available(_ names: [String]) -> [String] { names.filter { url("\($0).json") != nil } }

    /// Every fixture case and its configs.
    static let all: [(String, String)] = {
        let files = dirs.flatMap { (try? FileManager.default.contentsOfDirectory(atPath: $0.path)) ?? [] }
        return files.filter { $0.hasPrefix("tracker_") && $0.hasSuffix(".json") }.sorted().flatMap { f -> [(String, String)] in
            let name = String(f.dropLast(5))
            let c = try! GoldenCase(name)
            return c.configs.map { (name, $0.name) }
        }
    }()

    static var names: [String] { Array(Set(all.map(\.0))).sorted() }

    init(_ name: String) throws {
        self.name = name
        guard let jsonURL = Self.url("\(name).json"), let binURL = Self.url("\(name).bin") else {
            throw CocoaError(.fileNoSuchFile, userInfo: [NSFilePathErrorKey: name])
        }
        let json = try Data(contentsOf: jsonURL)
        meta = try JSONSerialization.jsonObject(with: json) as! [String: Any]
        // configs through JSONDecoder: its number parsing round-trips doubles (JSONSerialization's doesn't)
        let dec = JSONDecoder()
        dec.keyDecodingStrategy = .convertFromSnakeCase
        let decoded = try dec.decode(Decoded.self, from: json)
        configs = decoded.configs
        trustS = decoded.trustS
        let packed = try Data(contentsOf: binURL)
        data = try (packed as NSData).decompressed(using: .zlib) as Data  // raw DEFLATE
        precondition(data.count == meta["raw_bytes"] as! Int)
    }

    private func entry(_ key: String) -> (dtype: String, shape: [Int], offset: Int) {
        let a = (meta["arrays"] as! [String: [String: Any]])[key]!
        return (a["dtype"] as! String, a["shape"] as! [Int], a["offset"] as! Int)
    }

    func shape(_ key: String) -> [Int] { entry(key).shape }

    private func raw<T>(_ key: String, _ type: T.Type, _ dtype: String) -> [T] {
        let e = entry(key)
        precondition(e.dtype == dtype, "\(key): \(e.dtype), not \(dtype)")
        let n = e.shape.reduce(1, *)
        return data.withUnsafeBytes { buf in
            (0..<n).map { buf.loadUnaligned(fromByteOffset: e.offset + $0 * MemoryLayout<T>.size, as: T.self) }
        }
    }

    func f64(_ key: String) -> [Double] { raw(key, Double.self, "f8") }
    func f32(_ key: String) -> [Float] { raw(key, Float.self, "f4") }
    func f16(_ key: String) -> [Float] { raw(key, UInt16.self, "f2").map(halfToFloat) }
    func u8(_ key: String) -> [UInt8] { raw(key, UInt8.self, "u1") }
    func i16(_ key: String) -> [Int] { raw(key, Int16.self, "i2").map(Int.init) }
    func i32(_ key: String) -> [Int] { raw(key, Int32.self, "i4").map(Int.init) }

    var windowTicks: Int { meta["window_ticks"] as! Int }
    var height: Double { Double(meta["height"] as! Int) }
    var tap: SIMD2<Double> { let p = f64("tap"); return SIMD2(p[0], p[1]) }

    /// Candidates per tick, as Perception would hand them over.
    func ticks() -> [(t: Double, candidates: [Candidate])] {
        let boxes = f32("boxes"), feats = f16("feats"), valids = u8("valids"), embs = f16("embs")
        let offsets = i32("offsets"), ts = f64("t")
        let R = AppearanceSpec.regions.count, D = AppearanceSpec.dim, E = shape("embs")[1]
        return ts.indices.map { n in
            let cs = (offsets[n]..<offsets[n + 1]).map { i in
                Candidate(box: Box(Double(boxes[4 * i]), Double(boxes[4 * i + 1]), Double(boxes[4 * i + 2]), Double(boxes[4 * i + 3])),
                          feats: (0..<R).map { r in Array(feats[(i * R + r) * D ..< (i * R + r + 1) * D]) },
                          valids: (0..<R).map { valids[i * R + $0] != 0 },
                          emb: Array(embs[i * E ..< (i + 1) * E]))
            }
            return (ts[n], cs)
        }
    }

    func config(_ name: String) -> TrackerConfig { configs.first { $0.name == name }!.config }
}

/// IEEE half -> float (exact).
func halfToFloat(_ h: UInt16) -> Float {
    let sign = UInt32(h >> 15) << 31, exp = Int((h >> 10) & 0x1f), frac = UInt32(h & 0x3ff)
    if exp == 0 {
        return Float(sign: sign != 0 ? .minus : .plus, exponent: -24, significand: Float(frac))  // zero / subnormal
    }
    if exp == 31 { return Float(bitPattern: sign | 0x7f80_0000 | (frac << 13)) }
    return Float(bitPattern: sign | UInt32(exp - 15 + 127) << 23 | (frac << 13))
}

/// tracker_golden.py `spotter_of`: the same people seen by a second camera, 3 px right and down.
func spotterOf(_ cands: [Candidate]) -> [Candidate] {
    cands.map { var c = $0; c.box = c.box.offset(SIMD2(3, 3)); return c }
}

@Test(arguments: GoldenCase.names)
func goldenTag(name: String) throws {
    let g = try GoldenCase(name)
    let (profile, box, k) = try tag(window: Array(g.ticks()[..<g.windowTicks]), point: g.tap, frameHeight: g.height)
    #expect(k == g.meta["tag_index"] as! Int)
    #expect([box.x0, box.y0, box.x1, box.y1] == g.f64("tag/box"))
    #expect(profile.feat == g.f32("tag/feat"))
    #expect(profile.valid == g.u8("tag/valid").map { $0 != 0 })
    #expect(profile.emb == g.f32("tag/emb"))
}

struct Deviation: CustomStringConvertible {
    var box = 0.0, pred = 0.0, vel = 0.0, app = 0.0, score = 0.0, ground = 0.0, appLevel = 0.0
    var description: String {
        String(format: "box %.2g px, pred %.2g px, vel %.2g px/s, app %.2g, score %.2g, ground %.2g, appLevel %.2g",
               box, pred, vel, app, score, ground, appLevel)
    }
}

@Test(arguments: GoldenCase.all)
func goldenReplay(name: String, configName: String) throws {
    let g = try GoldenCase(name)
    let cfg = g.config(configName)
    let ticks = g.ticks()
    let nWin = g.windowTicks
    let spotter = g.meta["spotter"] as! Bool, trustS = g.trustS
    let shift = g.f64("shift"), conf = g.f64("conf")
    let p = configName + "/"
    let state = g.u8(p + "state"), chosen = g.i16(p + "chosen"), flags = g.u8(p + "flags"), source = g.u8(p + "source")
    let box = g.f64(p + "box"), pred = g.f64(p + "pred"), vel = g.f64(p + "vel")
    let app = g.f64(p + "app"), score = g.f64(p + "score"), ground = g.f64(p + "ground"), appLevel = g.f64(p + "app_level")
    let counts = g.i16(p + "counts"), act = g.i16(p + "act")
    let codes: [TrackState] = [.locked, .uncertain, .lost]

    let (profile, box0, k) = try tag(window: Array(ticks[..<nWin]), point: g.tap, frameHeight: g.height)
    let trk = TargetTracker(profile: profile, box: box0, t: ticks[k].t, config: cfg, frameHeight: g.height)
    var dev = Deviation(), firstMismatch: String?, mismatches = 0
    for (n, tick) in ticks.enumerated() {
        let cands = tick.candidates
        if n >= nWin {
            trk.camera(shift: SIMD2(shift[2 * n], shift[2 * n + 1]), conf: conf[n])
            let pb = trk.predictedBox(at: tick.t)
            dev.pred = max(dev.pred, zip([pb.x0, pb.y0, pb.x1, pb.y1], pred[4 * n ..< 4 * n + 4]).map { abs($0 - $1) }.max()!)
            // referee actions (tracker_golden.py `apply`), recorded as (veto, force, flags)
            let (veto, force, fl) = (act[3 * n], act[3 * n + 1], act[3 * n + 2])
            if veto >= 0 || force >= 0 || fl != 0 || trk.exclude != nil || !trk.allowReacquire {
                trk.exclude = veto < 0 ? nil : cands.indices.map { $0 == veto }
                trk.allowReacquire = fl & 2 == 0
                if fl & 1 != 0 { trk.release(t: tick.t) }
                if force >= 0 { trk.force(box: cands[force].box, t: tick.t, trustS: trustS) }
            }
        }
        let s = trk.step(t: tick.t, view: cands, spotter: spotter ? spotterOf(cands) : nil)
        let f = (trk.jumpT == tick.t ? 1 : 0) | (trk.unstable ? 2 : 0) | (trk.latched ? 4 : 0)
            | (trk.tiein ? 8 : 0) | (trk.shiftUnknown ? 16 : 0)
        let pr = trk.profile
        let got = [pr.gallery.count, pr.negs.count, pr.pending.count, trk.people.count]
        let want = Array(counts[4 * n ..< 4 * n + 4])
        let same = s.state == codes[Int(state[n])] && s.chosen == chosen[n] && f == Int(flags[n]) && got == want
            && (s.source == .spotter) == (source[n] != 0)
        if !same {
            mismatches += 1
            if firstMismatch == nil {
                firstMismatch = "tick \(n) t=\(tick.t): got \(s.state) chosen \(s.chosen) \(s.source) flags \(f) counts \(got), "
                    + "lab \(codes[Int(state[n])]) chosen \(chosen[n]) spotter \(source[n]) flags \(flags[n]) counts \(want)"
            }
        }
        dev.box = max(dev.box, zip([s.box.x0, s.box.y0, s.box.x1, s.box.y1], box[4 * n ..< 4 * n + 4]).map { abs($0 - $1) }.max()!)
        dev.vel = max(dev.vel, abs(s.velocity.x - vel[2 * n]), abs(s.velocity.y - vel[2 * n + 1]))
        dev.app = max(dev.app, abs(s.app - app[n]))
        dev.score = max(dev.score, abs(s.score - score[n]))
        dev.ground = max(dev.ground, abs(trk.ground - ground[n]))
        dev.appLevel = max(dev.appLevel, abs(trk.appLevel - appLevel[n]))
    }
    let results = (g.meta["results"] as! [String: [String: Any]])[configName]!
    let events = (results["events"] as! [[Any]]).map { ($0[0] as! Double, $0[1] as! String) }
    print("[golden] \(name) \(configName): \(ticks.count) ticks, \(mismatches) mismatches, \(trk.events.count) events; max deviation \(dev)")
    #expect(mismatches == 0, "\(firstMismatch ?? "")")
    #expect(dev.box < 1e-3 && dev.pred < 1e-3, "\(dev)")
    #expect(dev.vel < 1e-2 && dev.app < 1e-6 && dev.score < 1e-6 && dev.ground < 1e-3 && dev.appLevel < 1e-6, "\(dev)")
    #expect(trk.events.map(\.kind) == events.map(\.1))
    #expect(zip(trk.events, events).allSatisfy { abs($0.t - $1.0) < 1e-9 })
}

@Test func configMatchesLab() throws {
    // every lab Config field is ported (same snake_case name) with the same default
    let g = try GoldenCase("tracker_synth_tiny")  // a synthetic scene: its default config is Config()
    let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: GoldenCase.url("tracker_synth_tiny.json")!)) as! [String: Any]
    let lab = ((raw["configs"] as! [[String: Any]]).first { $0["name"] as! String == "default" }!)["config"] as! [String: Any]
    let enc = JSONEncoder()
    enc.keyEncodingStrategy = .convertToSnakeCase
    let ours = try JSONSerialization.jsonObject(with: enc.encode(TrackerConfig())) as! [String: Any]
    // Swift-only: directe_tracking/tracker.py's literals (0.1, 0.9, 0.92), fields so `recalibrate` moves them with reid_lo/hi
    let swiftOnly: Set = ["gallery_admit", "gallery_new", "neg_dedupe"]
    #expect(Set(ours.keys).subtracting(swiftOnly) == Set(lab.keys))
    #expect(g.config("default") == TrackerConfig())
    #expect(g.config("combo") == TrackerConfig.combo)
    // the scripted and plain-crop configs are recorded on a real, fixed-camera clip ($DIRECTE_GOLDEN)
    if let real = try? GoldenCase("tracker_IMG_4102_40") {
        #expect(real.config("combo_scripted") == TrackerConfig.combo)
        #expect(real.config("combo_plaincal") == TrackerConfig.comboPlain)
    }
}

@Test(arguments: GoldenCase.available(["tracker_IMG_4102_40", "tracker_IMG_2891_40"]))
func goldenStart(name: String) throws {
    // TargetTracker.start (tag + init + window replay) is the golden replay's tag phase
    let g = try GoldenCase(name)
    let ticks = g.ticks(), nWin = g.windowTicks
    let (trk, steps, k) = try TargetTracker.start(window: Array(ticks[..<nWin]), point: g.tap, frameHeight: g.height,
                                                  config: g.config("combo"))
    #expect(k == g.meta["tag_index"] as! Int && steps.count == nWin)
    let state = g.u8("combo/state"), box = g.f64("combo/box")
    let codes: [TrackState] = [.locked, .uncertain, .lost]
    for (n, s) in steps.enumerated() {
        #expect(s.state == codes[Int(state[n])])
        #expect([s.box.x0, s.box.y0, s.box.x1, s.box.y1] == Array(box[4 * n ..< 4 * n + 4]))
    }
    #expect(trk.t == ticks[nWin - 1].t)
}
