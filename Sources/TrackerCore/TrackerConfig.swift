import Foundation

/// directe_tracking/tracker.py `Config`: every field, same defaults, camelCase names. Decodes from the lab's
/// `dataclasses.asdict(Config)` JSON with `JSONDecoder.KeyDecodingStrategy.convertFromSnakeCase`
/// (missing keys keep their defaults).
public struct TrackerConfig: Codable, Sendable, Equatable {
    public enum ReidMode: String, Codable, Sendable {
        /// Mix re-ID with colour.
        case blend
        /// Re-ID only when colour can't judge.
        case fallback
    }

    public var appMin = 0.50          // absolute appearance floor to accept while locked/uncertain
    public var appRel = 0.75          // ...and relative to how well the target has been matching
    public var accept = 0.55          // combined score to accept
    public var reacquireApp = 0.75    // appearance needed to start a re-lock while lost
    public var reacquireFrames = 3    // consecutive consistent sightings to re-lock
    public var uncertainS = 2.0       // predict-only time before declaring lost
    public var adaptRate = 0.03       // colour profile adaptation per confident frame
    public var adaptApp = 0.70
    public var useColour = true
    public var useReid = true
    public var colourWeight = 0.2     // share of colour in the combined appearance (rest: re-ID)
    public var reidMode = ReidMode.blend
    public var reidLo = 0.35          // cosine mapped to 0 ...
    public var reidHi = 0.75          // ... and to 1
    public var gallerySize = 30
    public var galleryEveryS = 1.0
    // In cosine units like reidLo/reidHi, so they move with them for another embedder (`recalibrate`).
    public var galleryAdmit = 0.1     // a view joins the gallery only if within reidLo + this of the anchor
    public var galleryNew = 0.9       // ... and less similar than this to every view already there
    public var negDedupe = 0.92       // a negative at least this similar to a banked one is nothing new
    public var maxSigma = 3.0         // widest position window (body heights) while uncertain
    public var groundRule = true      // once the climber is up the wall, nobody at ground level is them
    public var maxSizeRatio = 2.5     // a candidate this much bigger/smaller than the climber is someone else
    // Jump guard: a lock may move to a box that doesn't touch the climber's predicted box only as a
    // plausible fall, and never onto someone who was already standing/climbing elsewhere.
    public var jumpGuard = true
    public var moveRate = 1.5         // body heights/s any direction beyond touching
    public var fallAccel = 11.5       // body heights/s^2, ~2 g at 1.7 m per body height
    public var jumpMargin = 0.1       // body heights of slack
    public var settleS = 0.5          // seen this long as a separate person = someone else
    public var othersAlways = false   // ...and someone else can't take the lock even when their box touches
    public var unstableShift = 1.0    // camera moved this many body heights in a frame: freeze identity
    // Ground rule independent of the lock being right: "the climber is up" holds through a loss and is
    // released only by following them down; the ground line moves with the camera; tagged mid-wall,
    // someone standing beside the tag gives the ground.
    public var groundLatch = false
    // Tie-in phase: the climber is whoever leaves the ground (see TargetTracker.tieIn).
    public var tieinS = 0.0           // watch this long after the tag (0: off; 60 in the deep dive)
    public var riserFeetH = 1.0       // riser's feet this many own heights above the lock's feet...
    public var riserHeadH = 0.5       // ...head this many above the lock's head
    public var riserSize = 0.25       // ...at least this fraction of the lock's height
    public var riserColumn = 1.5      // ...centre within this many lock widths
    public var riserHoldS = 1.5       // ...all of it for this long
    public var riserMargin = 0.15     // appearance vs the tag-time profile: riser >= lock + margin...
    public var riserFloor = 0.5       // ...and >= floor
    // Learn the profile only from steady views (not during falls, fast moves, camera motion, or
    // right after the guard refused a jump).
    public var learnGate = false
    public var learnMaxSpeed = 1.0    // body heights/s
    // After a fall toward the camera the climber's box grows 4-6x; judge size against that.
    public var fallSize = false
    public var fallSizeRatio = 6.0    // max size ratio within fallSizeS of fast downward motion
    public var fallSizeS = 4.0
    public var fallSpeed = 2.0        // body heights/s downward = falling
    // Learn who it isn't, and make learning undoable.
    public var negatives = false      // bank other settled people seen next to a confident lock
    public var negLam = 1.0           // re-ID cos -= lam * max(0, cos(nearest negative) - cos(profile) + margin)
    public var negMargin = 0.05
    public var negColour = true       // body regions a negative shares with the profile count less
    public var negEveryS = 0.5
    public var negSize = 60
    public var transact = false       // profile updates wait commitS with no doubt; any doubt discards them
    public var commitS = 5.0
    public var doubtRival = 0.05      // a rival this close is a doubt

    public init() {}

    /// directe_tracking/online.py VARIANTS["combo"], tracker fields (the runner's lost_focus, second_chance,
    /// motion_confirm and ground_handheld are Pipeline options).
    public static var combo: TrackerConfig {
        var c = TrackerConfig()
        c.learnGate = true; c.fallSize = true; c.negatives = true; c.transact = true
        c.groundLatch = true; c.tieinS = 60
        return c
    }

    /// directe_tracking/online.py VARIANTS["combo_plain"]: `combo` with the plain-crop re-ID calibration (the
    /// Perception side embeds plain crops: `plainReID`).
    public static var comboPlain: TrackerConfig {
        var c = combo
        c.reidLo = 0.47; c.reidHi = 0.78
        return c
    }

    public init(from decoder: Decoder) throws {
        self.init()
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func v<T: Decodable>(_ key: CodingKeys, _ value: inout T) throws {
            if let x = try c.decodeIfPresent(T.self, forKey: key) { value = x }
        }
        try v(.appMin, &appMin); try v(.appRel, &appRel); try v(.accept, &accept)
        try v(.reacquireApp, &reacquireApp); try v(.reacquireFrames, &reacquireFrames)
        try v(.uncertainS, &uncertainS); try v(.adaptRate, &adaptRate); try v(.adaptApp, &adaptApp)
        try v(.useColour, &useColour); try v(.useReid, &useReid); try v(.colourWeight, &colourWeight)
        try v(.reidMode, &reidMode); try v(.reidLo, &reidLo); try v(.reidHi, &reidHi)
        try v(.gallerySize, &gallerySize); try v(.galleryEveryS, &galleryEveryS); try v(.maxSigma, &maxSigma)
        try v(.groundRule, &groundRule); try v(.maxSizeRatio, &maxSizeRatio); try v(.jumpGuard, &jumpGuard)
        try v(.moveRate, &moveRate); try v(.fallAccel, &fallAccel); try v(.jumpMargin, &jumpMargin)
        try v(.settleS, &settleS); try v(.othersAlways, &othersAlways); try v(.unstableShift, &unstableShift)
        try v(.groundLatch, &groundLatch); try v(.tieinS, &tieinS); try v(.riserFeetH, &riserFeetH)
        try v(.riserHeadH, &riserHeadH); try v(.riserSize, &riserSize); try v(.riserColumn, &riserColumn)
        try v(.riserHoldS, &riserHoldS); try v(.riserMargin, &riserMargin); try v(.riserFloor, &riserFloor)
        try v(.learnGate, &learnGate); try v(.learnMaxSpeed, &learnMaxSpeed); try v(.fallSize, &fallSize)
        try v(.fallSizeRatio, &fallSizeRatio); try v(.fallSizeS, &fallSizeS); try v(.fallSpeed, &fallSpeed)
        try v(.negatives, &negatives); try v(.negLam, &negLam); try v(.negMargin, &negMargin)
        try v(.negColour, &negColour); try v(.negEveryS, &negEveryS); try v(.negSize, &negSize)
        try v(.transact, &transact); try v(.commitS, &commitS); try v(.doubtRival, &doubtRival)
        try v(.galleryAdmit, &galleryAdmit); try v(.galleryNew, &galleryNew); try v(.negDedupe, &negDedupe)
    }

    /// The re-ID calibration for another embedder: `reidLo`/`reidHi` set to `lo`/`hi`, and the constants in cosine
    /// units (gallery admission and "nothing new", the negatives' dedupe and margin) moved to the same place relative
    /// to them. Without it an embedder with a narrower cosine band (OSNet-IBN on plain crops: 0.53-0.72, vs 0.35-0.75)
    /// admits fewer views of the climber and calls every one new (docs/review/deep-dives/reid-models.md, Calibration).
    public mutating func recalibrate(lo: Double, hi: Double) {
        let s = (hi - lo) / (reidHi - reidLo)
        func move(_ c: Double) -> Double { lo + (c - reidLo) * s }
        galleryAdmit *= s
        negMargin *= s
        galleryNew = move(galleryNew)
        negDedupe = move(negDedupe)
        reidLo = lo
        reidHi = hi
    }
}
