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
    /// Lost, a sighting near where the climber was last seen (within `reacquireNear` of their height): the sightings
    /// may be up to this far apart (s) and still count towards `reacquireFrames`, instead of in consecutive looks
    /// (0: consecutive only, as the lab). A small climber is seen only every few looks, so in a row never came.
    public var reacquireWindow = 0.0
    public var reacquireNear = 1.0
    /// The velocity isn't learnt from looks closer together than this (s; 0: always, as the lab). A first look at the
    /// tag's own moment (dt clamped to 1 ms) read 7 px as 1,000 px/s, and the prediction slid 600 px down the wall away
    /// from a climber resting at the top (30 Oct recording, 565 s): lost 92 s, then locked onto someone else.
    public var minUpdateDt = 0.0
    /// A small climber (shorter than `stillSmallBelow` of the frame's height) who was barely moving (under 0.2 of
    /// their height a second) when the detector stopped seeing them stays uncertain, not lost, for this long (s; 0:
    /// off, as the lab), so they're matched again as themselves when seen. Crouched at the top of the wall, folded
    /// among the holds, a climber isn't seen as a person for 3-5 s at a time (30 Oct recording, 622 s and 635 s).
    public var stillUncertainS = 0.0
    public var stillSmallBelow = 0.06
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
    // Two people stay two people: look-alikes on neighbouring routes, seen from a steep side or from below, stack
    // up with touching boxes and the same look. Someone else (seen beside the locked climber for settleS, or anyone
    // in view while the track the lock has held for settleS is) may take the lock, touching or not, only by beating
    // the climber's own track by switchMargin, and not while that track is unseen or ruled out; people the detector
    // misses stay remembered for trackKeepS.
    public var switchMargin = 0.1     // score; < 0: off
    public var trackKeepS = 1.0       // remember someone the detector missed this long (0: last frame only)
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
    // A doubt drops only the updates of its last doubtReach seconds (< 0: all of them). In a crowd the doubts come
    // every few seconds (someone crossing in front, a look-alike nearby), and dropping everything pending meant
    // nothing was ever committed: 17 s into a busy gym clip the profile had learnt no other people and no new views,
    // and a belayer in a similar top took the lock while the climber was hidden (owner's report, 2 Nov).
    public var doubtReach = -1.0
    public var doubtRival = 0.05      // a rival this close is a doubt
    // One person, one track: this frame's people take over last frame's tracks, and those of people missed lately,
    // one to one, best overlap first. Each taking its own best overlap let two people inherit one track: a climber
    // coming out from behind someone who crossed in front of them became that person, "seen beside the climber".
    public var oneToOne = true
    // Two people stay two people through a loss too: while lost, nobody seen beside the locked climber for settleS
    // is re-acquired.
    public var keepApart = true
    // The lock's own track keeps it through an appearance dip: held on that one-to-one track for settleS, with nobody
    // within ownGap body heights of it for ownAloneS and where the climber was predicted (motion >= ownMotion), a
    // candidate needs only ownKeep instead of the relative floor, and doesn't move appLevel. On in `combo`.
    public var ownKeep = -1.0         // appearance floor for the lock's own track while nobody is near (< 0: off)
    public var ownAloneS = 1.0
    public var ownGap = 0.5           // body heights
    public var ownMotion = 0.6
    // Everyone else's short position track (TargetTracker `Bystanders`): on confident frames everyone apart from the
    // climber earns "not her" by the second; anyone who is one of those tracks carrying on with `bystanderS` of it is
    // ruled out, whatever they look like. A track loses it near the climber's place (`bystanderNear` body heights of the
    // lock or the prediction) unless the lock is seen and they barely touch (then it's only set aside), crossing someone who
    // hasn't earned it (IoU `bystanderCross`), or picked up again after `bystanderGapS` unseen.
    public var bystanders = false
    public var bystanderS = 1.0
    public var bystanderKeepS = 2.0     // a track unseen this long ends
    public var bystanderIoU = 0.2       // a person takes over a track overlapping where it was heading by more than this
    public var bystanderNear = 0.0
    public var bystanderCross = 0.3     // boxes overlapping this much (IoU) are a crossing: the label goes
    public var bystanderGapS = 0.5
    public var bystanderMatch = true    // ...also while locked or uncertain (false: only while lost)

    public init() {}

    /// directe_tracking/online.py VARIANTS["combo"], tracker fields (the runner's lost_focus, second_chance,
    /// motion_confirm and ground_handheld are Pipeline options).
    public static var combo: TrackerConfig {
        var c = TrackerConfig()
        c.learnGate = true; c.fallSize = true; c.negatives = true; c.transact = true
        c.groundLatch = true; c.tieinS = 60; c.ownKeep = 0.4
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
        try v(.reacquireWindow, &reacquireWindow); try v(.reacquireNear, &reacquireNear)
        try v(.minUpdateDt, &minUpdateDt); try v(.stillUncertainS, &stillUncertainS); try v(.stillSmallBelow, &stillSmallBelow)
        try v(.uncertainS, &uncertainS); try v(.adaptRate, &adaptRate); try v(.adaptApp, &adaptApp)
        try v(.useColour, &useColour); try v(.useReid, &useReid); try v(.colourWeight, &colourWeight)
        try v(.reidMode, &reidMode); try v(.reidLo, &reidLo); try v(.reidHi, &reidHi)
        try v(.gallerySize, &gallerySize); try v(.galleryEveryS, &galleryEveryS); try v(.maxSigma, &maxSigma)
        try v(.groundRule, &groundRule); try v(.maxSizeRatio, &maxSizeRatio); try v(.jumpGuard, &jumpGuard)
        try v(.moveRate, &moveRate); try v(.fallAccel, &fallAccel); try v(.jumpMargin, &jumpMargin)
        try v(.settleS, &settleS); try v(.othersAlways, &othersAlways); try v(.unstableShift, &unstableShift)
        try v(.switchMargin, &switchMargin); try v(.trackKeepS, &trackKeepS)
        try v(.groundLatch, &groundLatch); try v(.tieinS, &tieinS); try v(.riserFeetH, &riserFeetH)
        try v(.riserHeadH, &riserHeadH); try v(.riserSize, &riserSize); try v(.riserColumn, &riserColumn)
        try v(.riserHoldS, &riserHoldS); try v(.riserMargin, &riserMargin); try v(.riserFloor, &riserFloor)
        try v(.learnGate, &learnGate); try v(.learnMaxSpeed, &learnMaxSpeed); try v(.fallSize, &fallSize)
        try v(.fallSizeRatio, &fallSizeRatio); try v(.fallSizeS, &fallSizeS); try v(.fallSpeed, &fallSpeed)
        try v(.negatives, &negatives); try v(.negLam, &negLam); try v(.negMargin, &negMargin)
        try v(.negColour, &negColour); try v(.negEveryS, &negEveryS); try v(.negSize, &negSize)
        try v(.transact, &transact); try v(.commitS, &commitS); try v(.doubtRival, &doubtRival)
        try v(.doubtReach, &doubtReach)
        try v(.galleryAdmit, &galleryAdmit); try v(.galleryNew, &galleryNew); try v(.negDedupe, &negDedupe)
        try v(.oneToOne, &oneToOne); try v(.keepApart, &keepApart)
        try v(.ownKeep, &ownKeep); try v(.ownAloneS, &ownAloneS); try v(.ownGap, &ownGap); try v(.ownMotion, &ownMotion)
        try v(.bystanders, &bystanders); try v(.bystanderS, &bystanderS); try v(.bystanderKeepS, &bystanderKeepS)
        try v(.bystanderIoU, &bystanderIoU); try v(.bystanderNear, &bystanderNear); try v(.bystanderCross, &bystanderCross); try v(.bystanderGapS, &bystanderGapS)
        try v(.bystanderMatch, &bystanderMatch)
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
