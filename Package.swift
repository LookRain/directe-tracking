// swift-tools-version: 6.0
// Directe's climber tracker, as two modules: TrackerCore (pure Swift: who the climber is, frame to frame, through
// crowds, occlusion and falls; a bit-exact port of python/directe_tracking/tracker.py) and Perception (Core ML pose,
// masks and re-ID, colour features, camera motion). `swift test` on a Mac. See README.md.
import PackageDescription

let package = Package(
    name: "directe-tracking",
    platforms: [.iOS(.v18), .macOS(.v15)],
    products: [
        .library(name: "TrackerCore", targets: ["TrackerCore"]),
        .library(name: "Perception", targets: ["Perception"]),
    ],
    targets: [
        .target(name: "TrackerCore"),
        .target(name: "Perception", dependencies: ["TrackerCore"]),
        .testTarget(name: "TrackerCoreTests", dependencies: ["TrackerCore"], resources: [.copy("Fixtures")]),
        .testTarget(name: "PerceptionTests", dependencies: ["Perception"], exclude: ["Fixtures"]),
    ],
    swiftLanguageModes: [.v5]
)
