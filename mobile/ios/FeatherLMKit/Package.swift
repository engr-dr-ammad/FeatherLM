// swift-tools-version:5.9
// FeatherLMKit: on-device FeatherLM inference for iOS via ONNX Runtime.
// Developed by Ammad Aslam with AI-assisted tooling. UNTESTED: requires Xcode on macOS.
import PackageDescription

let package = Package(
    name: "FeatherLMKit",
    platforms: [.iOS(.v15), .macOS(.v13)],
    products: [.library(name: "FeatherLMKit", targets: ["FeatherLMKit"])],
    dependencies: [
        .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager", from: "1.20.0"),
    ],
    targets: [
        .target(name: "FeatherLMKit", dependencies: [
            .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager"),
        ]),
        .testTarget(name: "FeatherLMKitTests", dependencies: ["FeatherLMKit"]),
    ]
)
