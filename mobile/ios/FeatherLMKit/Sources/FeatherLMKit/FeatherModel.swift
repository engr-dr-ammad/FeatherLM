import Foundation
import onnxruntime_objc

/// Runs the fixed-window ONNX export (see featherlm/export/onnx_export.py).
/// Mirrors mobile/reference_runtime.py, which is tested on desktop ONNX Runtime.
public final class FeatherModel {
    public let tokenizer: FeatherTokenizer
    private let env: ORTEnv
    private let session: ORTSession
    private let window: Int, padId: Int64, eosId: Int

    public init(modelPath: String, configURL: URL, tokenizer: FeatherTokenizer, threads: Int32 = 2) throws {
        let cfg = try JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as! [String: Any]
        window = cfg["window"] as! Int
        padId = Int64(cfg["pad_id"] as! Int)
        eosId = cfg["eos_id"] as! Int
        self.tokenizer = tokenizer
        env = try ORTEnv(loggingLevel: .warning)
        let opts = try ORTSessionOptions()
        try opts.setIntraOpNumThreads(threads)
        try opts.setGraphOptimizationLevel(.all)
        // Core ML execution provider can be appended here once the graph's ops are verified on device.
        session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: opts)
    }

    public func generate(prompt: String, maxTokens: Int = 120, temperature: Float = 0.8, topK: Int = 40,
                         isCancelled: () -> Bool = { false }, onToken: (Int) -> Void) throws {
        var ids = tokenizer.encode(prompt)
        if ids.isEmpty { ids = [1] }
        for _ in 0..<maxTokens {
            if isCancelled() { return }
            let ctx = Array(ids.suffix(window))
            var buf = [Int64](repeating: padId, count: window)
            for (i, t) in ctx.enumerated() { buf[i] = Int64(t) }
            let data = NSMutableData(bytes: &buf, length: buf.count * MemoryLayout<Int64>.size)
            let input = try ORTValue(tensorData: data, elementType: .int64, shape: [1, NSNumber(value: window)])
            let out = try session.run(withInputs: ["input_ids": input], outputNames: ["logits"], runOptions: nil)
            let raw = try out["logits"]!.tensorData() as Data
            let vocab = raw.count / MemoryLayout<Float>.size / window
            var logits: [Float] = raw.withUnsafeBytes { p in
                let f = p.bindMemory(to: Float.self)
                return Array(f[(ctx.count - 1) * vocab ..< ctx.count * vocab])
            }
            for v in tokenizer.vocabSize..<logits.count { logits[v] = -.infinity }
            let next = sample(&logits, temperature: temperature, topK: topK)
            if next == eosId { return }
            ids.append(next)
            onToken(next)
        }
    }

    private func sample(_ logits: inout [Float], temperature: Float, topK: Int) -> Int {
        if temperature <= 0 { return logits.indices.max { logits[$0] < logits[$1] }! }
        var s = logits.map { Double($0 / temperature) }
        if topK > 0 && topK < s.count {
            let kth = s.sorted(by: >)[topK - 1]
            for i in s.indices where s[i] < kth { s[i] = -.infinity }
        }
        let m = s.max()!
        let p = s.map { exp($0 - m) }
        var r = Double.random(in: 0..<p.reduce(0, +))
        for (i, v) in p.enumerated() { r -= v; if r <= 0 { return i } }
        return p.count - 1
    }
}
