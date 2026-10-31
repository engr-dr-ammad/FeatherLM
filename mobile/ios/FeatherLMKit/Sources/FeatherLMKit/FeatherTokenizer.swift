import Foundation

/// Byte-level BPE tokenizer compatible with featherlm/tokenizer/tokenizer.py (format version 1).
public final class FeatherTokenizer {
    private let nSpecial: Int
    private var ranks: [UInt64: Int] = [:]
    private(set) var vocab: [[UInt8]] = []
    // ICU regular expressions treat \w as Unicode word characters, matching Python's re.
    private let regex = try! NSRegularExpression(
        pattern: #"'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d{1,3}| ?[^\s\w]+|\s+(?!\S)|\s+"#)

    public var vocabSize: Int { vocab.count }

    public init(json: Data) throws {
        guard let o = try JSONSerialization.jsonObject(with: json) as? [String: Any],
              o["format"] as? String == "featherlm-byte-bpe",
              let specials = o["special_tokens"] as? [String],
              let merges = o["merges"] as? [[Int]] else {
            throw NSError(domain: "FeatherLM", code: 1, userInfo: [NSLocalizedDescriptionKey: "not a FeatherLM tokenizer"])
        }
        nSpecial = specials.count
        vocab = specials.map { Array($0.utf8) } + (0..<256).map { [UInt8($0)] }
        for (i, m) in merges.enumerated() {
            ranks[Self.key(m[0], m[1])] = i
            vocab.append(vocab[m[0]] + vocab[m[1]])
        }
    }

    private static func key(_ a: Int, _ b: Int) -> UInt64 { (UInt64(a) << 32) | UInt64(UInt32(b)) }

    public func encode(_ text: String) -> [Int] {
        let s = text.precomposedStringWithCanonicalMapping  // NFC, like the Python tokenizer
        let ns = s as NSString
        var out: [Int] = []
        for m in regex.matches(in: s, range: NSRange(location: 0, length: ns.length)) {
            var ids = Array(ns.substring(with: m.range).utf8).map { Int($0) + nSpecial }
            while ids.count > 1 {
                var best = Int.max, at = -1
                for i in 0..<(ids.count - 1) {
                    if let r = ranks[Self.key(ids[i], ids[i + 1])], r < best { best = r; at = i }
                }
                if at < 0 { break }
                ids[at] = nSpecial + 256 + best
                ids.remove(at: at + 1)
            }
            out += ids
        }
        return out
    }

    public func bytes(_ ids: [Int]) -> [UInt8] {
        ids.filter { $0 >= nSpecial && $0 < vocab.count }.flatMap { vocab[$0] }
    }

    public func decode(_ ids: [Int]) -> String { String(decoding: bytes(ids), as: UTF8.self) }
}
