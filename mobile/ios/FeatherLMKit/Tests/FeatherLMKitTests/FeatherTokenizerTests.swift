import XCTest
@testable import FeatherLMKit

final class FeatherTokenizerTests: XCTestCase {
    func testByteFallbackRoundTrip() throws {
        let json = #"{"format":"featherlm-byte-bpe","version":1,"special_tokens":["<|pad|>","<|bos|>","<|eos|>"],"merges":[]}"#
        let tok = try FeatherTokenizer(json: Data(json.utf8))
        let text = "Hello, café 東京 🙂"
        XCTAssertEqual(tok.decode(tok.encode(text)), text)
    }
}
