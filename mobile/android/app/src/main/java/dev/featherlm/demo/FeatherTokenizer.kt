package dev.featherlm.demo

import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.text.Normalizer
import java.util.regex.Pattern

/** Byte-level BPE tokenizer compatible with featherlm/tokenizer/tokenizer.py (format version 1). */
class FeatherTokenizer(json: String) {
    private val nSpecial: Int
    private val ranks = HashMap<Long, Int>()
    private val vocab = ArrayList<ByteArray>()
    // Same pattern as the Python pre-tokenizer; UNICODE_CHARACTER_CLASS makes \w Unicode-aware like Python's re.
    private val pattern = Pattern.compile(
        "'(?:s|t|re|ve|m|ll|d)| ?[^\\W\\d_]+| ?\\d{1,3}| ?(?:[^\\s\\w]|_)+|\\s+(?!\\S)|\\s+",
        Pattern.UNICODE_CHARACTER_CLASS
    )

    init {
        val o = JSONObject(json)
        require(o.getString("format") == "featherlm-byte-bpe") { "not a FeatherLM tokenizer" }
        val specials = o.getJSONArray("special_tokens")
        nSpecial = specials.length()
        for (i in 0 until nSpecial) vocab.add(specials.getString(i).toByteArray())
        for (b in 0 until 256) vocab.add(byteArrayOf(b.toByte()))
        val merges = o.getJSONArray("merges")
        for (i in 0 until merges.length()) {
            val m = merges.getJSONArray(i)
            val a = m.getInt(0); val b = m.getInt(1)
            ranks[key(a, b)] = i
            vocab.add(vocab[a] + vocab[b])
        }
    }

    val vocabSize: Int get() = vocab.size

    private fun key(a: Int, b: Int): Long = (a.toLong() shl 32) or (b.toLong() and 0xffffffffL)

    fun encode(text: String): IntArray {
        val out = ArrayList<Int>()
        val m = pattern.matcher(Normalizer.normalize(text, Normalizer.Form.NFC))
        while (m.find()) {
            var ids = m.group().toByteArray(Charsets.UTF_8).map { (it.toInt() and 0xff) + nSpecial }.toMutableList()
            while (ids.size > 1) {
                var best = Int.MAX_VALUE; var at = -1
                for (i in 0 until ids.size - 1) {
                    val r = ranks[key(ids[i], ids[i + 1])] ?: continue
                    if (r < best) { best = r; at = i }
                }
                if (at < 0) break
                ids[at] = nSpecial + 256 + best
                ids.removeAt(at + 1)
            }
            out.addAll(ids)
        }
        return out.toIntArray()
    }

    /** Raw bytes for ids (special tokens skipped). Callers decode UTF-8 incrementally. */
    fun bytes(ids: List<Int>): ByteArray {
        val bos = ByteArrayOutputStream()
        for (i in ids) if (i >= nSpecial && i < vocab.size) bos.write(vocab[i])
        return bos.toByteArray()
    }
}
