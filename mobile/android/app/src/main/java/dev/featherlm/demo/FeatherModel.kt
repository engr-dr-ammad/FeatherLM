package dev.featherlm.demo

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.nio.LongBuffer
import kotlin.math.exp
import kotlin.random.Random

/**
 * Runs the fixed-window ONNX export: input_ids (1, W) right-padded -> logits (1, W, V).
 * Mirrors mobile/reference_runtime.py, which is tested on desktop ONNX Runtime.
 * Each token re-runs the whole window (simple, O(W) per token); a stateful single-step
 * export is the planned optimisation.
 */
class FeatherModel(modelBytes: ByteArray, configJson: String, val tokenizer: FeatherTokenizer, threads: Int = 2) {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val window: Int
    private val padId: Long
    private val eosId: Int

    init {
        val cfg = JSONObject(configJson)
        window = cfg.getInt("window")
        padId = cfg.getLong("pad_id")
        eosId = cfg.getInt("eos_id")
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(threads)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(modelBytes, opts)
    }

    fun generate(
        prompt: String, maxTokens: Int, temperature: Float, topK: Int, seed: Int,
        isCancelled: () -> Boolean, onToken: (Int) -> Unit,
    ) {
        val rng = Random(seed)
        val ids = tokenizer.encode(prompt).toMutableList().ifEmpty { mutableListOf(1) }
        val buf = LongArray(window)
        repeat(maxTokens) {
            if (isCancelled()) return
            val ctx = if (ids.size > window) ids.subList(ids.size - window, ids.size) else ids
            buf.fill(padId)
            for (i in ctx.indices) buf[i] = ctx[i].toLong()
            val logits = OnnxTensor.createTensor(env, LongBuffer.wrap(buf), longArrayOf(1, window.toLong())).use { input ->
                session.run(mapOf("input_ids" to input)).use { out ->
                    @Suppress("UNCHECKED_CAST")
                    val all = (out[0].value as Array<Array<FloatArray>>)[0]
                    all[ctx.size - 1].copyOf()
                }
            }
            for (v in tokenizer.vocabSize until logits.size) logits[v] = Float.NEGATIVE_INFINITY
            val next = sample(logits, temperature, topK, rng)
            if (next == eosId) return
            ids.add(next)
            onToken(next)
        }
    }

    private fun sample(logits: FloatArray, temperature: Float, topK: Int, rng: Random): Int {
        if (temperature <= 0f) return logits.indices.maxBy { logits[it] }
        val scaled = DoubleArray(logits.size) { logits[it].toDouble() / temperature }
        if (topK in 1 until scaled.size) {
            val kth = scaled.sortedDescending()[topK - 1]
            for (i in scaled.indices) if (scaled[i] < kth) scaled[i] = Double.NEGATIVE_INFINITY
        }
        val max = scaled.max()
        val p = DoubleArray(scaled.size) { exp(scaled[it] - max) }
        var r = rng.nextDouble() * p.sum()
        for (i in p.indices) { r -= p[i]; if (r <= 0) return i }
        return p.size - 1
    }

    fun close() = session.close()
}
