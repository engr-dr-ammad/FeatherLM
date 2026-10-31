package dev.featherlm.demo

import android.os.Bundle
import android.os.SystemClock
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction

/** Minimal on-device demo: prompt in, locally generated text out. No network access. */
class MainActivity : AppCompatActivity() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private var model: FeatherModel? = null
    private var job: Job? = null
    @Volatile private var cancelled = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val prompt = findViewById<EditText>(R.id.prompt)
        val output = findViewById<TextView>(R.id.output)
        val stats = findViewById<TextView>(R.id.stats)
        val gen = findViewById<Button>(R.id.generate)
        val stop = findViewById<Button>(R.id.stop)
        gen.isEnabled = false
        stats.text = "Loading model…"

        scope.launch {
            model = withContext(Dispatchers.IO) {
                val tok = FeatherTokenizer(assets.open("tokenizer.json").bufferedReader().readText())
                FeatherModel(assets.open("model.onnx").readBytes(),
                    assets.open("onnx_config.json").bufferedReader().readText(), tok)
            }
            stats.text = "Model loaded"
            gen.isEnabled = true
        }

        gen.setOnClickListener {
            val m = model ?: return@setOnClickListener
            val text = prompt.text.toString()
            output.text = text
            cancelled = false
            gen.isEnabled = false; stop.isEnabled = true
            job = scope.launch {
                val t0 = SystemClock.elapsedRealtime()
                var n = 0
                val pending = ArrayList<Int>()
                withContext(Dispatchers.Default) {
                    m.generate(text, maxTokens = 120, temperature = 0.8f, topK = 40, seed = 0,
                        isCancelled = { cancelled }) { id ->
                        n++
                        pending.add(id)
                        val piece = decodeComplete(m.tokenizer.bytes(pending))
                        if (piece != null) {
                            pending.clear()
                            runOnUiThread { output.append(piece) }
                        }
                    }
                }
                val s = (SystemClock.elapsedRealtime() - t0) / 1000.0
                stats.text = "%d tokens in %.1f s (%.1f tok/s)".format(n, s, if (s > 0) n / s else 0.0)
                gen.isEnabled = true; stop.isEnabled = false
            }
        }
        stop.setOnClickListener { cancelled = true }
    }

    /** Returns text if the bytes form complete UTF-8, else null (wait for more tokens). */
    private fun decodeComplete(bytes: ByteArray): String? = try {
        Charsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
            .decode(ByteBuffer.wrap(bytes)).toString()
    } catch (e: Exception) { if (bytes.size > 8) String(bytes, Charsets.UTF_8) else null }

    override fun onDestroy() {
        cancelled = true
        scope.cancel()
        model?.close()
        super.onDestroy()
    }
}
