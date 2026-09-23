package com.drone.detector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

data class DetectResult(
    val detections: List<Detection>,
    val maxScore: Float,
    val inferMs: Long,
    val preprocessMs: Long,
    val backend: String,
)

class YoloDetector(context: Context) : AutoCloseable {

    private val interpreter: Interpreter
    private val nnApiDelegate: NnApiDelegate?
    private val gpuDelegate: GpuDelegate?
    val backend: String
    private val inputBuffer: ByteBuffer
    private val outputBuffer: ByteBuffer
    private val outputFloats = FloatArray(5 * NUM_PREDS)
    private val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
    private val canvasBmp = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
    private val canvas = Canvas(canvasBmp)
    private val filterPaint = Paint(Paint.FILTER_BITMAP_FLAG)

    init {
        val model = AssetPack.load(context)
        val created = createInterpreter(model)
        interpreter = created.interpreter
        nnApiDelegate = created.nnApi
        gpuDelegate = created.gpu
        backend = created.backend
        interpreter.allocateTensors()
        inputBuffer = ByteBuffer.allocateDirect(1 * 3 * INPUT_SIZE * INPUT_SIZE * 4)
            .order(ByteOrder.nativeOrder())
        outputBuffer = ByteBuffer.allocateDirect(1 * 5 * NUM_PREDS * 4)
            .order(ByteOrder.nativeOrder())
        Log.i(TAG, "TFLite backend=$backend")
    }

    fun detect(bitmap: Bitmap): DetectResult {
        val t0 = SystemClock.elapsedRealtime()
        val letterbox = letterbox(bitmap)
        fillNchw(canvasBmp)
        val t1 = SystemClock.elapsedRealtime()
        inputBuffer.rewind()
        outputBuffer.rewind()
        interpreter.run(inputBuffer, outputBuffer)
        outputBuffer.rewind()
        outputBuffer.asFloatBuffer().get(outputFloats)
        val t2 = SystemClock.elapsedRealtime()
        val decoded = decode(letterbox, bitmap.width, bitmap.height)
        return DetectResult(
            detections = decoded.first,
            maxScore = decoded.second,
            inferMs = t2 - t1,
            preprocessMs = t1 - t0,
            backend = backend,
        )
    }

    private fun decode(letterbox: Letterbox, imageWidth: Int, imageHeight: Int): Pair<List<Detection>, Float> {
        var maxScore = 0f
        val raw = ArrayList<Detection>(32)
        for (i in 0 until NUM_PREDS) {
            val score = outputFloats[4 * NUM_PREDS + i]
            if (score > maxScore) maxScore = score
            if (score < SCORE_FLOOR) continue
            val cx = outputFloats[i] * INPUT_SIZE
            val cy = outputFloats[NUM_PREDS + i] * INPUT_SIZE
            val w = outputFloats[2 * NUM_PREDS + i] * INPUT_SIZE
            val h = outputFloats[3 * NUM_PREDS + i] * INPUT_SIZE
            var left = (cx - w / 2f - letterbox.padX) / letterbox.scale
            var top = (cy - h / 2f - letterbox.padY) / letterbox.scale
            var right = (cx + w / 2f - letterbox.padX) / letterbox.scale
            var bottom = (cy + h / 2f - letterbox.padY) / letterbox.scale
            left = left.coerceIn(0f, imageWidth.toFloat())
            top = top.coerceIn(0f, imageHeight.toFloat())
            right = right.coerceIn(0f, imageWidth.toFloat())
            bottom = bottom.coerceIn(0f, imageHeight.toFloat())
            if (right - left < 4f || bottom - top < 4f) continue
            raw += Detection(left, top, right, bottom, score)
        }
        return nms(raw) to maxScore
    }

    private fun nms(dets: List<Detection>): List<Detection> {
        val sorted = dets.sortedByDescending { it.score }.toMutableList()
        val kept = ArrayList<Detection>(min(MAX_DETECTIONS, sorted.size))
        while (sorted.isNotEmpty() && kept.size < MAX_DETECTIONS) {
            val best = sorted.removeAt(0)
            kept += best
            val iterator = sorted.iterator()
            while (iterator.hasNext()) {
                if (iou(best, iterator.next()) > IOU_THRESHOLD) iterator.remove()
            }
        }
        return kept
    }

    private fun fillNchw(bitmap: Bitmap) {
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        inputBuffer.rewind()
        val area = INPUT_SIZE * INPUT_SIZE
        for (c in 0..2) {
            for (i in 0 until area) {
                val p = pixels[i]
                val value = when (c) {
                    0 -> ((p shr 16) and 0xFF) * INV_255
                    1 -> ((p shr 8) and 0xFF) * INV_255
                    else -> (p and 0xFF) * INV_255
                }
                inputBuffer.putFloat(value)
            }
        }
    }

    private fun letterbox(src: Bitmap): Letterbox {
        val scale = min(INPUT_SIZE / src.width.toFloat(), INPUT_SIZE / src.height.toFloat())
        val newW = max(1, (src.width * scale).toInt())
        val newH = max(1, (src.height * scale).toInt())
        val padX = (INPUT_SIZE - newW) / 2f
        val padY = (INPUT_SIZE - newH) / 2f
        canvas.drawColor(Color.rgb(114, 114, 114))
        canvas.drawBitmap(src, null, RectF(padX, padY, padX + newW, padY + newH), filterPaint)
        return Letterbox(scale, padX, padY)
    }

    override fun close() {
        interpreter.close()
        gpuDelegate?.close()
        nnApiDelegate?.close()
        canvasBmp.recycle()
    }

    private data class Letterbox(
        val scale: Float,
        val padX: Float,
        val padY: Float,
    )

    companion object {
        private const val TAG = "AntiUav"
        const val MODEL_ASSET = "cfg.dat"
        const val INPUT_SIZE = 640
        const val NUM_PREDS = 8400
        const val SCORE_FLOOR = 0.60f
        const val IOU_THRESHOLD = 0.45f
        const val MAX_DETECTIONS = 10
        private const val INV_255 = 1f / 255f

        private fun createInterpreter(model: ByteBuffer): CreatedInterpreter {
            tryNnapi(model)?.let { return it }
            tryGpu(model)?.let { return it }
            model.rewind()
            val cpu = Interpreter(
                model,
                Interpreter.Options().apply {
                    setNumThreads(4)
                    setUseXNNPACK(true)
                },
            )
            Log.i(TAG, "Using CPU/XNNPACK")
            return CreatedInterpreter(cpu, null, null, "CPU/XNNPACK")
        }

        private fun tryNnapi(model: ByteBuffer): CreatedInterpreter? {
            var nnapi: NnApiDelegate? = null
            return try {
                val nnapiOptions = NnApiDelegate.Options().apply {
                    setAllowFp16(true)
                    setUseNnapiCpu(false)
                    setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER)
                }
                nnapi = NnApiDelegate(nnapiOptions)
                model.rewind()
                val interp = Interpreter(model, Interpreter.Options().addDelegate(nnapi))
                Log.i(TAG, "NNAPI (NPU) enabled")
                CreatedInterpreter(interp, nnapi, null, "NPU/NNAPI")
            } catch (t: Throwable) {
                Log.w(TAG, "NNAPI failed, trying GPU", t)
                nnapi?.close()
                null
            }
        }

        private fun tryGpu(model: ByteBuffer): CreatedInterpreter? {
            var gpu: GpuDelegate? = null
            return try {
                val compat = CompatibilityList()
                if (!compat.isDelegateSupportedOnThisDevice) return null
                gpu = GpuDelegate(compat.bestOptionsForThisDevice)
                model.rewind()
                val interp = Interpreter(model, Interpreter.Options().addDelegate(gpu))
                Log.i(TAG, "GPU delegate enabled")
                CreatedInterpreter(interp, null, gpu, "GPU")
            } catch (t: Throwable) {
                Log.w(TAG, "GPU failed, using CPU", t)
                gpu?.close()
                null
            }
        }
    }

    private data class CreatedInterpreter(
        val interpreter: Interpreter,
        val nnApi: NnApiDelegate?,
        val gpu: GpuDelegate?,
        val backend: String,
    )
}

fun iou(a: Detection, b: Detection): Float {
    val x1 = max(a.left, b.left)
    val y1 = max(a.top, b.top)
    val x2 = min(a.right, b.right)
    val y2 = min(a.bottom, b.bottom)
    val inter = max(0f, x2 - x1) * max(0f, y2 - y1)
    val union = a.width * a.height + b.width * b.height - inter
    return if (union <= 0f) 0f else inter / union
}
