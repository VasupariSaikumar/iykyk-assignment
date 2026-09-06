package com.iykyk.collage.domain

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import java.io.FileInputStream
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Wraps a FaceNet-style TFLite model.
 * Optimized with GPU acceleration and Lite Support library for faster pre-processing.
 */
class FaceEmbedder(context: Context, modelAssetName: String = "facenet.tflite") {

    private val inputSize = 160
    private val embeddingSize = 512

    private val interpreter: Interpreter
    private var gpuDelegate: GpuDelegate? = null

    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(127.5f, 127.5f))
        .build()

    init {
        try {
            val afd = context.assets.openFd(modelAssetName)
            val inputStream = FileInputStream(afd.fileDescriptor)
            val modelBuffer = inputStream.channel.map(
                FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength
            )
            
            val options = Interpreter.Options().apply {
                try {
                    gpuDelegate = GpuDelegate()
                    addDelegate(gpuDelegate)
                } catch (e: Exception) {
                    android.util.Log.w("FaceEmbedder", "GPU not available, falling back to CPU", e)
                    setNumThreads(4)
                }
            }
            interpreter = Interpreter(modelBuffer, options)
        } catch (e: Exception) {
            throw IllegalStateException("Critical Error: Failed to load FaceNet TFLite model '$modelAssetName' from assets.", e)
        }
    }

    /** Tight crop for embedding input. */
    private fun tightCrop(source: Bitmap, bbox: Rect): Bitmap {
        val left = max(0, bbox.left)
        val top = max(0, bbox.top)
        val right = min(source.width, bbox.right)
        val bottom = min(source.height, bbox.bottom)
        val w = (right - left).coerceAtLeast(1)
        val h = (bottom - top).coerceAtLeast(1)
        val cropped = Bitmap.createBitmap(source, left, top, w, h)
        return Bitmap.createScaledBitmap(cropped, inputSize, inputSize, true)
    }

    fun embed(sourceFrame: Bitmap, bbox: Rect): FloatArray {
        val faceBitmap = tightCrop(sourceFrame, bbox)
        
        // Use TensorImage and ImageProcessor for much faster pre-processing than manual loops
        var tensorImage = TensorImage(org.tensorflow.lite.DataType.FLOAT32)
        tensorImage.load(faceBitmap)
        tensorImage = imageProcessor.process(tensorImage)

        val output = FloatArray(embeddingSize)
        interpreter.run(tensorImage.buffer, arrayOf(output))
        return l2Normalize(output)
    }

    private fun l2Normalize(vec: FloatArray): FloatArray {
        var sumSq = 0f
        for (v in vec) sumSq += v * v
        val norm = sqrt(sumSq).coerceAtLeast(1e-6f)
        return FloatArray(vec.size) { vec[it] / norm }
    }

    fun close() {
        interpreter.close()
        gpuDelegate?.close()
    }

    companion object {
        fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
            var dot = 0f
            for (i in a.indices) dot += a[i] * b[i]
            return dot
        }
    }
}
