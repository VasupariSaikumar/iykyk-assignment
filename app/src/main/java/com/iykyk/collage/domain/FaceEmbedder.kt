package com.iykyk.collage.domain

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Wraps a FaceNet-style TFLite model.
 * Throws IllegalStateException if model fails to load to prevent silent failures.
 */
class FaceEmbedder(context: Context, modelAssetName: String = "facenet.tflite") {

    private val inputSize = 160
    private val embeddingSize = 512

    private val interpreter: Interpreter

    init {
        try {
            val afd = context.assets.openFd(modelAssetName)
            val inputStream = FileInputStream(afd.fileDescriptor)
            val modelBuffer = inputStream.channel.map(
                FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength
            )
            interpreter = Interpreter(modelBuffer, Interpreter.Options().apply { setNumThreads(4) })
        } catch (e: Exception) {
            throw IllegalStateException("Critical Error: Failed to load FaceNet TFLite model '$modelAssetName' from assets. " +
                    "Ensure the file exists and is not compressed.", e)
        }
    }

    /** Tight crop for embedding input -- NOT the generous collage crop. */
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
        val face = tightCrop(sourceFrame, bbox)
        val inputBuffer = bitmapToByteBuffer(face)
        // Match the model's output shape: [1, 512]
        val output = FloatArray(embeddingSize)
        interpreter.run(inputBuffer, arrayOf(output))
        return l2Normalize(output)
    }

    private fun bitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        buffer.order(ByteOrder.nativeOrder())
        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        for (pixel in pixels) {
            buffer.putFloat(((pixel shr 16 and 0xFF) - 127.5f) / 127.5f)
            buffer.putFloat(((pixel shr 8 and 0xFF) - 127.5f) / 127.5f)
            buffer.putFloat(((pixel and 0xFF) - 127.5f) / 127.5f)
        }
        buffer.rewind()
        return buffer
    }

    private fun l2Normalize(vec: FloatArray): FloatArray {
        var sumSq = 0f
        for (v in vec) sumSq += v * v
        val norm = sqrt(sumSq).coerceAtLeast(1e-6f)
        return FloatArray(vec.size) { vec[it] / norm }
    }

    fun close() = interpreter.close()

    companion object {
        fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
            var dot = 0f
            for (i in a.indices) dot += a[i] * b[i]
            return dot
        }
    }
}
