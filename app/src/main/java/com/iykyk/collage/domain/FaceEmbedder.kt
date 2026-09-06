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
 * Ultimate Diagnostic Version.
 */
class FaceEmbedder(context: Context, modelAssetName: String = "facenet.tflite") {

    private val inputSize: Int
    private val interpreter: Interpreter
    private val outputSize: Int
    private val isFloatModel: Boolean

    init {
        try {
            val afd = context.assets.openFd(modelAssetName)
            val inputStream = FileInputStream(afd.fileDescriptor)
            val modelBuffer = inputStream.channel.map(
                FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength
            )
            
            val options = Interpreter.Options().apply { setNumThreads(4) }
            interpreter = Interpreter(modelBuffer, options)
            
            val inputShape = interpreter.getInputTensor(0).shape()
            inputSize = inputShape[1] // Assumes [1, size, size, 3]
            
            val outputShape = interpreter.getOutputTensor(0).shape()
            outputSize = outputShape.last()
            
            isFloatModel = interpreter.getInputTensor(0).dataType() == org.tensorflow.lite.DataType.FLOAT32
            
            android.util.Log.i("FaceEmbedder", "DIAGNOSTIC: InputSize=$inputSize, OutputSize=$outputSize, isFloat=$isFloatModel")
        } catch (e: Exception) {
            throw IllegalStateException("Critical Error: Failed to load FaceNet TFLite model '$modelAssetName' from assets.", e)
        }
    }

    private fun robustCrop(source: Bitmap, bbox: Rect): Bitmap {
        val margin = (bbox.width() * 0.1f).toInt()
        val left = max(0, bbox.left - margin)
        val top = max(0, bbox.top - margin)
        val right = min(source.width, bbox.right + margin)
        val bottom = min(source.height, bbox.bottom + margin)
        val w = (right - left).coerceAtLeast(1)
        val h = (bottom - top).coerceAtLeast(1)
        val cropped = Bitmap.createBitmap(source, left, top, w, h)
        return Bitmap.createScaledBitmap(cropped, inputSize, inputSize, true)
    }

    fun embed(sourceFrame: Bitmap, bbox: Rect): FloatArray {
        val faceBitmap = robustCrop(sourceFrame, bbox)
        val pixels = IntArray(inputSize * inputSize)
        faceBitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        
        // Manual whitening
        var sum = 0f
        for (p in pixels) sum += ((p shr 16) and 0xFF) + ((p shr 8) and 0xFF) + (p and 0xFF)
        val mean = sum / (pixels.size * 3f)
        
        var sumSq = 0f
        for (p in pixels) {
            val r = ((p shr 16) and 0xFF).toFloat(); val g = ((p shr 8) and 0xFF).toFloat(); val b = (p and 0xFF).toFloat()
            sumSq += (r - mean) * (r - mean) + (g - mean) * (g - mean) + (b - mean) * (b - mean)
        }
        val std = sqrt(sumSq / (pixels.size * 3f))
        val stdAdj = max(std, 1.0f / sqrt(pixels.size * 3f))

        val byteBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4)
        byteBuffer.order(ByteOrder.nativeOrder())
        for (p in pixels) {
            byteBuffer.putFloat((((p shr 16) and 0xFF) - mean) / stdAdj)
            byteBuffer.putFloat((((p shr 8) and 0xFF) - mean) / stdAdj)
            byteBuffer.putFloat(((p and 0xFF) - mean) / stdAdj)
        }

        val output = Array(1) { FloatArray(outputSize) }
        byteBuffer.rewind()
        interpreter.run(byteBuffer, output)
        
        val raw = output[0]
        val normalized = l2Normalize(raw)
        
        // LOG RAW DATA SPREAD: help detect if model is "collapsed"
        val minVal = raw.minOrNull() ?: 0f
        val maxVal = raw.maxOrNull() ?: 0f
        android.util.Log.v("FaceEmbedder", "Face at ${bbox.centerX()},${bbox.centerY()}: Raw range [${"%.3f".format(minVal)}, ${"%.3f".format(maxVal)}]")
        
        return normalized
    }

    private fun l2Normalize(vec: FloatArray): FloatArray {
        var sumSq = 0f
        for (v in vec) sumSq += v * v
        val norm = sqrt(sumSq).coerceAtLeast(1e-6f)
        return FloatArray(vec.size) { vec[it] / norm }
    }

    fun close() {
        interpreter.close()
    }

    companion object {
        fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
            if (a.isEmpty() || b.isEmpty() || a.size != b.size) return 0f
            var dot = 0f
            for (i in a.indices) dot += a[i] * b[i]
            return dot
        }
    }
}
