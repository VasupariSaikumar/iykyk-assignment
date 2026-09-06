package com.iykyk.collage.domain

import android.graphics.Bitmap
import android.graphics.Rect
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

object ShotScorer {

    /**
     * Laplacian-variance sharpness estimate on a grayscale crop.
     * Higher variance = more high-frequency detail = more in-focus.
     * Cheap 3x3 convolution -- fine at face-crop resolution (~100-300px), called once
     * per detected face so keep the crop small before calling this.
     */
    fun computeSharpness(source: Bitmap, bbox: Rect): Float {
        val left = max(0, bbox.left)
        val top = max(0, bbox.top)
        val right = min(source.width, bbox.right)
        val bottom = min(source.height, bbox.bottom)
        val w = (right - left).coerceAtLeast(2)
        val h = (bottom - top).coerceAtLeast(2)

        // Downscale to a fixed small size purely for a fast, size-invariant sharpness metric.
        val scaled = Bitmap.createScaledBitmap(
            Bitmap.createBitmap(source, left, top, w, h), 64, 64, true
        )
        val gray = Array(64) { y -> FloatArray(64) { x ->
            val p = scaled.getPixel(x, y)
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            0.299f * r + 0.587f * g + 0.114f * b
        }}

        var mean = 0.0
        val lap = Array(62) { FloatArray(62) }
        for (y in 1..62) {
            for (x in 1..62) {
                val value = 4 * gray[y][x] - gray[y - 1][x] - gray[y + 1][x] -
                        gray[y][x - 1] - gray[y][x + 1]
                lap[y - 1][x - 1] = value
                mean += value
            }
        }
        mean /= (62 * 62)

        var variance = 0.0
        for (row in lap) for (v in row) variance += (v - mean) * (v - mean)
        return (variance / (62 * 62)).toFloat()
    }

    /**
     * Combined quality score for choosing a representative shot. Weights are a starting
     * point -- tune against the sample videos and note your final weights in the README.
     */
    fun score(obs: FaceObservation, maxSharpnessInSet: Float): Float {
        val frontality = 1f - (min(1f, abs(obs.headEulerAngleY) / 45f) * 0.7f +
                min(1f, abs(obs.headEulerAngleZ) / 30f) * 0.3f)
        val normalizedSharpness = if (maxSharpnessInSet > 0f) {
            (obs.sharpness / maxSharpnessInSet).coerceIn(0f, 1f)
        } else 0f
        val eyesOpen = listOfNotNull(obs.leftEyeOpenProb, obs.rightEyeOpenProb)
            .let { if (it.isEmpty()) 0.5f else it.average().toFloat() }
        val smiling = obs.smilingProb ?: 0.5f

        return frontality * 0.35f +
                normalizedSharpness * 0.35f +
                eyesOpen * 0.20f +
                smiling * 0.10f
    }

    fun pickBestShot(observations: List<FaceObservation>): FaceObservation {
        val maxSharp = observations.maxOf { it.sharpness }
        return observations.maxBy { score(it, maxSharp) }
    }
}
