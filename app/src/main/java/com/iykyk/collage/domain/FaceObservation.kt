package com.iykyk.collage.domain

import android.graphics.Bitmap
import android.graphics.Rect

/** One detected face, at one timestamp, with its embedding and quality signals attached. */
data class FaceObservation(
    val timestampMs: Long,
    val bbox: Rect,
    val embedding: FloatArray,
    val headEulerAngleY: Float,
    val headEulerAngleZ: Float,
    val leftEyeOpenProb: Float?,
    val rightEyeOpenProb: Float?,
    val smilingProb: Float?,
    val sharpness: Float,
    val faceCrop: Bitmap // Stored crop for the final collage, NOT the full frame (save memory!)
)

/** A continuous visible segment of one person, per the assignment's appearance definition. */
data class Appearance(
    val startMs: Long,
    val endMs: Long,
    val observations: List<FaceObservation>
) {
    /**
     * Mean embedding across the sharpest observations -- much more stable and accurate than 
     * using every single frame or just the first seen.
     */
    fun representativeEmbedding(): FloatArray {
        if (observations.isEmpty()) return floatArrayOf()
        
        // Take top 5 sharpest observations to form a high-quality signature
        val bestObs = observations.sortedByDescending { it.sharpness }.take(5)
        
        val dim = bestObs.first().embedding.size
        val avg = FloatArray(dim)
        for (obs in bestObs) {
            for (i in 0 until dim) avg[i] += obs.embedding[i]
        }
        for (i in 0 until dim) avg[i] /= bestObs.size
        
        // Re-normalize to ensure the average vector is still on the hypersphere
        var norm = 0f
        for (v in avg) norm += v * v
        norm = kotlin.math.sqrt(norm).coerceAtLeast(1e-6f)
        return FloatArray(dim) { avg[it] / norm }
    }
}

/** Final output: one clustered identity, with all its appearances and a chosen best shot. */
data class PersonIdentity(
    val appearances: List<Appearance>,
    val bestShot: FaceObservation
) {
    val appearanceCount: Int get() = appearances.size
}
