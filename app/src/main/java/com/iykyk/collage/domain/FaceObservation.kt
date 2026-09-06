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
    val sourceFrame: Bitmap
)

/** A continuous visible segment of one person, per the assignment's appearance definition. */
data class Appearance(
    val startMs: Long,
    val endMs: Long,
    val observations: List<FaceObservation>
) {
    /** Mean embedding across the appearance -- more stable than any single frame. */
    fun representativeEmbedding(): FloatArray {
        val dim = observations.first().embedding.size
        val avg = FloatArray(dim)
        for (obs in observations) {
            for (i in 0 until dim) avg[i] += obs.embedding[i]
        }
        for (i in 0 until dim) avg[i] /= observations.size
        return avg
    }
}

/** Final output: one clustered identity, with all its appearances and a chosen best shot. */
data class PersonIdentity(
    val appearances: List<Appearance>,
    val bestShot: FaceObservation
) {
    val appearanceCount: Int get() = appearances.size
}
