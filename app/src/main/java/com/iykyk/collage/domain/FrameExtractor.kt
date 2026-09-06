package com.iykyk.collage.domain

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * A single sampled frame from the source video.
 * timestampMs is kept because appearance-counting is defined in terms of continuous
 * *time* segments, not frame indices.
 */
data class SampledFrame(
    val timestampMs: Long,
    val bitmap: Bitmap
)

/**
 * Pulls frames from the video at a fixed rate.
 *
 * Rate tradeoff: 30s clip @ 6fps = 180 frames. That's plenty of temporal resolution to
 * catch "continuous visible segments" per the assignment's appearance definition, while
 * staying cheap enough to run face detection + embedding on every frame on a mid-range phone
 * within a reasonable processing time. Drop to 4fps first if perf becomes an issue -- do NOT
 * drop resolution, since representative-shot quality (sharpness/frontality) depends on it.
 */
class FrameExtractor(private val context: Context) {

    suspend fun extractFrames(
        videoUri: Uri,
        samplesPerSecond: Int = 6,
        onProgress: (Float) -> Unit = {}
    ): List<SampledFrame> = withContext(Dispatchers.Default) {
        val retriever = MediaMetadataRetriever()
        val frames = mutableListOf<SampledFrame>()

        try {
            retriever.setDataSource(context, videoUri)

            val rotation = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)
                ?.toIntOrNull() ?: 0

            val durationMs = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull() ?: 0L

            val stepMs = (1000L / samplesPerSecond).coerceAtLeast(1L)
            var t = 0L
            val totalSteps = (durationMs / stepMs).coerceAtLeast(1L)
            var step = 0L

            while (t < durationMs) {
                // OPTION_CLOSEST is slower than OPTION_CLOSEST_SYNC but gives the actual
                // frame at time t rather than snapping to the nearest keyframe -- matters
                // for accurate appearance timing.
                val rawBmp = retriever.getFrameAtTime(t * 1000, MediaMetadataRetriever.OPTION_CLOSEST)
                if (rawBmp != null) {
                    val bmp = if (rotation != 0) {
                        val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
                        Bitmap.createBitmap(rawBmp, 0, 0, rawBmp.width, rawBmp.height, matrix, true)
                    } else {
                        rawBmp
                    }
                    frames.add(SampledFrame(t, bmp))
                }
                t += stepMs
                step++
                onProgress((step.toFloat() / totalSteps).coerceIn(0f, 1f))
            }
        } finally {
            retriever.release()
        }

        frames
    }
}
