package com.iykyk.collage.domain

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicInteger

data class SampledFrame(
    val timestampMs: Long,
    val bitmap: Bitmap
)

class FrameExtractor(private val context: Context) {

    /**
     * Extracts frames in parallel using multiple retriever instances.
     * Dramatically faster than sequential extraction while maintaining frame accuracy.
     */
    suspend fun extractFrames(
        videoUri: Uri,
        samplesPerSecond: Int = 6,
        onProgress: (Float) -> Unit = {}
    ): List<SampledFrame> = withContext(Dispatchers.IO) {
        val mainRetriever = MediaMetadataRetriever()
        mainRetriever.setDataSource(context, videoUri)
        val durationMs = mainRetriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
        val rotation = mainRetriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toIntOrNull() ?: 0
        mainRetriever.release()

        if (durationMs <= 0) return@withContext emptyList()

        val stepMs = (1000L / samplesPerSecond).coerceAtLeast(1L)
        val timestamps = mutableListOf<Long>()
        var t = 0L
        while (t < durationMs) {
            timestamps.add(t)
            t += stepMs
        }

        val totalFrames = timestamps.size
        val processedCount = AtomicInteger(0)
        
        // Use 4 parallel workers for extraction
        val numWorkers = 4
        val chunkSize = (totalFrames + numWorkers - 1) / numWorkers

        val allFrames = timestamps.chunked(chunkSize).map { chunk ->
            async {
                val retriever = MediaMetadataRetriever()
                retriever.setDataSource(context, videoUri)
                val frames = chunk.map { time ->
                    val rawBmp = retriever.getFrameAtTime(time * 1000, MediaMetadataRetriever.OPTION_CLOSEST)
                    val processedBmp = rawBmp?.let {
                        if (rotation != 0) {
                            val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
                            Bitmap.createBitmap(it, 0, 0, it.width, it.height, matrix, true)
                        } else it
                    }
                    val current = processedCount.incrementAndGet()
                    onProgress(current.toFloat() / totalFrames)
                    
                    if (processedBmp != null) SampledFrame(time, processedBmp) else null
                }.filterNotNull()
                retriever.release()
                frames
            }
        }.awaitAll().flatten().sortedBy { it.timestampMs }

        allFrames
    }
}
