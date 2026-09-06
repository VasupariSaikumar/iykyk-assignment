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
     * Uses OPTION_CLOSEST for accurate (non-duplicate) frames.
     */
    suspend fun <T> processFramesParallel(
        videoUri: Uri,
        samplesPerSecond: Int = 2,
        onProgress: (Float) -> Unit = {},
        processor: suspend (SampledFrame) -> T
    ): List<T> = withContext(Dispatchers.IO) {
        val metaRetriever = MediaMetadataRetriever()
        try {
            metaRetriever.setDataSource(context, videoUri)
        } catch (e: Exception) {
            android.util.Log.e("FrameExtractor", "Failed to set data source: ${e.message}", e)
            return@withContext emptyList()
        }
        
        val durationMs = metaRetriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
        val rotation = metaRetriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toIntOrNull() ?: 0
        metaRetriever.release()

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
        
        // Use 3 parallel workers (staying safe with memory)
        val numWorkers = 3
        val chunkSize = (totalFrames + numWorkers - 1) / numWorkers

        val results = timestamps.chunked(chunkSize).map { chunk ->
            async {
                val workerRetriever = MediaMetadataRetriever()
                workerRetriever.setDataSource(context, videoUri)
                val chunkResults = chunk.map { timeMs ->
                    // OPTION_CLOSEST is essential for seeing different people across time
                    val rawBmp = workerRetriever.getFrameAtTime(timeMs * 1000, MediaMetadataRetriever.OPTION_CLOSEST)
                    val result = if (rawBmp != null) {
                        val processedBmp = if (rotation != 0) {
                            val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
                            val rotated = Bitmap.createBitmap(rawBmp, 0, 0, rawBmp.width, rawBmp.height, matrix, true)
                            rawBmp.recycle()
                            rotated
                        } else rawBmp
                        
                        val frameResult = processor(SampledFrame(timeMs, processedBmp))
                        processedBmp.recycle()
                        frameResult
                    } else null
                    
                    val current = processedCount.incrementAndGet()
                    onProgress(current.toFloat() / totalFrames)
                    result
                }.filterNotNull()
                workerRetriever.release()
                chunkResults
            }
        }.awaitAll().flatten()

        results
    }
}
