package com.iykyk.collage.domain

import android.graphics.Rect
import kotlin.math.max
import kotlin.math.min

/**
 * Turns a per-frame stream of face observations into "appearances": continuous visible
 * segments of the same person, per the assignment's definition. This runs WITHIN one
 * video only, before cross-video-wide identity clustering.
 *
 * Two independent signals decide whether a detection continues an existing track:
 *  1. Embedding similarity (same person should look similar frame-to-frame)
 *  2. Spatial continuity (a real continuous appearance doesn't teleport across the frame)
 * Using both prevents two similar-looking people from getting merged into one track when
 * they're both on screen (the two-person-shares-a-frame case from the spec), since spatial
 * position disambiguates when embeddings alone might be close.
 *
 * A track closes -- finalizing one Appearance -- if it goes unmatched for `maxGapFrames`
 * consecutive sampled frames (covers brief occlusion / blur without over-fragmenting a
 * single real appearance into several).
 */
class AppearanceTracker(
    private val similarityThreshold: Float = 0.70f,
    private val maxCenterDistanceFraction: Float = 0.20f, // balanced
    private val maxGapFrames: Int = 2
) {

    private class Track(
        var lastBbox: Rect,
        var lastSeenFrameIndex: Int,
        val observations: MutableList<FaceObservation>
    )

    fun track(
        framesInOrder: List<List<FaceObservation>>, // outer list = one entry per sampled frame, in time order
        frameWidth: Int,
        frameHeight: Int
    ): List<Appearance> {
        val diagonal = kotlin.math.sqrt((frameWidth * frameWidth + frameHeight * frameHeight).toDouble()).toFloat()
        val maxCenterDist = diagonal * maxCenterDistanceFraction

        val activeTracks = mutableListOf<Track>()
        val finishedAppearances = mutableListOf<Appearance>()

        framesInOrder.forEachIndexed { frameIndex, facesInFrame ->
            android.util.Log.d("AppearanceTracker", "Frame $frameIndex: ${facesInFrame.size} faces detected")
            val unmatchedTracks = activeTracks.toMutableList()
            val usedTracks = mutableSetOf<Track>()

            // Greedy best-match: for each detected face this frame, find the closest
            // active track by embedding similarity among spatially-plausible candidates.
            for (face in facesInFrame) {
                var bestTrack: Track? = null
                var bestScore = similarityThreshold

                for (track in unmatchedTracks) {
                    if (track in usedTracks) continue
                    val centerDist = centerDistance(track.lastBbox, face.bbox)
                    
                    val sim = FaceEmbedder.cosineSimilarity(
                        track.observations.last().embedding, face.embedding
                    )
                    
                    android.util.Log.v("AppearanceTracker", "Compare: Track at ${track.lastBbox.centerX()},${track.lastBbox.centerY()} vs Face at ${face.bbox.centerX()},${face.bbox.centerY()}. Dist: ${"%.1f".format(centerDist)} (max: ${"%.1f".format(maxCenterDist)}), Sim: ${"%.3f".format(sim)} (min: $similarityThreshold)")

                    if (centerDist <= maxCenterDist && sim > bestScore) {
                        bestScore = sim
                        bestTrack = track
                    }
                }

                if (bestTrack != null) {
                    bestTrack.observations.add(face)
                    bestTrack.lastBbox = face.bbox
                    bestTrack.lastSeenFrameIndex = frameIndex
                    usedTracks.add(bestTrack)
                    android.util.Log.d("AppearanceTracker", "Matched face to track. New size: ${bestTrack.observations.size}")
                } else {
                    val newTrack = Track(face.bbox, frameIndex, mutableListOf(face))
                    activeTracks.add(newTrack)
                    android.util.Log.d("AppearanceTracker", "Created NEW track at ${face.bbox.centerX()},${face.bbox.centerY()}")
                }
            }

            // Close out tracks that have gone quiet for too long.
            val stillActive = mutableListOf<Track>()
            for (track in activeTracks) {
                if (frameIndex - track.lastSeenFrameIndex > maxGapFrames) {
                    android.util.Log.d("AppearanceTracker", "Closing track after gap. Observations: ${track.observations.size}")
                    finishedAppearances.add(
                        Appearance(
                            startMs = track.observations.first().timestampMs,
                            endMs = track.observations.last().timestampMs,
                            observations = track.observations
                        )
                    )
                } else {
                    stillActive.add(track)
                }
            }
            activeTracks.clear()
            activeTracks.addAll(stillActive)
        }

        // Close anything still open at video end.
        for (track in activeTracks) {
            finishedAppearances.add(
                Appearance(
                    startMs = track.observations.first().timestampMs,
                    endMs = track.observations.last().timestampMs,
                    observations = track.observations
                )
            )
        }

        // Drop tracks that are almost certainly noise (a single blurry whip-pan frame,
        // per the spec's "blurred whip-pan passes count for nobody").
        // REDUCED FOR DEBUGGING: was 2
        return finishedAppearances.filter { it.observations.size >= 1 }
    }

    private fun centerDistance(a: Rect, b: Rect): Float {
        val ax = a.centerX().toFloat(); val ay = a.centerY().toFloat()
        val bx = b.centerX().toFloat(); val by = b.centerY().toFloat()
        return kotlin.math.sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by))
    }
}
