package com.iykyk.collage.domain

import android.graphics.Bitmap
import android.graphics.Rect
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/** One face detected in one sampled frame, before embedding/tracking. */
data class RawFaceDetection(
    val timestampMs: Long,
    val bbox: Rect,
    val headEulerAngleY: Float,   // yaw -- 0 = looking straight at camera
    val headEulerAngleZ: Float,   // roll
    val leftEyeOpenProb: Float?,
    val rightEyeOpenProb: Float?,
    val smilingProb: Float?,
    val sourceFrame: Bitmap       // full frame, NOT cropped -- crop generously later, per spec
)

class FaceDetectorWrapper {

    private val options = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
        .setMinFaceSize(0.1f)
        .enableTracking()
        .build()

    private val detector = FaceDetection.getClient(options)

    /**
      * Runs detection on a single frame. Blocking-style via Tasks.await, called from
     * Dispatchers.Default -- never call this from the main thread.
     */
    suspend fun detect(frame: SampledFrame): List<RawFaceDetection> = withContext(Dispatchers.Default) {
        val image = InputImage.fromBitmap(frame.bitmap, 0)
        val faces = try {
            Tasks.await(detector.process(image))
        } catch (e: Exception) {
            android.util.Log.e("FaceDetectorWrapper", "ML Kit Error: ${e.message}", e)
            emptyList()
        }

        if (faces.isNotEmpty()) {
            android.util.Log.d("FaceDetectorWrapper", "Detected ${faces.size} faces at ${frame.timestampMs}ms")
        }

        faces.map { face ->
            RawFaceDetection(
                timestampMs = frame.timestampMs,
                bbox = face.boundingBox,
                headEulerAngleY = face.headEulerAngleY,
                headEulerAngleZ = face.headEulerAngleZ,
                leftEyeOpenProb = face.leftEyeOpenProbability,
                rightEyeOpenProb = face.rightEyeOpenProbability,
                smilingProb = face.smilingProbability,
                sourceFrame = frame.bitmap
            )
        }
    }
}
