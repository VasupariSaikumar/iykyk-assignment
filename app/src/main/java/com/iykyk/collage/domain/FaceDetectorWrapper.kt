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
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
        .setMinFaceSize(0.15f)
        .build()

    private val detector = FaceDetection.getClient(options)

    /**
      * Runs detection on a single frame.
     */
    suspend fun detect(frame: SampledFrame): List<RawFaceDetection> = withContext(Dispatchers.Default) {
        // Fast path: Scale down for detection. 
        // ML Kit performs best when faces are ~100-200 pixels.
        val targetSize = 480
        val scale = if (frame.bitmap.width > targetSize) targetSize.toFloat() / frame.bitmap.width else 1.0f
        
        val detectionBitmap = if (scale < 1.0f) {
            Bitmap.createScaledBitmap(
                frame.bitmap,
                (frame.bitmap.width * scale).toInt(),
                (frame.bitmap.height * scale).toInt(),
                true
            )
        } else {
            frame.bitmap
        }

        val image = InputImage.fromBitmap(detectionBitmap, 0)
        val faces = try {
            Tasks.await(detector.process(image))
        } catch (e: Exception) {
            android.util.Log.e("FaceDetectorWrapper", "ML Kit Error: ${e.message}", e)
            emptyList()
        }

        faces.map { face ->
            val box = if (scale < 1.0f) {
                Rect(
                    (face.boundingBox.left / scale).toInt(),
                    (face.boundingBox.top / scale).toInt(),
                    (face.boundingBox.right / scale).toInt(),
                    (face.boundingBox.bottom / scale).toInt()
                )
            } else {
                face.boundingBox
            }

            RawFaceDetection(
                timestampMs = frame.timestampMs,
                bbox = box,
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
