package com.iykyk.collage

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.iykyk.collage.domain.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

sealed class ProcessingState {
    data object Idle : ProcessingState()
    data class ExtractingFrames(val progress: Float) : ProcessingState()
    data class DetectingFaces(val progress: Float) : ProcessingState()
    data object ClusteringIdentities : ProcessingState()
    data object ComposingCollage : ProcessingState()
    data class Done(val collage: Bitmap, val identities: List<PersonIdentity>) : ProcessingState()
    data class Error(val message: String) : ProcessingState()
}

class ProcessingViewModel(application: Application) : AndroidViewModel(application) {

    private val _state = MutableStateFlow<ProcessingState>(ProcessingState.Idle)
    val state: StateFlow<ProcessingState> = _state

    private val frameExtractor by lazy { FrameExtractor(application) }
    private val faceDetector by lazy { FaceDetectorWrapper() }
    private val faceEmbedderLazy = lazy { FaceEmbedder(application) }
    private val faceEmbedder by faceEmbedderLazy

    // Tuning parameters
    private var trackingThreshold = 0.6f
    private var identityThreshold = 0.5f

    fun reset() {
        _state.value = ProcessingState.Idle
    }

    fun processVideo(videoUri: Uri) {
        // Reset state before starting new process
        _state.value = ProcessingState.Idle
        viewModelScope.launch {
            try {
                // 1. Extract frames (at 4fps instead of 6 for speed)
                val frames = frameExtractor.extractFrames(videoUri, samplesPerSecond = 4) { progress ->
                    _state.value = ProcessingState.ExtractingFrames(progress)
                }
                android.util.Log.d("ProcessingViewModel", "Extracted ${frames.size} frames")
                if (frames.isEmpty()) {
                    _state.value = ProcessingState.Error("No frames could be extracted from this video.")
                    return@launch
                }

                // 2. Detect + embed every frame
                val perFrameObservations = withContext(Dispatchers.Default) {
                    var totalFaces = 0
                    val results = frames.mapIndexed { index, frame ->
                        _state.value = ProcessingState.DetectingFaces((index + 1f) / frames.size)
                        val rawFaces = faceDetector.detect(frame)
                        totalFaces += rawFaces.size
                        rawFaces.map { raw ->
                            FaceObservation(
                                timestampMs = raw.timestampMs,
                                bbox = raw.bbox,
                                embedding = faceEmbedder.embed(raw.sourceFrame, raw.bbox),
                                headEulerAngleY = raw.headEulerAngleY,
                                headEulerAngleZ = raw.headEulerAngleZ,
                                leftEyeOpenProb = raw.leftEyeOpenProb,
                                rightEyeOpenProb = raw.rightEyeOpenProb,
                                smilingProb = raw.smilingProb,
                                sharpness = ShotScorer.computeSharpness(raw.sourceFrame, raw.bbox),
                                sourceFrame = raw.sourceFrame
                            )
                        }
                    }
                    android.util.Log.d("ProcessingViewModel", "Total faces detected across all frames: $totalFaces")

                    // Added debug logging
                    android.util.Log.d("DetectDebug", "Total frames: ${frames.size}")
                    results.forEachIndexed { i, faces ->
                        if (faces.isNotEmpty()) {
                            android.util.Log.d("DetectDebug", "frame $i (${frames[i].timestampMs}ms): ${faces.size} faces, sizes=${faces.map { it.bbox.width() }}")
                        }
                    }

                    results
                }

                // 3. Track continuous appearances, then cluster appearances into identities
                _state.value = ProcessingState.ClusteringIdentities
                val (frameW, frameH) = frames.first().bitmap.let { it.width to it.height }
                val appearances = withContext(Dispatchers.Default) {
                    AppearanceTracker(similarityThreshold = trackingThreshold).track(perFrameObservations, frameW, frameH)
                }
                val identities = withContext(Dispatchers.Default) {
                    val idents = IdentityClusterer(similarityThreshold = identityThreshold).cluster(appearances)
                    android.util.Log.d("ProcessingViewModel", 
                        "Clustering complete. ${idents.size} identities: ${idents.map { it.appearanceCount }}")
                    idents
                }

                // 4. Compose collage
                _state.value = ProcessingState.ComposingCollage
                val collage = withContext(Dispatchers.Default) {
                    CollageComposer.composeCollage(identities)
                }

                _state.value = ProcessingState.Done(collage, identities)
            } catch (e: Exception) {
                _state.value = ProcessingState.Error(e.message ?: "Unknown error during processing")
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        if (faceEmbedderLazy.isInitialized()) {
            faceEmbedder.close()
        }
    }
}
