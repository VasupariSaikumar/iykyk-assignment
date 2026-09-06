package com.iykyk.collage

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.iykyk.collage.domain.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.util.concurrent.atomic.AtomicInteger

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
                // 1. Extract frames (6fps back for accuracy, now parallelized)
                val frames = frameExtractor.extractFrames(videoUri, samplesPerSecond = 6) { progress ->
                    _state.value = ProcessingState.ExtractingFrames(progress)
                }
                android.util.Log.d("ProcessingViewModel", "Extracted ${frames.size} frames")
                if (frames.isEmpty()) {
                    _state.value = ProcessingState.Error("No frames could be extracted from this video.")
                    return@launch
                }

                // 2. Detect + embed every frame in PARALLEL
                val perFrameObservations = withContext(Dispatchers.Default) {
                    val progressCount = AtomicInteger(0)
                    val deferredResults = frames.map { frame ->
                        async {
                            val rawFaces = faceDetector.detect(frame)
                            val observations = rawFaces.map { raw ->
                                // Pre-processing (cropping) happens in parallel
                                // Inference is synchronized to prevent interpreter conflicts
                                val embedding = synchronized(faceEmbedder) {
                                    faceEmbedder.embed(raw.sourceFrame, raw.bbox)
                                }
                                FaceObservation(
                                    timestampMs = raw.timestampMs,
                                    bbox = raw.bbox,
                                    embedding = embedding,
                                    headEulerAngleY = raw.headEulerAngleY,
                                    headEulerAngleZ = raw.headEulerAngleZ,
                                    leftEyeOpenProb = raw.leftEyeOpenProb,
                                    rightEyeOpenProb = raw.rightEyeOpenProb,
                                    smilingProb = raw.smilingProb,
                                    sharpness = ShotScorer.computeSharpness(raw.sourceFrame, raw.bbox),
                                    sourceFrame = raw.sourceFrame
                                )
                            }
                            val current = progressCount.incrementAndGet()
                            _state.value = ProcessingState.DetectingFaces(current.toFloat() / frames.size)
                            observations
                        }
                    }
                    deferredResults.awaitAll()
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
