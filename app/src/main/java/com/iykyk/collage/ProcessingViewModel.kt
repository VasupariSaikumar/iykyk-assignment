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
    data class ReadingVideo(val progress: Float) : ProcessingState()
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

    // Tuning parameters: 0.75 is balanced for FaceNet separation and variation.
    private var trackingThreshold = 0.70f
    private var identityThreshold = 0.75f
    private val MIN_SHARPNESS = 15.0f // Strict filter to ensure high-quality collage

    fun reset() {
        _state.value = ProcessingState.Idle
    }

    fun processVideo(videoUri: Uri) {
        android.util.Log.d("ProcessingViewModel", "processVideo called with URI: $videoUri")
        _state.value = ProcessingState.Idle
        viewModelScope.launch {
            try {
                // 1 & 2. Accurate, Parallel, Memory-Safe Processing
                _state.value = ProcessingState.ReadingVideo(0f)
                val perFrameObservations = frameExtractor.processFramesParallel(
                    videoUri, 
                    samplesPerSecond = 2, // 2 unique frames per second is plenty with accurate extraction
                    onProgress = { _state.value = ProcessingState.ReadingVideo(it) }
                ) { frame ->
                    val rawFaces = faceDetector.detect(frame)
                    rawFaces.mapNotNull { raw ->
                        val sharpness = ShotScorer.computeSharpness(raw.sourceFrame, raw.bbox)
                        if (sharpness < MIN_SHARPNESS) return@mapNotNull null
                        
                        val embedding = synchronized(faceEmbedder) {
                            faceEmbedder.embed(raw.sourceFrame, raw.bbox)
                        }
                        
                        // Capture crop immediately so the large source frame can be recycled by the worker
                        val faceCrop = CollageComposer.generousCrop(raw.sourceFrame, raw.bbox)
                        
                        FaceObservation(
                            timestampMs = raw.timestampMs,
                            bbox = raw.bbox,
                            embedding = embedding,
                            headEulerAngleY = raw.headEulerAngleY,
                            headEulerAngleZ = raw.headEulerAngleZ,
                            leftEyeOpenProb = raw.leftEyeOpenProb,
                            rightEyeOpenProb = raw.rightEyeOpenProb,
                            smilingProb = raw.smilingProb,
                            sharpness = sharpness,
                            faceCrop = faceCrop 
                        )
                    }
                }

                val allObservations = perFrameObservations.flatten()
                if (allObservations.isEmpty()) {
                    _state.value = ProcessingState.Error("No high-quality faces found. Try a clearer video.")
                    return@launch
                }

                // 3. Track continuous appearances, then cluster appearances into identities
                _state.value = ProcessingState.ClusteringIdentities
                
                // Get frame dimensions from the first observation (or metadata if needed)
                val frameW = 1080; val frameH = 1920 // Fallback defaults, tracker is robust to scale

                val appearances = withContext(Dispatchers.Default) {
                    AppearanceTracker(similarityThreshold = trackingThreshold).track(perFrameObservations, frameW, frameH)
                }
                val identities = withContext(Dispatchers.Default) {
                    val idents = IdentityClusterer(similarityThreshold = identityThreshold).cluster(appearances)
                    android.util.Log.d("ProcessingViewModel", "Clustering complete. ${idents.size} identities.")
                    idents
                }

                // 4. Compose collage
                _state.value = ProcessingState.ComposingCollage
                val collage = withContext(Dispatchers.Default) {
                    CollageComposer.composeCollage(identities)
                }

                _state.value = ProcessingState.Done(collage, identities)
            } catch (e: Exception) {
                android.util.Log.e("ProcessingViewModel", "Error during processing", e)
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
