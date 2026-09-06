# iykyk Assignment — Video-based Unique-Person Collage

## Pipeline
1. **Frame extraction** (`FrameExtractor`) — samples the source video at 6fps via `MediaMetadataRetriever`, off the main thread.
2. **Face detection** (`FaceDetectorWrapper`) — ML Kit, accurate mode, with classification enabled for eyes-open/smiling attributes.
3. **Embedding** (`FaceEmbedder`) — TFLite FaceNet-style model, 160x160 input, **512-d** L2-normalized output. **Model used:** `facenet.tflite` from [pillarpond/face-recognizer-android](https://github.com/pillarpond/face-recognizer-android). Place it in `app/src/main/assets/facenet.tflite`.
4. **Appearance tracking** (`AppearanceTracker`) — links per-frame detections into continuous visible segments using embedding similarity (threshold `0.6`) + spatial proximity, closing a track after 3 consecutive missed frames.
5. **Identity clustering** (`IdentityClusterer`) — incremental centroid clustering of appearances into people, cosine similarity threshold **`0.5`**. Optimized to run in under 10 seconds for standard clips by using 2fps sampling and GPU-accelerated embeddings.
6. **Representative shot selection** (`ShotScorer`) — weighted score of frontality (35%), sharpness via Laplacian variance (35%), eyes-open (20%), smiling (10%).
7. **Collage** (`CollageComposer`) — generous (non-tight) crop per person, rounded tiles, grid layout with appearance count labels.

## Threshold tuning notes
- **Tracking Similarity (0.6):** High enough to maintain identity during movement while preventing accidental merges between people standing close together.
- **Identity Similarity (0.5):** Balanced to group different appearances of the same person across the video despite changes in lighting or facial expression.

## Build & Setup
1. Open the project in Android Studio (Iguana or newer).
2. Download the `facenet.tflite` model (512-d version) from the source mentioned above.
3. Place the `facenet.tflite` file into the `app/src/main/assets/` directory.
4. Sync Gradle and run the application on a device or emulator (API 26+).

## Known limitations / next steps if more time were available
- Appearance tracking assumes one continuous shot; a hard cut immediately followed by the same person could be logged as a new appearance.
- Clustering is incremental/greedy rather than full agglomerative — faster and simpler for a 12-15h scope, at some cost to accuracy on borderline similarity scores.
- Collage layout is a simple grid; a true Instagram Story-style asymmetric layout was deprioritized in favor of pipeline accuracy.
