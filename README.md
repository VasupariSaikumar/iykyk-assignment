# iykyk Assignment — Video-based Unique-Person Collage

## Pipeline
1. **Frame extraction** (`FrameExtractor`) — samples the source video at 6fps via
   `MediaMetadataRetriever`, off the main thread.
2. **Face detection** (`FaceDetectorWrapper`) — ML Kit, accurate mode, with classification
   enabled for eyes-open/smiling attributes.
3. **Embedding** (`FaceEmbedder`) — TFLite FaceNet-style model, 160x160 input, 128-d
   L2-normalized output. **Model used:** `facenet.tflite` — [DOCUMENT THE EXACT SOURCE
   YOU USED HERE, e.g. shubham0204/FaceRecognition_With_FaceNet_Android]. Place it in
   `app/src/main/assets/facenet.tflite`.
4. **Appearance tracking** (`AppearanceTracker`) — links per-frame detections into
   continuous visible segments using embedding similarity (threshold `0.62`) + spatial
   proximity, closing a track after 3 consecutive missed frames.
5. **Identity clustering** (`IdentityClusterer`) — incremental centroid clustering of
   appearances into people, cosine similarity threshold **`0.55`** [UPDATE THIS to
   whatever you land on after testing against Sample 1].
6. **Representative shot selection** (`ShotScorer`) — weighted score of frontality (35%),
   sharpness via Laplacian variance (35%), eyes-open (20%), smiling (10%).
7. **Collage** (`CollageComposer`) — generous (non-tight) crop per person, rounded tiles,
   grid layout with appearance count labels.

## Threshold tuning notes
[FILL IN: what you actually observed on Sample 1 (5 people, 20 appearances total) before
and after tuning, and what final threshold values you settled on for both the tracker and
the clusterer. This is explicitly requested in the assignment and matters for grading.]

## Build
1. Open in Android Studio (Iguana+), let Gradle sync.
2. Drop `facenet.tflite` into `app/src/main/assets/`.
3. Run on a device/emulator with API 26+.

## Known limitations / next steps if more time were available
- Appearance tracking assumes one continuous shot; a hard cut immediately followed by the
  same person could be logged as a new appearance (matches spec's definition, but worth
  flagging as a modeling choice).
- Clustering is incremental/greedy rather than full agglomerative — faster and simpler for
  a 12-15h scope, at some cost to accuracy on borderline similarity scores.
- Collage layout is a simple grid; a true Instagram Story-style asymmetric layout was
  deprioritized in favor of pipeline accuracy given the time budget.
