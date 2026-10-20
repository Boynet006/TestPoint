# MediaPipe (vendored)

Camera background blur / backdrop at the weigh-in desk runs in the browser, offline
(`bgfx-worker.js` runs the model; `index.html` draws the picture).

- `vision_bundle_worker.js` (the package's `vision_bundle.cjs`, renamed so it is served as
  JavaScript) and `wasm/vision_wasm_internal.*`: `@mediapipe/tasks-vision` 0.10.21 (npm), Apache-2.0.
- `selfie_multiclass_256x256.tflite`: MediaPipe multiclass selfie segmenter (float32), from
  `storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/`, Apache-2.0.

Only the picture on screen changes; face recognition always gets the camera's original frame.
