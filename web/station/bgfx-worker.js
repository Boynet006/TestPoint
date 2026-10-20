/* Camera background: the person segmentation model, off the page's main thread.
 *
 * The page sends a small ImageBitmap of each camera frame; this worker runs MediaPipe's
 * multiclass selfie segmenter (GPU through OffscreenCanvas WebGL, CPU as fallback) and sends
 * back the person confidence (0..1) as a transferable Float32Array. The page composites
 * every camera frame with the latest mask, so the model's ~25 ms never blocks the UI.
 */
/* global importScripts */
self.exports = {};
self.module = { exports: self.exports };
importScripts("vendor/mediapipe/vision_bundle_worker.js");   // the CommonJS build, renamed .js so it is served as JavaScript
const { ImageSegmenter } = self.exports;

let seg = null;
const ready = (async () => {
  const base = new URL("vendor/mediapipe/", self.location.href).href;
  const files = { wasmLoaderPath: base + "wasm/vision_wasm_internal.js", wasmBinaryPath: base + "wasm/vision_wasm_internal.wasm" };
  const opts = delegate => ({
    baseOptions: { modelAssetPath: base + "selfie_multiclass_256x256.tflite", delegate },
    runningMode: "VIDEO", outputConfidenceMasks: true, outputCategoryMask: false,
  });
  try {
    seg = await ImageSegmenter.createFromOptions(files, opts("GPU"));
  } catch {
    seg = await ImageSegmenter.createFromOptions(files, opts("CPU"));
  }
})();

let lastTs = 0;
self.onmessage = async ({ data }) => {
  const { frame, ts } = data;
  try {
    await ready;
    const t = Math.max(ts, lastTs + 1);
    lastTs = t;
    let out = null;
    seg.segmentForVideo(frame, t, r => {
      const masks = r.confidenceMasks;
      if (!masks || !masks.length) return;
      // multiclass: 0 background, 1 hair, 2 body skin, 3 face skin, 4 clothes, 5 others (accessories).
      // The person is 1-4: "others" also picks up things touching them (a towel, a chair back).
      const m = masks[0], n = m.width * m.height, person = new Float32Array(n);
      if (masks.length > 2) {
        for (const k of [1, 2, 3, 4]) { const c = masks[k].getAsFloat32Array(); for (let i = 0; i < n; i++) person[i] += c[i]; }
      } else person.set(masks[masks.length - 1].getAsFloat32Array());
      out = { mask: person, w: m.width, h: m.height };
    });
    if (out) self.postMessage(out, [out.mask.buffer]);
    else self.postMessage({ mask: null });
  } catch (e) {
    self.postMessage({ error: String(e && e.message || e) });
  } finally {
    frame.close();
  }
};
