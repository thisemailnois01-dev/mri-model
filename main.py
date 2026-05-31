"""FastAPI MRI Classifier — ONNX Runtime, Railway ready"""
import io, base64, os
from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.getenv("MODEL_PATH", "mri_model.onnx")
IMG_SIZE    = 384
MEAN        = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD         = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_NAMES = os.getenv("CLASS_NAMES", "glioma,meningioma,notumor,pituitary").split(",")

# ── Load ONNX session ─────────────────────────────────────────────────────────
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
session   = ort.InferenceSession(MODEL_PATH, providers=providers)
INPUT_NAME  = session.get_inputs()[0].name
OUTPUT_NAME = session.get_outputs()[0].name
DEVICE_USED = "cuda" if "CUDAExecutionProvider" in session.get_providers() else "cpu"
print(f"ONNX model loaded | classes={CLASS_NAMES} | provider={session.get_providers()[0]}")

# ── Pre-processing ────────────────────────────────────────────────────────────
def preprocess(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 → float32 NCHW tensor normalised with ImageNet stats."""
    img = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    img = (img - MEAN) / STD                      # HWC normalise
    img = img.transpose(2, 0, 1)[np.newaxis]      # NCHW
    return np.ascontiguousarray(img, dtype=np.float32)

# ── Simple saliency map (GradCAM-lite via input sensitivity) ──────────────────
def saliency_map(rgb: np.ndarray, pred_idx: int) -> np.ndarray:
    """
    Occlusion-free saliency: resize image to 24×24 grid,
    measure logit drop when each patch is greyed out → heatmap.
    Pure NumPy/ONNX, no autograd needed.
    """
    G = 24
    base_inp = preprocess(rgb)
    base_logit = session.run([OUTPUT_NAME], {INPUT_NAME: base_inp})[0][0, pred_idx]

    heat = np.zeros((G, G), dtype=np.float32)
    patch_h = IMG_SIZE // G
    patch_w = IMG_SIZE // G
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))

    for r in range(G):
        row_data = []
        for c in range(G):
            masked = resized.copy()
            masked[r*patch_h:(r+1)*patch_h, c*patch_w:(c+1)*patch_w] = 128
            row_data.append(preprocess(masked))
        batch = np.concatenate(row_data, axis=0)
        logits = session.run([OUTPUT_NAME], {INPUT_NAME: batch})[0][:, pred_idx]
        heat[r] = base_logit - logits          # drop = importance

    heat = np.clip(heat, 0, None)
    mn, mx = heat.min(), heat.max()
    heat = (heat - mn) / (mx - mn + 1e-8)
    return heat

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="MRI Brain Tumor Classifier", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

@app.get("/", response_class=HTMLResponse)
async def root():
    p = Path("templates/index.html")
    return p.read_text() if p.exists() else "<h1>MRI Classifier API</h1><a href='/docs'>Docs</a>"

@app.get("/health")
async def health():
    return {
        "status":    "ok",
        "classes":   CLASS_NAMES,
        "provider":  session.get_providers()[0],
        "model":     MODEL_PATH,
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Need an image file.")
    try:
        data = await file.read()
        arr  = np.frombuffer(data, np.uint8)
        bgr  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(400, "Cannot decode image.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # ── Inference ─────────────────────────────────────────────────────────
        inp    = preprocess(rgb)
        logits = session.run([OUTPUT_NAME], {INPUT_NAME: inp})[0][0]
        probs  = softmax(logits)
        pidx   = int(np.argmax(probs))
        cls    = CLASS_NAMES[pidx]

        # ── Saliency overlay ──────────────────────────────────────────────────
        cam  = saliency_map(rgb, pidx)
        h, w = rgb.shape[:2]
        camr = cv2.resize(cam, (w, h))
        heat = cv2.cvtColor(
            cv2.applyColorMap(np.uint8(255 * camr), cv2.COLORMAP_JET),
            cv2.COLOR_BGR2RGB
        )
        over = cv2.addWeighted(rgb, 0.55, heat, 0.45, 0)

        def b64(img):
            _, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return base64.b64encode(buf).decode()

        tumor = cls.lower() not in ["notumor", "no_tumor", "no tumor", "normal"]
        return JSONResponse({
            "prediction":          cls,
            "is_tumor":            tumor,
            "confidence":          float(probs[pidx]),
            "confidence_pct":      f"{probs[pidx]*100:.1f}%",
            "all_probabilities":   {c: float(p) for c, p in zip(CLASS_NAMES, probs)},
            "original_b64":        b64(rgb),
            "heatmap_b64":         b64(heat),
            "gradcam_overlay_b64": b64(over),
            "explanation": (
                f"The model detected {cls.upper()} with "
                f"{probs[pidx]*100:.1f}% confidence. "
                f"Warm regions in the saliency map highlight "
                f"the areas the model focused on."
            ),
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
