# 🧠 MRI Brain Tumor Classifier — Render Deploy

FastAPI + ONNX Runtime app for MRI brain tumor classification.

## Classes
`glioma` · `meningioma` · `notumor` · `pituitary`

## ⚠️ Model Files — Git LFS Required

This project uses **Git LFS** for large model files:
- `mri_model.onnx` — ONNX model
- `mri_model.onnx.data` — ONNX external weights (**required**)

### Setup Git LFS (one time):
```bash
git lfs install
git lfs track "*.onnx" "*.data"
git add .gitattributes
```

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET  | `/`        | Web UI |
| GET  | `/health`  | Health check |
| POST | `/predict` | Upload image → prediction + heatmap |

## Render Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `mri_model.onnx` | Path to ONNX model |
| `CLASS_NAMES` | `glioma,meningioma,notumor,pituitary` | Comma-separated class names |

## Local Run
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Deploy on Render
1. Push to GitHub (with Git LFS for model files)
2. New Web Service → Connect GitHub repo
3. Runtime: **Docker**
4. Render auto-detects `render.yaml`
