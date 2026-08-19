"""
GreenCycle Sorting-Line Scanner — FastAPI backend.

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000

Expects two files trained in the Colab notebook to sit next to this file:
    wastenet_final.pth   (model weights)
    class_names.json     (ordered list of class names, matching training)
"""
import io
import json
import os

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms

from model_def import WasteNet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "wastenet_final.pth")
CLASSES_PATH = os.path.join(BASE_DIR, "class_names.json")
IMG_SIZE = 160

HAZARD_CLASSES = {"battery"}

app = FastAPI(title="GreenCycle Sorting-Line Scanner")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(WEIGHTS_PATH) or not os.path.exists(CLASSES_PATH):
    raise RuntimeError(
        "Missing wastenet_final.pth or class_names.json. "
        "Copy both files into this folder before starting the server."
    )

with open(CLASSES_PATH) as f:
    CLASS_NAMES = json.load(f)

model = WasteNet(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.to(device)
model.eval()

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device), "classes": CLASS_NAMES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    x = eval_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0].cpu().tolist()

    scores = {cls: round(p, 4) for cls, p in zip(CLASS_NAMES, probs)}
    top_class = max(scores, key=scores.get)
    top_confidence = scores[top_class]
    is_hazard = top_class in HAZARD_CLASSES

    return JSONResponse({
        "predicted_class": top_class,
        "confidence": top_confidence,
        "scores": scores,
        "hazard_flag": is_hazard,
        "hazard_message": "CONTAMINANT DETECTED — DIVERT FROM STREAM" if is_hazard else None,
    })


app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
