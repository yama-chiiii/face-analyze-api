from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ultralytics import YOLO
import cv2
import numpy as np
import time
import os

app = FastAPI()

model = YOLO("face_yolov8n.pt")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("captured", exist_ok=True)

@app.post("/analyze")
async def analyze_face(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"status": "error", "message": "画像の読み込みに失敗しました"}

    timestamp = int(time.time() * 1000)
    filename = f"captured_{timestamp}.jpg"
    filepath = f"captured/{filename}"
    cv2.imwrite(filepath, frame)

    results = model(frame)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        if label == "face":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append({
                "label": label,
                "confidence": conf,
                "box": [x1, y1, x2, y2]
            })

    return {
        "status": "success",
        "result": {
            "detections": detections,
            "image_url": f"http://localhost:8000/image/{filename}"
        }
    }

@app.get("/image/{filename}")
def get_image(filename: str):
    return FileResponse(path=f"captured/{filename}", media_type="image/jpeg")
