from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ultralytics import YOLO
from deepface import DeepFace
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
    for file_name in os.listdir("captured"):
        if file_name.endswith(".jpg"):
            try:
                os.remove(os.path.join("captured", file_name))
            except Exception as e:
                print(f"削除失敗: {file_name} - {e}")

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"status": "error", "message": "画像の読み込みに失敗しました"}

    results = model(frame)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        detections.append({
            "label": label,
            "confidence": conf,
            "box": [x1, y1, x2, y2]
        })

    timestamp = int(time.time() * 1000)
    filename = f"captured_{timestamp}.jpg"
    filepath = f"captured/{filename}"
    cv2.imwrite(filepath, frame)
    cv2.imwrite("captured/latest.jpg", frame)

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

@app.get("/diagnose")
def diagnose_latest_image():
    image_path = "captured/latest.jpg"
    if not os.path.exists(image_path):
        return {"status": "error", "message": "latest.jpg が見つかりません"}

    frame = cv2.imread(image_path)

    results = model(frame)[0]
    if not results.boxes:
        return {"status": "error", "message": "顔が検出できませんでした"}

    (x1, y1, x2, y2) = map(int, results.boxes[0].xyxy[0])
    face_img = frame[y1:y2, x1:x2]

    try:
        analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
        dominant_emotion = str(analysis[0]['dominant_emotion'])  # numpy型でないことを保証
        emotions_raw = analysis[0]['emotion']

        # numpy.float32 を float に変換
        emotions = {k: float(v) for k, v in emotions_raw.items()}

        return {
            "status": "success",
            "emotion": dominant_emotion,
            "emotions": emotions
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"感情分析に失敗しました: {str(e)}"
        }
