from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import cv2
import numpy as np
import time

app = FastAPI()

# 顔検出器の読み込み
cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開発中は * でOK
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
def analyze_face_color():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return {"status": "error", "message": "カメラを開けませんでした"}

    # 連続でフレームを読み込んでから使う（最初の数フレームは不安定）
    for _ in range(5):
        ret, frame = cap.read()
    cap.release()

    if not ret:
        return {"status": "error", "message": "画像取得失敗"}

    # ファイル名をタイムスタンプで一意に
    timestamp = int(time.time() * 1000)
    filename = f"captured_{timestamp}.jpg"
    filepath = f"captured/{filename}"  # 保存先フォルダ

    # 保存
    cv2.imwrite(filepath, frame)

    # 顔検出
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return {"status": "error", "message": "顔が見つかりませんでした"}

    # 顔の赤み評価
    (x, y, w, h) = faces[0]
    face_roi = frame[y:y + h, x:x + w]
    avg_color = np.mean(face_roi, axis=(0, 1))
    r = avg_color[2]
    face_status = "良い" if r > 120 else "普通" if r > 90 else "悪い"

    return {
        "status": "success",
        "result": {
            "color_score": int(r),
            "face_condition": face_status,
            "image_url": f"http://localhost:5000/image/{filename}"
        }
    }


@app.get("/image/{filename}")
def get_image(filename: str):
    return FileResponse(path=f"captured/{filename}", media_type="image/jpeg")
