import os
import uuid
import sqlite3
import traceback
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from ultralytics import YOLO
from PIL import Image
import boto3
import torch

# Disable GPU
torch.cuda.is_available = lambda: False

app = FastAPI()

UPLOAD_DIR = "uploads/original"
PREDICTED_DIR = "uploads/predicted"
DB_PATH = "predictions.db"
bucket_name = os.getenv('S3_BUCKET_NAME') or 'maisa-polybot-images'
print("[YOLO] Using S3 bucket:", bucket_name)

logger = logging.getLogger(__name__)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)

model = YOLO("yolov8n.pt")
predictions = {}

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_sessions (
                uid TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                original_image TEXT,
                predicted_image TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detection_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_uid TEXT,
                label TEXT,
                score REAL,
                box TEXT,
                FOREIGN KEY (prediction_uid) REFERENCES prediction_sessions (uid)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prediction_uid ON detection_objects (prediction_uid)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_label ON detection_objects (label)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_score ON detection_objects (score)")

init_db()

def save_prediction_session(uid, original_image, predicted_image):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO prediction_sessions (uid, original_image, predicted_image)
            VALUES (?, ?, ?)
        """, (uid, original_image, predicted_image))

def save_detection_object(prediction_uid, label, score, box):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO detection_objects (prediction_uid, label, score, box)
            VALUES (?, ?, ?, ?)
        """, (prediction_uid, label, score, str(box)))

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        print("📦 All headers:", dict(request.headers))
        headers = {k.lower(): v for k, v in request.headers.items()}
        user_id = headers.get("x-user-id", "unknown")
        print(f"📬 Received X-User-ID: {user_id}")

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        original_dir = os.path.join("uploads", "original", user_id)
        predicted_dir = os.path.join("uploads", "predicted", user_id)
        os.makedirs(original_dir, exist_ok=True)
        os.makedirs(predicted_dir, exist_ok=True)

        original_path = os.path.join(original_dir, f"{timestamp}.jpg")
        with open(original_path, "wb") as f:
            f.write(await file.read())

        results = model(original_path)

        predicted_path = os.path.join(predicted_dir, f"{timestamp}_predicted.jpg")
        annotated = results[0].plot()
        annotated_img = Image.fromarray(annotated)
        annotated_img.save(predicted_path)

        s3 = boto3.client('s3')
        original_s3_key = f"original/{user_id}/{timestamp}.jpg"
        predicted_s3_key = f"predicted/{user_id}/{timestamp}_predicted.jpg"

        s3.upload_file(original_path, bucket_name, original_s3_key)
        print(f"✅ Uploaded original image to S3: {original_s3_key}")
        s3.upload_file(predicted_path, bucket_name, predicted_s3_key)
        print(f"✅ Uploaded predicted image to S3: {predicted_s3_key}")

        labels = []
        prediction_uid = str(uuid.uuid4())
        save_prediction_session(prediction_uid, original_path, predicted_path)

        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            label = model.names[cls_id]
            labels.append(label)
            score = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            save_detection_object(prediction_uid, label, score, bbox)

        return JSONResponse({
            "prediction_uid": prediction_uid,
            "detection_count": len(labels),
            "labels": labels
        })

    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/prediction/{uid}")
def get_prediction_by_uid(uid: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        session = conn.execute("SELECT * FROM prediction_sessions WHERE uid = ?", (uid,)).fetchone()
        if not session:
            raise HTTPException(status_code=404, detail="Prediction not found")

        objects = conn.execute("SELECT * FROM detection_objects WHERE prediction_uid = ?", (uid,)).fetchall()
        return {
            "uid": session["uid"],
            "timestamp": session["timestamp"],
            "original_image": session["original_image"],
            "predicted_image": session["predicted_image"],
            "detection_objects": [
                {
                    "id": obj["id"],
                    "label": obj["label"],
                    "score": obj["score"],
                    "box": obj["box"]
                } for obj in objects
            ]
        }

@app.get("/predictions/label/{label}")
def get_predictions_by_label(label: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT DISTINCT ps.uid, ps.timestamp
            FROM prediction_sessions ps
            JOIN detection_objects do ON ps.uid = do.prediction_uid
            WHERE do.label = ?
        """, (label,)).fetchall()

        return [{"uid": row["uid"], "timestamp": row["timestamp"]} for row in rows]

@app.get("/predictions/score/{min_score}")
def get_predictions_by_score(min_score: float):
    if not (0.0 <= min_score <= 1.0):
        raise HTTPException(status_code=400, detail="Score must be between 0.0 and 1.0")

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT DISTINCT ps.uid, ps.timestamp
            FROM prediction_sessions ps
            JOIN detection_objects do ON ps.uid = do.prediction_uid
            WHERE do.score >= ?
        """, (min_score,)).fetchall()

        return [{"uid": row["uid"], "timestamp": row["timestamp"]} for row in rows]

@app.get("/image/{type}/{filename}")
def get_image(type: str, filename: str):
    if type not in ["original", "predicted"]:
        raise HTTPException(status_code=400, detail="Invalid image type")
    path = os.path.join("uploads", type, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)

@app.get("/prediction/{uid}/image")
def get_prediction_image(uid: str, request: Request):
    accept = request.headers.get("accept", "")
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT predicted_image FROM prediction_sessions WHERE uid = ?", (uid,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Prediction not found")
        image_path = row[0]

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Predicted image file not found")

    if "image/png" in accept:
        return FileResponse(image_path, media_type="image/png")
    elif "image/jpeg" in accept or "image/jpg" in accept:
        return FileResponse(image_path, media_type="image/jpeg")
    else:
        raise HTTPException(status_code=406, detail="Client does not accept an image format")

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
