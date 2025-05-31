import boto3
from fastapi import FastAPI, HTTPException, Request, Form, UploadFile, File
from fastapi.responses import FileResponse
from ultralytics import YOLO
from PIL import Image
import sqlite3
import os
import uuid
import traceback
import torch
from datetime import datetime

# Disable GPU usage
torch.cuda.is_available = lambda: False

app = FastAPI()

UPLOAD_DIR = "uploads/original"
PREDICTED_DIR = "uploads/predicted"
DB_PATH = "predictions.db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)

model = YOLO("yolov8n.pt")

bucket_name = os.getenv("S3_BUCKET_NAME")
print("[YOLO] Using S3 bucket:", bucket_name)

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
def predict(file: UploadFile = File(...)):
    print(f"[YOLO] Incoming /predict with file: {file.filename}")

    try:
        s3 = boto3.client('s3')
        ext = os.path.splitext(file.filename)[1] or ".jpg"
        uid = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Get user ID from the request if available
        user_id = request.headers.get('X-User-ID', 'unknown')
        
        original_filename = f"{user_id}/{timestamp}.jpg"
        predicted_filename = f"{user_id}/{timestamp}_predicted.jpg"
        original_path = os.path.join(UPLOAD_DIR, original_filename)
        predicted_path = os.path.join(PREDICTED_DIR, predicted_filename)

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(original_path), exist_ok=True)
        os.makedirs(os.path.dirname(predicted_path), exist_ok=True)

        print(f"[YOLO] Will save original image to: {original_path}")
        print(f"[YOLO] Will save predicted image to: {predicted_path}")

        # Save uploaded file locally
        print(f"[YOLO] Reading uploaded file...")
        file_content = file.file.read()
        print(f"[YOLO] Read {len(file_content)} bytes from uploaded file")

        print(f"[YOLO] Writing file to {original_path}")
        with open(original_path, "wb") as f:
            f.write(file_content)
        print(f"[YOLO] Successfully saved original image")

        # Upload original image to S3
        original_s3_key = f"original/{user_id}/{timestamp}.jpg"
        predicted_s3_key = f"predicted/{user_id}/{timestamp}_predicted.jpg"
        print(f"[YOLO] Uploading original image to s3://{bucket_name}/{original_s3_key}")
        s3.upload_file(original_path, bucket_name, original_s3_key)
        print(f"[YOLO] Successfully uploaded original image to S3")

        # Run YOLO detection
        print(f"[YOLO] Running YOLO detection on {original_path}")
        results = model(original_path, device="cpu")
        print(f"[YOLO] YOLO detection completed")
        
        annotated_frame = results[0].plot()
        annotated_image = Image.fromarray(annotated_frame)
        print(f"[YOLO] Saving predicted image to {predicted_path}")
        annotated_image.save(predicted_path)
        print(f"[YOLO] Successfully saved predicted image")

        save_prediction_session(uid, original_path, predicted_path)

        detected_labels = []
        for box in results[0].boxes:
            label_idx = int(box.cls[0].item())
            label = model.names[label_idx]
            score = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            save_detection_object(uid, label, score, bbox)
            detected_labels.append(label)

        print(f"[YOLO] Uploading predicted image to s3://{bucket_name}/{predicted_s3_key}")
        s3.upload_file(predicted_path, bucket_name, predicted_s3_key)
        print(f"[YOLO] Successfully uploaded predicted image to S3")

        return {
            "prediction_uid": uid,
            "detection_count": len(detected_labels),
            "labels": detected_labels
        }

    except Exception as e:
        print("[YOLO ERROR]", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.get("/prediction/{uid}")
def get_prediction_by_uid(uid: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        session = conn.execute("SELECT * FROM prediction_sessions WHERE uid = ?", (uid,)).fetchone()
        if not session:
            raise HTTPException(status_code=404, detail="Prediction not found")

        objects = conn.execute(
            "SELECT * FROM detection_objects WHERE prediction_uid = ?",
            (uid,)
        ).fetchall()

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
