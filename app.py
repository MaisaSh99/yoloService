import os
import uuid
import sqlite3
import traceback
from datetime import datetime
import logging
import hashlib
from typing import Set

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

# Store processed request hashes to prevent duplicates
processed_requests: Set[str] = set()


def get_request_hash(request: Request, file_content: bytes) -> str:
    """Generate a unique hash for the request to prevent duplicates."""
    content_hash = hashlib.md5(file_content).hexdigest()
    return f"{request.client.host}:{content_hash}"


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
        # Read file content first
        file_content = await file.read()

        # Generate request hash
        request_hash = get_request_hash(request, file_content)

        # Check if this request was already processed
        if request_hash in processed_requests:
            logger.info(f"🔄 Duplicate request detected: {request_hash}")
            return JSONResponse(content={"message": "Request already processed"}, status_code=200)

        # Add to processed requests
        processed_requests.add(request_hash)

        # ✅ Robust extraction of header (case-insensitive)
        all_headers = dict(request.headers)
        print("📦 All headers:", all_headers)
        print("📦 Header keys:", list(all_headers.keys()))

        # Try different header variations
        user_id = (
                request.headers.get("X-User-ID") or
                request.headers.get("x-user-id") or
                request.headers.get("X-USER-ID") or
                "unknown"
        )
        print(f"📬 Received X-User-ID: {user_id}")
        print(f"📬 Raw header value: {request.headers.get('X-User-ID')}")

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        original_dir = os.path.join("uploads", "original", user_id)
        predicted_dir = os.path.join("uploads", "predicted", user_id)
        os.makedirs(original_dir, exist_ok=True)
        os.makedirs(predicted_dir, exist_ok=True)

        # Save original image
        original_path = os.path.join(original_dir, f"{timestamp}.jpg")
        with open(original_path, "wb") as f:
            f.write(file_content)

        try:
            # Run YOLO
            results = model(original_path)

            # Save prediction image
            predicted_path = os.path.join(predicted_dir, f"{timestamp}_predicted.jpg")
            annotated = results[0].plot()
            annotated_img = Image.fromarray(annotated)
            annotated_img.save(predicted_path)

            # Upload to S3
            s3 = boto3.client('s3')
            original_s3_key = f"original/{user_id}/{timestamp}.jpg"
            predicted_s3_key = f"predicted/{user_id}/{timestamp}_predicted.jpg"

            s3.upload_file(original_path, bucket_name, original_s3_key)
            print(f"✅ Uploaded original image to S3: {original_s3_key}")

            s3.upload_file(predicted_path, bucket_name, predicted_s3_key)
            print(f"✅ Uploaded predicted image to S3: {predicted_s3_key}")

            # Extract labels
            labels = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                label = model.names[cls_id]
                labels.append(label)

            # Store prediction
            prediction_uid = str(uuid.uuid4())
            save_prediction_session(prediction_uid, original_path, predicted_path)
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                label = model.names[cls_id]
                score = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                save_detection_object(prediction_uid, label, score, bbox)

            return JSONResponse({
                "prediction_uid": prediction_uid,
                "detection_count": len(labels),
                "labels": labels
            })

        except Exception as e:
            logger.error(f"❌ YOLO processing failed: {e}")
            logger.error(traceback.format_exc())
            # Clean up files if they exist
            for path in [original_path, predicted_path]:
                if os.path.exists(path):
                    os.remove(path)
            return JSONResponse(content={"error": str(e)}, status_code=200)  # Return 200 to prevent retries

    except Exception as e:
        logger.error(f"❌ Request processing failed: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=200)  # Return 200 to prevent retries