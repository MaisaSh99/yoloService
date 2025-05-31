import boto3
from fastapi import FastAPI, HTTPException, Request, Form, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from ultralytics import YOLO
from PIL import Image
import sqlite3
import os
import uuid
import traceback
import torch
from datetime import datetime
import logging

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

logger = logging.getLogger(__name__)

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
async def predict(file: UploadFile = File(...)):
    try:
        logger.info("📥 Received prediction request")
        logger.info(f"📄 File info: {file.filename}, {file.content_type}")

        # Get user ID from headers
        user_id = request.headers.get('X-User-ID', 'unknown')
        logger.info(f"👤 User ID from headers: {user_id}")

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        logger.info(f"⏰ Generated timestamp: {timestamp}")

        # Create filenames with user ID and timestamp
        original_filename = f"{user_id}/{timestamp}.jpg"
        predicted_filename = f"{user_id}/{timestamp}_predicted.jpg"
        logger.info(f"📂 Original filename: {original_filename}")
        logger.info(f"📂 Predicted filename: {predicted_filename}")

        # Create directories if they don't exist
        original_dir = os.path.join(UPLOAD_DIR, 'original', user_id)
        predicted_dir = os.path.join(UPLOAD_DIR, 'predicted', user_id)
        logger.info(f"📁 Creating directories if they don't exist:")
        logger.info(f"   - Original: {original_dir}")
        logger.info(f"   - Predicted: {predicted_dir}")
        
        os.makedirs(original_dir, exist_ok=True)
        os.makedirs(predicted_dir, exist_ok=True)

        # Save original file
        original_path = os.path.join(UPLOAD_DIR, 'original', original_filename)
        logger.info(f"💾 Saving original file to: {original_path}")
        
        # Read file content
        file_content = await file.read()
        logger.info(f"📥 Read {len(file_content)} bytes from uploaded file")
        
        # Save file
        with open(original_path, 'wb') as f:
            f.write(file_content)
        logger.info(f"✅ Original file saved successfully")

        # Process with YOLO
        logger.info("🔍 Processing image with YOLO")
        results = model(original_path)
        logger.info(f"✅ YOLO processing complete. Results: {results}")

        # Save predicted image
        predicted_path = os.path.join(UPLOAD_DIR, 'predicted', predicted_filename)
        logger.info(f"💾 Saving predicted image to: {predicted_path}")
        results.save(predicted_path)
        logger.info(f"✅ Predicted image saved successfully")

        # Upload to S3
        original_s3_key = f"original/{user_id}/{timestamp}.jpg"
        predicted_s3_key = f"predicted/{user_id}/{timestamp}_predicted.jpg"
        logger.info(f"📤 Uploading to S3:")
        logger.info(f"   - Original: {original_s3_key}")
        logger.info(f"   - Predicted: {predicted_s3_key}")

        try:
            s3 = boto3.client('s3')
            s3.upload_file(original_path, bucket_name, original_s3_key)
            logger.info("✅ Original image uploaded to S3")
            s3.upload_file(predicted_path, bucket_name, predicted_s3_key)
            logger.info("✅ Predicted image uploaded to S3")
        except Exception as e:
            logger.error(f"❌ S3 upload failed: {e}")
            return JSONResponse(
                status_code=500,
                content={'error': 'Failed to upload to S3'}
            )

        # Get labels from results
        labels = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{model.names[cls]}: {conf:.2f}"
                labels.append(label)
        logger.info(f"🏷️ Detected labels: {labels}")

        save_prediction_session(timestamp, original_path, predicted_path)

        for label in labels:
            save_detection_object(timestamp, label, conf, str(box.xyxy[0].tolist()))

        return JSONResponse({
            'prediction_uid': timestamp,
            'detection_count': len(labels),
            'labels': labels
        })

    except Exception as e:
        logger.error(f"❌ Error in predict endpoint: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )

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
