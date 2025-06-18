import boto3
from fastapi import FastAPI, HTTPException, Request, Form, UploadFile, File
from fastapi.responses import FileResponse
from ultralytics import YOLO
from PIL import Image
import os
import uuid
import traceback
import torch
from datetime import datetime
from fastapi import Path

# Import our storage layer
from storage import get_storage

# Disable GPU usage
torch.cuda.is_available = lambda: False

app = FastAPI()

UPLOAD_DIR = "uploads/original"
PREDICTED_DIR = "uploads/predicted"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)

model = YOLO("yolov8n.pt")

bucket_name = os.getenv("S3_BUCKET_NAME")
print("[YOLO] Using S3 bucket:", bucket_name)

# Initialize storage on startup and log the configuration
print(f"[YOLO] Storage Configuration:")
print(f"[YOLO] STORAGE_TYPE: {os.getenv('STORAGE_TYPE', 'NOT_SET')}")
print(f"[YOLO] DYNAMODB_TABLE: {os.getenv('DYNAMODB_TABLE', 'NOT_SET')}")
print(f"[YOLO] AWS_DEFAULT_REGION: {os.getenv('AWS_DEFAULT_REGION', 'NOT_SET')}")

try:
    storage = get_storage()
    print(f"[YOLO] ✅ Storage initialized: {type(storage).__name__}")
    if hasattr(storage, 'table_name'):
        print(f"[YOLO] DynamoDB table: {storage.table_name}")
    elif hasattr(storage, 'db_path'):
        print(f"[YOLO] SQLite database: {storage.db_path}")
except Exception as e:
    print(f"[YOLO] ❌ Storage initialization failed: {e}")
    raise


@app.post("/predict")
def predict(request: Request, file: UploadFile = File(...)):
    print(f"[YOLO] Incoming /predict with file: {file.filename}")
    print(f"[YOLO] Using storage: {type(storage).__name__}")

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

        # Save prediction session using storage layer
        print(f"[YOLO] Saving prediction to {type(storage).__name__}")
        storage.save_prediction(uid, original_path, predicted_path)
        print(f"[YOLO] Successfully saved prediction session")

        detected_labels = []
        for box in results[0].boxes:
            label_idx = int(box.cls[0].item())
            label = model.names[label_idx]
            score = float(box.conf[0])
            bbox = box.xyxy[0].tolist()

            # Save detection using storage layer
            print(f"[YOLO] Saving detection: {label} (score: {score:.2f})")
            storage.save_detection(uid, label, score, bbox)
            detected_labels.append(label)

        print(f"[YOLO] Successfully saved {len(detected_labels)} detections")

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
    result = storage.get_prediction(uid)
    if not result:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return result


@app.get("/predictions/label/{label}")
def get_predictions_by_label(label: str):
    return storage.get_predictions_by_label(label)


@app.get("/predictions/score/{min_score}")
def get_predictions_by_score(min_score: float = Path(..., ge=0.0, le=1.0)):
    return storage.get_predictions_by_score(min_score)


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

    image_path = storage.get_prediction_image_path(uid)
    if not image_path:
        raise HTTPException(status_code=404, detail="Prediction not found")

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