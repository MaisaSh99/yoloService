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
import threading
import json
import time
import requests
from urllib.parse import urlparse

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
print(f"[YOLO] ENVIRONMENT: {os.getenv('ENVIRONMENT', 'NOT_SET')}")

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


class SQSConsumer:
    def __init__(self):
        self.sqs = boto3.client('sqs', region_name='us-east-2')
        self.s3 = boto3.client('s3', region_name='us-east-2')

        # Determine which queue to use based on environment
        env = os.getenv('ENVIRONMENT', 'dev').lower()
        if env == 'prod':
            self.queue_name = 'maisa-polybot-chat-messages'
        else:
            self.queue_name = 'maisa-polybot-chat-messages-dev'

        try:
            response = self.sqs.get_queue_url(QueueName=self.queue_name)
            self.queue_url = response['QueueUrl']
            print(f"[SQS] ✅ Using queue: {self.queue_name}")
            print(f"[SQS] Queue URL: {self.queue_url}")
        except Exception as e:
            print(f"[SQS] ❌ Failed to get queue URL: {e}")
            self.queue_url = None

    def download_from_s3(self, s3_url, local_path):
        """Download image from S3 URL to local path"""
        try:
            # Parse s3://bucket/key format
            if not s3_url.startswith('s3://'):
                print(f"[SQS] ❌ Invalid S3 URL format: {s3_url}")
                return False

            parsed = urlparse(s3_url)
            bucket = parsed.netloc
            key = parsed.path.lstrip('/')

            print(f"[SQS] 📥 Downloading s3://{bucket}/{key} to {local_path}")

            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3.download_file(bucket, key, local_path)
            print(f"[SQS] ✅ Download successful")
            return True

        except Exception as e:
            print(f"[SQS] ❌ S3 download failed: {e}")
            return False

    def send_result_to_polybot(self, callback_url, result_data):
        """Send processing result back to Polybot service"""
        try:
            print(f"[SQS] 📤 Sending result to Polybot: {callback_url}")
            response = requests.post(callback_url, json=result_data, timeout=30)
            print(f"[SQS] ✅ Result sent, status: {response.status_code}")
            return response.status_code == 200
        except Exception as e:
            print(f"[SQS] ❌ Failed to send result to Polybot: {e}")
            return False

    def process_yolo_request(self, message_data):
        """Process YOLO request from SQS message"""
        try:
            chat_id = message_data['chat_id']
            image_url = message_data['image_url']
            prediction_id = message_data['prediction_id']
            callback_url = message_data.get('callback_url')

            print(f"[SQS] 🔄 Processing YOLO request: {prediction_id[:8]} for chat {chat_id}")

            # Download image from S3
            local_path = os.path.join(UPLOAD_DIR, f"{prediction_id}.jpg")
            if not self.download_from_s3(image_url, local_path):
                if callback_url:
                    self.send_result_to_polybot(callback_url, {
                        'chat_id': chat_id,
                        'prediction_id': prediction_id,
                        'status': 'error',
                        'error': 'Failed to download image from S3'
                    })
                return False

            # Process with YOLO
            try:
                print(f"[SQS] 🔍 Running YOLO detection...")
                results = model(local_path, device="cpu")

                # Create annotated image
                annotated_frame = results[0].plot()
                annotated_image = Image.fromarray(annotated_frame)

                # Save predicted image
                user_id = chat_id
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                predicted_filename = f"{user_id}/{timestamp}_predicted.jpg"
                predicted_path = os.path.join(PREDICTED_DIR, predicted_filename)

                os.makedirs(os.path.dirname(predicted_path), exist_ok=True)
                annotated_image.save(predicted_path)
                print(f"[SQS] 💾 Predicted image saved: {predicted_path}")

                # Save to storage
                storage.save_prediction(prediction_id, local_path, predicted_path)

                # Extract and save detections
                detected_labels = []
                for box in results[0].boxes:
                    label_idx = int(box.cls[0].item())
                    label = model.names[label_idx]
                    score = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()

                    storage.save_detection(prediction_id, label, score, bbox)
                    detected_labels.append(label)

                print(f"[SQS] ✅ Detected {len(detected_labels)} objects: {detected_labels}")

                # Upload predicted image to S3
                try:
                    s3_key = f"predicted/{user_id}/{timestamp}_predicted.jpg"
                    self.s3.upload_file(predicted_path, bucket_name, s3_key)
                    print(f"[SQS] ✅ Uploaded predicted image to S3: {s3_key}")
                except Exception as e:
                    print(f"[SQS] ⚠️ Failed to upload predicted image to S3: {e}")

                # Send result back to Polybot
                if callback_url:
                    result_data = {
                        'chat_id': chat_id,
                        'prediction_id': prediction_id,
                        'status': 'success',
                        'labels': detected_labels
                    }
                    self.send_result_to_polybot(callback_url, result_data)

                print(f"[SQS] ✅ YOLO processing complete for {prediction_id[:8]}")

            except Exception as e:
                print(f"[SQS] ❌ YOLO processing failed: {e}")
                traceback.print_exc()

                if callback_url:
                    self.send_result_to_polybot(callback_url, {
                        'chat_id': chat_id,
                        'prediction_id': prediction_id,
                        'status': 'error',
                        'error': f"YOLO processing failed: {str(e)}"
                    })
                return False

            # Cleanup local files
            try:
                if os.path.exists(local_path):
                    os.remove(local_path)
                if os.path.exists(predicted_path):
                    # Keep predicted image for API access, don't delete
                    pass
            except Exception as e:
                print(f"[SQS] ⚠️ Cleanup warning: {e}")

            return True

        except Exception as e:
            print(f"[SQS] ❌ Failed to process YOLO request: {e}")
            traceback.print_exc()
            return False

    def start_consuming(self):
        """Start consuming messages from SQS queue"""
        if not self.queue_url:
            print("[SQS] ❌ Queue URL not available, skipping SQS consumer")
            return

        print(f"[SQS] 🎯 Starting SQS consumer for {self.queue_name}")

        while True:
            try:
                # Receive messages from SQS
                response = self.sqs.receive_message(
                    QueueUrl=self.queue_url,
                    MaxNumberOfMessages=1,
                    WaitTimeSeconds=20,  # Long polling
                    MessageAttributeNames=['All'],
                    VisibilityTimeoutSeconds=300  # 5 minutes to process
                )

                messages = response.get('Messages', [])

                if not messages:
                    print("[SQS] ⏳ No messages in queue...")
                    continue

                for message in messages:
                    try:
                        message_body = message['Body']
                        receipt_handle = message['ReceiptHandle']
                        message_attributes = message.get('MessageAttributes', {})

                        print(f"[SQS] 📨 Received message: {message_body[:100]}...")

                        # Check if this is a YOLO request
                        message_type = message_attributes.get('MessageType', {}).get('StringValue')

                        if message_type == 'yolo_request':
                            print("[SQS] 🎯 Processing YOLO request message")
                            message_data = json.loads(message_body)

                            if self.process_yolo_request(message_data):
                                # Delete message after successful processing
                                self.sqs.delete_message(
                                    QueueUrl=self.queue_url,
                                    ReceiptHandle=receipt_handle
                                )
                                print("[SQS] ✅ Message processed and deleted")
                            else:
                                print("[SQS] ❌ Message processing failed, will retry later")
                        else:
                            # Not a YOLO request, check if it has 'type' field
                            try:
                                parsed_message = json.loads(message_body)
                                if parsed_message.get('type') == 'yolo_request':
                                    print("[SQS] 🎯 Processing YOLO request (type field)")
                                    if self.process_yolo_request(parsed_message):
                                        self.sqs.delete_message(
                                            QueueUrl=self.queue_url,
                                            ReceiptHandle=receipt_handle
                                        )
                                        print("[SQS] ✅ Message processed and deleted")
                                    else:
                                        print("[SQS] ❌ Message processing failed, will retry later")
                                else:
                                    # Not a YOLO message, delete it to avoid clogging
                                    print(
                                        f"[SQS] ℹ️ Ignoring non-YOLO message: {parsed_message.get('type', 'unknown')}")
                                    self.sqs.delete_message(
                                        QueueUrl=self.queue_url,
                                        ReceiptHandle=receipt_handle
                                    )
                            except json.JSONDecodeError:
                                print("[SQS] ⚠️ Invalid JSON message, deleting")
                                self.sqs.delete_message(
                                    QueueUrl=self.queue_url,
                                    ReceiptHandle=receipt_handle
                                )

                    except Exception as e:
                        print(f"[SQS] ❌ Error processing individual message: {e}")
                        traceback.print_exc()

            except KeyboardInterrupt:
                print("[SQS] 🛑 Stopping SQS consumer...")
                break
            except Exception as e:
                print(f"[SQS] ❌ SQS consumer error: {e}")
                time.sleep(5)  # Wait before retrying


def start_sqs_consumer():
    """Start SQS consumer in background thread"""
    try:
        consumer = SQSConsumer()
        consumer_thread = threading.Thread(target=consumer.start_consuming, daemon=True)
        consumer_thread.start()
        print("[SQS] ✅ SQS consumer started in background thread")
    except Exception as e:
        print(f"[SQS] ❌ Failed to start SQS consumer: {e}")


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


# Initialize SQS consumer when the app starts
start_sqs_consumer()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)