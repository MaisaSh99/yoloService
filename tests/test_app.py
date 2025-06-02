import sys
import os
import shutil
import glob
from fastapi.testclient import TestClient
import time
import boto3

# Add project root to sys.path to import app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app

client = TestClient(app)

original_dir = "uploads/original"
predicted_dir = "uploads/predicted"
test_image_src = "tests/test_image.jpg"
test_image_dst = os.path.join(original_dir, "test_image.jpg")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
S3_TEST_KEY = "test_image.jpg"

# Global variable to share prediction ID between tests
prediction_uid = None

def setup_module(module):
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(predicted_dir, exist_ok=True)

    if S3_BUCKET_NAME:
        s3 = boto3.client("s3")
        try:
            s3.download_file(S3_BUCKET_NAME, S3_TEST_KEY, test_image_dst)
        except s3.exceptions.ClientError:
            if os.path.exists(test_image_src):
                print("Uploading test image to S3...")
                s3.upload_file(test_image_src, S3_BUCKET_NAME, S3_TEST_KEY)
                s3.download_file(S3_BUCKET_NAME, S3_TEST_KEY, test_image_dst)
            else:
                raise FileNotFoundError(f"Test image not found at {test_image_src}")
    else:
        if os.path.exists(test_image_src):
            shutil.copy(test_image_src, test_image_dst)
        else:
            raise FileNotFoundError(f"Test image not found at {test_image_src}")

def test_predict_with_real_image():
    global prediction_uid
    print("Testing: /predict with real image...")
    with open(test_image_dst, "rb") as f:
        response = client.post("/predict", files={"file": ("test_image.jpg", f, "image/jpeg")})
    assert response.status_code == 200
    result = response.json()
    assert "prediction_uid" in result
    prediction_uid = result["prediction_uid"]
    assert prediction_uid
    print("Prediction UID:", prediction_uid)
    time.sleep(1)

def test_get_prediction_by_uid():
    global prediction_uid
    print("Testing: /prediction/{uid}")
    assert prediction_uid is not None
    response = client.get(f"/prediction/{prediction_uid}")
    assert response.status_code == 200
    json_data = response.json()
    assert "detection_objects" in json_data
    assert isinstance(json_data["detection_objects"], list)

def test_get_prediction_image():
    global prediction_uid
    print("Testing: /prediction/{uid}/image")
    response = client.get(
        f"/prediction/{prediction_uid}/image",
        headers={"Accept": "image/jpeg"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")

def test_get_predictions_by_label():
    print("Testing: /predictions/label/{label}")
    response = client.get("/predictions/label/person")
    assert response.status_code in [200, 404]  # 404 if no match, still valid

def test_get_predictions_by_score_valid():
    print("Testing: /predictions/score/0.2")
    response = client.get("/predictions/score/0.2")
    assert response.status_code == 200

def test_get_predictions_by_score_invalid():
    print("Testing: /predictions/score/1.5 (invalid)")
    response = client.get("/predictions/score/1.5")
    assert response.status_code == 422
    assert "should be less than or equal to" in response.text

def test_get_original_image():
    print("Testing: /image/original/test_image.jpg")
    response = client.get(f"/image/original/test_image.jpg")
    assert response.status_code == 200

def test_get_predicted_image():
    print("Testing: /image/predicted/{filename}")
    pred_images = glob.glob(os.path.join(predicted_dir, "**", "*.jpg"), recursive=True)
    if not pred_images:
        print("⚠️ No predicted images found.")
        for root, dirs, files in os.walk(predicted_dir):
            print(f"In {root}: {files}")
    assert pred_images, "No predicted image found to test /image/predicted"

    full_path = pred_images[0]
    filename = os.path.basename(full_path)
    top_level_path = os.path.join(predicted_dir, filename)

    # Only copy if the file is in a subdirectory
    if os.path.abspath(full_path) != os.path.abspath(top_level_path):
        shutil.copy(full_path, top_level_path)

    response = client.get(f"/image/predicted/{filename}")
    assert response.status_code == 200



def test_prediction_not_found():
    print("Testing: /prediction/invalid_uid")
    response = client.get("/prediction/invalid_uid")
    assert response.status_code == 404
    assert "Prediction not found" in response.text

def test_prediction_image_not_found():
    print("Testing: /prediction/invalid_uid/image")
    response = client.get("/prediction/invalid_uid/image", headers={"Accept": "image/jpeg"})
    assert response.status_code == 404
    assert "Prediction not found" in response.text

def test_prediction_image_not_acceptable_format():
    global prediction_uid
    print("Testing: /prediction/{uid}/image with unacceptable format")
    response = client.get(
        f"/prediction/{prediction_uid}/image",
        headers={"Accept": "application/json"}
    )
    assert response.status_code == 406
    assert "Client does not accept an image format" in response.text

def test_invalid_image_type_endpoint():
    print("Testing: /image/invalid_type/test.jpg")
    response = client.get("/image/invalid_type/test.jpg")
    assert response.status_code == 400
    assert "Invalid image type" in response.text

def test_original_image_not_found():
    print("Testing: /image/original/nonexistent.jpg")
    response = client.get("/image/original/nonexistent.jpg")
    assert response.status_code == 404
    assert "Image not found" in response.text

def test_predicted_image_not_found():
    print("Testing: /image/predicted/nonexistent.jpg")
    response = client.get("/image/predicted/nonexistent.jpg")
    assert response.status_code == 404
    assert "Image not found" in response.text

def test_health_check():
    print("Testing: /health")
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json().get("status") == "ok"
