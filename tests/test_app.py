import sys
import os
import shutil
import glob
from fastapi.testclient import TestClient
import time
import json

# Add project root to sys.path to import app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app

client = TestClient(app)

original_dir = "uploads/original"
predicted_dir = "uploads/predicted"
test_image_src = "tests/test_image.jpg"
test_image_dst = os.path.join(original_dir, "test_image.jpg")

# Global variable to share prediction ID between tests
prediction_uid = None

def setup_module(module):
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(predicted_dir, exist_ok=True)
    if os.path.exists(test_image_src):
        shutil.copy(test_image_src, test_image_dst)
    else:
        raise FileNotFoundError(f"Test image not found at {test_image_src}")

def test_predict_with_real_image():
    global prediction_uid
    with open(test_image_dst, "rb") as f:
        response = client.post("/predict", files={"file": ("test_image.jpg", f, "image/jpeg")})
    assert response.status_code == 200
    result = response.json()
    assert "prediction_uid" in result
    prediction_uid = result["prediction_uid"]
    assert prediction_uid
    time.sleep(1)

def test_get_prediction_by_uid():
    global prediction_uid
    assert prediction_uid is not None
    response = client.get(f"/prediction/{prediction_uid}")
    assert response.status_code == 200
    json_data = response.json()
    assert "detection_objects" in json_data
    assert isinstance(json_data["detection_objects"], list)

def test_get_prediction_image():
    global prediction_uid
    response = client.get(
        f"/prediction/{prediction_uid}/image",
        headers={"Accept": "image/jpeg"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")

def test_get_predictions_by_label():
    response = client.get("/predictions/label/person")
    assert response.status_code in [200, 404]

def test_get_predictions_by_score_valid():
    response = client.get("/predictions/score/0.2")
    assert response.status_code == 200

def test_get_predictions_by_score_invalid():
    response = client.get("/predictions/score/1.5")
    assert response.status_code == 400
    assert "Score must be between 0 and 1" in response.text

def test_get_original_image():
    response = client.get(f"/image/original/test_image.jpg")
    assert response.status_code == 200

def test_get_predicted_image():
    pred_images = glob.glob(os.path.join(predicted_dir, "*.jpg"))
    assert pred_images, "No predicted image found to test /image/predicted"
    filename = os.path.basename(pred_images[0])
    response = client.get(f"/image/predicted/{filename}")
    assert response.status_code == 200

def test_prediction_not_found():
    response = client.get("/prediction/invalid_uid")
    assert response.status_code == 404
    assert "Prediction not found" in response.text

def test_prediction_image_not_found():
    response = client.get("/prediction/invalid_uid/image", headers={"Accept": "image/jpeg"})
    assert response.status_code == 404
    assert "Prediction not found" in response.text

def test_prediction_image_not_acceptable_format():
    global prediction_uid
    response = client.get(
        f"/prediction/{prediction_uid}/image",
        headers={"Accept": "application/json"}
    )
    assert response.status_code == 406
    assert "Client does not accept an image format" in response.text

def test_invalid_image_type_endpoint():
    response = client.get("/image/invalid_type/test.jpg")
    assert response.status_code == 400
    assert "Invalid image type" in response.text

def test_original_image_not_found():
    response = client.get("/image/original/nonexistent.jpg")
    assert response.status_code == 404
    assert "Image not found" in response.text

def test_predicted_image_not_found():
    response = client.get("/image/predicted/nonexistent.jpg")
    assert response.status_code == 404
    assert "Image not found" in response.text

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json().get("status") == "ok"
