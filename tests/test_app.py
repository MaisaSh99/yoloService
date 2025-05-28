import sys
import os
import shutil
import glob
from fastapi.testclient import TestClient
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app

client = TestClient(app)

original_dir = "uploads/original"
predicted_dir = "uploads/predicted"
test_image_src = "tests/test_image.jpg"
test_image_dst = os.path.join(original_dir, "test_image.jpg")

prediction_uid = None  # GLOBAL for all tests

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
    response = client.get(f"/prediction/{prediction_uid}")
    assert response.status_code == 200
    assert "labels" in response.json()

def test_get_prediction_image():
    global prediction_uid
    response = client.get(f"/prediction/{prediction_uid}/image")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")

def test_get_predictions_by_label():
    response = client.get("/predictions/label/person")
    assert response.status_code in [200, 404]

def test_get_predictions_by_score():
    response = client.get("/predictions/score/0.2")
    assert response.status_code == 200

def test_get_original_image():
    response = client.get(f"/image/original/test_image.jpg")
    assert response.status_code == 200

def test_get_predicted_image():
    pred_images = glob.glob(os.path.join(predicted_dir, "*.jpg"))
    if pred_images:
        filename = os.path.basename(pred_images[0])
        response = client.get(f"/image/predicted/{filename}")
        assert response.status_code == 200
    else:
        assert False, "No predicted image found to test /image/predicted"
