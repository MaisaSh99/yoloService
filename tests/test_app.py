import sys
import os
import shutil
import glob
from fastapi.testclient import TestClient


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app

client = TestClient(app)


def test_predict_with_real_image():
    original_dir = "uploads/original"
    predicted_dir = "uploads/predicted"
    test_image_src = "tests/test_image.jpg"
    test_image_dst = os.path.join(original_dir, "test_image.jpg")

    assert os.path.exists(test_image_src), f"Test image not found at {test_image_src}"

    for f in glob.glob(os.path.join(predicted_dir, "*.jpg")):
        os.remove(f)

    shutil.copy(test_image_src, test_image_dst)
    assert os.path.exists(test_image_dst), "Failed to copy image to original folder"

    with open(test_image_dst, "rb") as f:
        files = {"file": ("test_image.jpg", f, "image/jpeg")}
        response = client.post("/predict", files=files)

    assert response.status_code == 200, "Predict endpoint failed"

    result = response.json()
    assert "prediction_id" in result or "labels" in result, "Response missing prediction data"

    predicted_files = glob.glob(os.path.join(predicted_dir, "*.jpg"))
    assert predicted_files, "No predicted image found in predicted folder"
