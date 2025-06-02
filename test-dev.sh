#!/bin/bash
set -e

echo "🔁 Activating virtual environment..."
source .venv/bin/activate

echo "📦 Installing test dependencies..."
pip install pytest opencv-python numpy

echo "🧪 Creating uploads/ folders if missing..."
mkdir -p uploads/original uploads/predicted

# Set S3 bucket environment variable for dev
export S3_BUCKET_NAME="maisa-dev-bucket"

if [ ! -f tests/test_image.jpg ]; then
  echo "🖼️ Creating placeholder test image"
  python -c "import cv2, numpy as np; cv2.imwrite('tests/test_image.jpg', np.zeros((100, 100, 3), dtype=np.uint8))"
fi

echo "🚀 Running tests in DEV environment..."
pytest tests/test_app.py --disable-warnings -v 