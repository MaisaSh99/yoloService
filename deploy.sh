#!/bin/bash
set -e

echo "Installing system packages..."
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git libgl1

echo "Cloning repo or pulling latest changes..."
cd ~
REPO_NAME="yoloService"
REPO_URL="https://github.com/MaisaSh99/yoloService.git"
if [ -d "$REPO_NAME" ]; then
    cd "$REPO_NAME"
    git pull origin main
else
    git clone -b main "$REPO_URL"
    cd "$REPO_NAME"
fi

echo "Setting up virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r torch-requirements.txt
pip install -r requirements.txt
pip install opencv-python numpy

echo "Stopping any service using port 8080..."
sudo fuser -k 8080/tcp || true

echo "Copying and enabling YOLO production service..."
sudo cp ~/yolo-prod.service /etc/systemd/system/yolo-prod.service
sudo systemctl daemon-reload
sudo systemctl enable yolo-prod.service
sudo systemctl restart yolo-prod.service

echo "Checking YOLO production service status..."
sleep 3
sudo systemctl status yolo-prod.service || (journalctl -u yolo-prod.service -n 50 --no-pager && exit 1)

echo "Testing /health endpoint..."
curl -s http://localhost:8080/health || (echo "❌ Health check failed" && exit 1)

echo "✅ YOLO production service deployed and running!"
