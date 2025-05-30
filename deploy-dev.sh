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
    git pull
else
    git clone "$REPO_URL"
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

echo "Copying and enabling YOLO dev service..."
sudo cp ~/yolo-dev.service /etc/systemd/system/yolo.service
sudo systemctl daemon-reload
sudo systemctl enable yolo.service
sudo systemctl restart yolo.service

echo "✅ YOLO development service deployed and running!"
