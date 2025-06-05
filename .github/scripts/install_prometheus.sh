#!/bin/bash

echo "📦 Installing Prometheus..."

cd /tmp

# Hardcoded working version
PROM_VERSION="2.51.1"
FILENAME="prometheus-${PROM_VERSION}.linux-amd64.tar.gz"
FOLDER="prometheus-${PROM_VERSION}.linux-amd64"

wget https://github.com/prometheus/prometheus/releases/download/v${PROM_VERSION}/${FILENAME}
tar -xzf ${FILENAME}
sudo mv ${FOLDER} /opt/prometheus

# Create systemd service
sudo tee /etc/systemd/system/prometheus.service > /dev/null <<EOL
[Unit]
Description=Prometheus Monitoring
Wants=network-online.target
After=network-online.target

[Service]
ExecStart=/opt/prometheus/prometheus \
  --config.file=/opt/prometheus/prometheus.yml \
  --storage.tsdb.path=/opt/prometheus/data \
  --web.listen-address=:9090
Restart=always

[Install]
WantedBy=multi-user.target
EOL

# Start Prometheus
sudo systemctl daemon-reload
sudo systemctl enable prometheus
sudo systemctl restart prometheus
