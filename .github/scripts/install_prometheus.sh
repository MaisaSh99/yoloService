#!/bin/bash
set -e

PROM_VERSION="2.51.1"
FOLDER="prometheus-${PROM_VERSION}.linux-amd64"
FILENAME="${FOLDER}.tar.gz"
OTELCOL_IP=${OTELCOL_IP:-"127.0.0.1"}  # fallback to localhost if not passed

if [ -f "/usr/local/bin/prometheus" ]; then
  echo "🔁 Prometheus already installed at /usr/local/bin/prometheus — skipping installation."
  exit 0
fi

echo "📦 Installing Prometheus v${PROM_VERSION}..."

cd /tmp
wget https://github.com/prometheus/prometheus/releases/download/v${PROM_VERSION}/${FILENAME}
tar -xzf ${FILENAME}
cd ${FOLDER}

# Move binaries to system path
sudo mv prometheus /usr/local/bin/
sudo mv promtool /usr/local/bin/

# Setup configuration and data directories
sudo mkdir -p /etc/prometheus /var/lib/prometheus
sudo cp -r consoles console_libraries /etc/prometheus/

# Generate config with dev and prod environments
sudo tee /etc/prometheus/prometheus.yml > /dev/null <<EOL
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # Development Environment
  - job_name: 'otel-collector-polybot-dev'
    static_configs:
      - targets: ['10.0.0.7:8889']
    metrics_path: '/metrics'
    scrape_interval: 15s
    scrape_timeout: 10s

  - job_name: 'otel-collector-yolo-dev'
    static_configs:
      - targets: ['10.0.1.17:8889']
    metrics_path: '/metrics'
    scrape_interval: 15s
    scrape_timeout: 10s

  # Production Environment
  - job_name: 'otel-collector-polybot-prod'
    static_configs:
      - targets: ['10.0.0.135:8889']
    metrics_path: '/metrics'
    scrape_interval: 15s
    scrape_timeout: 10s

  - job_name: 'otel-collector-yolo-prod'
    static_configs:
      - targets: ['10.0.1.143:8889']
    metrics_path: '/metrics'
    scrape_interval: 15s
    scrape_timeout: 10s
EOL

# Create Prometheus systemd service
sudo tee /etc/systemd/system/prometheus.service > /dev/null <<EOL
[Unit]
Description=Prometheus Monitoring
After=network.target

[Service]
ExecStart=/usr/local/bin/prometheus \
  --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.path=/var/lib/prometheus/ \
  --web.listen-address=:9090
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOL

# Reload and start service
sudo systemctl daemon-reload
sudo systemctl enable prometheus
sudo systemctl restart prometheus

echo "✅ Prometheus v${PROM_VERSION} installed and running!"
