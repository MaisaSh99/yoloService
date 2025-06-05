#!/bin/bash
set -e

echo "📦 Installing Prometheus..."
cd ~
wget https://github.com/prometheus/prometheus/releases/latest/download/prometheus-2.51.1.linux-amd64.tar.gz
tar -xzf prometheus-2.51.1.linux-amd64.tar.gz
cd prometheus-2.51.1.linux-amd64

cat <<EOF > prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'otelcol'
    static_configs:
      - targets: ['localhost:8889']
EOF

echo "🛠️ Creating systemd service..."
sudo tee /etc/systemd/system/prometheus.service > /dev/null <<EOF
[Unit]
Description=Prometheus
After=network.target

[Service]
ExecStart=/home/ubuntu/prometheus-2.51.1.linux-amd64/prometheus --config.file=/home/ubuntu/prometheus-2.51.1.linux-amd64/prometheus.yml
Restart=always
User=ubuntu

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable prometheus
sudo systemctl restart prometheus
