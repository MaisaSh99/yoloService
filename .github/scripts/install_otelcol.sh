#!/bin/bash
set -e

echo "📦 Installing OpenTelemetry Collector..."
wget https://github.com/open-telemetry/opentelemetry-collector-releases/releases/latest/download/otelcol-linux-amd64
chmod +x otelcol-linux-amd64
sudo mv otelcol-linux-amd64 /usr/local/bin/otelcol

echo "🛠️ Creating systemd service..."
sudo tee /etc/systemd/system/otelcol.service > /dev/null <<EOF
[Unit]
Description=OpenTelemetry Collector
After=network.target

[Service]
ExecStart=/usr/local/bin/otelcol --config /home/ubuntu/otelcol-config.yaml
Restart=always
User=ubuntu

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable otelcol
sudo systemctl restart otelcol
