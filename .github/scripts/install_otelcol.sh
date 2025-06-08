#!/bin/bash
set -e

VERSION="0.96.0"
FILENAME="otelcol_${VERSION}_linux_amd64.tar.gz"
URL="https://github.com/open-telemetry/opentelemetry-collector-releases/releases/download/v${VERSION}/${FILENAME}"

echo "📦 Installing OpenTelemetry Collector v${VERSION}..."

cd /tmp
wget "$URL"
tar -xzf "$FILENAME"
sudo mv otelcol /usr/local/bin/otelcol

# Setup config directory
sudo mkdir -p /etc/otelcol

# Systemd service
sudo tee /etc/systemd/system/otelcol.service > /dev/null <<EOL
[Unit]
Description=OpenTelemetry Collector
After=network.target

[Service]
ExecStart=/usr/local/bin/otelcol --config /etc/otelcol/config.yaml
Restart=always

[Install]
WantedBy=multi-user.target
EOL

sudo systemctl daemon-reload
sudo systemctl enable otelcol
sudo systemctl restart otelcol

echo "✅ OpenTelemetry Collector installed and running!"
