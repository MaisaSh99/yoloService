#!/bin/bash

echo "📦 Installing OpenTelemetry Collector..."

cd /tmp

# Download the latest tarball release
wget https://github.com/open-telemetry/opentelemetry-collector-releases/releases/latest/download/otelcol-linux-amd64.tar.gz
tar -xzf otelcol-linux-amd64.tar.gz
sudo mv otelcol /usr/local/bin/otelcol

# Create config directory
sudo mkdir -p /etc/otelcol

# Copy config from home directory (uploaded during deployment)
sudo cp ~/otelcol-config.yaml /etc/otelcol/config.yaml

# Create systemd service
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

# Start otelcol
sudo systemctl daemon-reload
sudo systemctl enable otelcol
sudo systemctl restart otelcol

echo "✅ OpenTelemetry Collector installed and running!"
