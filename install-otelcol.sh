#!/bin/bash
set -e

echo "📦 Installing OpenTelemetry Collector..."

sudo apt update
sudo apt install -y wget gnupg

wget -qO - https://packages.signalfx.com/publickey | sudo apt-key add -
echo "deb [signed-by=/usr/share/keyrings/opentelemetry-collector.gpg] https://packages.signalfx.com/debs stable main" | sudo tee /etc/apt/sources.list.d/opentelemetry-collector.list

sudo apt update
sudo apt install -y otelcol

cat <<EOF | sudo tee /etc/otelcol/config.yaml
receivers:
  hostmetrics:
    collection_interval: 15s
    scrapers:
      cpu:
      memory:
      disk:
      filesystem:
      load:
      network:
      processes:

exporters:
  prometheus:
    endpoint: "0.0.0.0:8889"

service:
  pipelines:
    metrics:
      receivers: [hostmetrics]
      exporters: [prometheus]
EOF

echo "🔁 Restarting otelcol..."
sudo systemctl restart otelcol
sudo systemctl enable otelcol

echo "✅ Otelcol is running. You can test it via:"
echo "curl http://localhost:8889/metrics"
