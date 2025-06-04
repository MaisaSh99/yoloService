#!/bin/bash

echo "🛠️ Running OpenTelemetry Collector setup..."

echo "📦 Installing OpenTelemetry Collector..."
sudo apt update
sudo apt install -y wget gnupg

wget -qO - https://apt.opentelemetry.io/otel-gpg-key.pub | sudo gpg --dearmor -o /usr/share/keyrings/otel-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/otel-archive-keyring.gpg] https://apt.opentelemetry.io/apt stable main" | sudo tee /etc/apt/sources.list.d/otel.list

sudo apt update
sudo apt install -y otel-collector

echo "⚙️ Configuring Collector..."
sudo tee /etc/otelcol/config.yaml > /dev/null <<EOF
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

echo "✅ OpenTelemetry Collector is installed and running!"
