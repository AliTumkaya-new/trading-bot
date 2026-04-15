#!/bin/bash
# ============================================================
#  Trading Bot — Oracle Cloud Free Tier Deploy
#  Tamamen bedava, ömür boyu çalışır
# ============================================================
set -e

APP_DIR="/opt/trading-bot"
SERVICE_NAME="trading-bot"
PYTHON_VER="python3.12"

echo "========================================="
echo "  Trading Bot — Bedava VPS Kurulumu"
echo "========================================="

# 1. Python ve bağımlılıklar
echo "[1/4] Python kuruluyor..."
sudo apt-get update -y
sudo apt-get install -y python3.12 python3.12-venv python3-pip

# 2. Proje dizini
echo "[2/4] Proje ayarlanıyor..."
sudo mkdir -p "$APP_DIR"
sudo cp -r . "$APP_DIR/"
cd "$APP_DIR"

# 3. Sanal ortam
echo "[3/4] Python sanal ortam kuruluyor..."
$PYTHON_VER -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir -r trading_system_v1/requirements.txt

# 4. Systemd servisi
echo "[4/4] Servis kuruluyor..."
sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null <<EOF
[Unit]
Description=Trading Bot 7/24
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=${APP_DIR}/trading_system_v1/src
ExecStart=${APP_DIR}/.venv/bin/python main.py
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal

Environment=PYTHONUNBUFFERED=1
Environment=TZ=Europe/Istanbul

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}
sudo systemctl start ${SERVICE_NAME}

echo ""
echo "========================================="
echo "  ✅ Kurulum tamamlandı!"
echo "========================================="
echo ""
echo "Komutlar:"
echo "  Durum    : sudo systemctl status ${SERVICE_NAME}"
echo "  Loglar   : sudo journalctl -u ${SERVICE_NAME} -f"
echo "  Durdur   : sudo systemctl stop ${SERVICE_NAME}"
echo "  Yeniden  : sudo systemctl restart ${SERVICE_NAME}"
echo ""
