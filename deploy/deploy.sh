#!/bin/bash
# ============================================================
#  Trading Bot — VPS Deploy Script
#  Kullanım: ssh ile sunucuya bağlan, bu scripti çalıştır
# ============================================================
set -e

APP_DIR="/opt/trading-bot"
SERVICE_NAME="trading-bot"

echo "========================================="
echo "  Trading Bot Deploy"
echo "========================================="

# 1. Docker yoksa kur
if ! command -v docker &> /dev/null; then
    echo "[1/5] Docker kuruluyor..."
    curl -fsSL https://get.docker.com | sh
    sudo systemctl enable docker
    sudo systemctl start docker
else
    echo "[1/5] Docker zaten kurulu ✓"
fi

# 2. Docker Compose yoksa kur
if ! docker compose version &> /dev/null; then
    echo "[2/5] Docker Compose kuruluyor..."
    sudo apt-get install -y docker-compose-plugin
else
    echo "[2/5] Docker Compose zaten kurulu ✓"
fi

# 3. Uygulama dosyalarını kopyala
echo "[3/5] Uygulama dosyaları ayarlanıyor..."
sudo mkdir -p "$APP_DIR"
sudo cp -r . "$APP_DIR/"

# 4. systemd servisi kur
echo "[4/5] Sistem servisi kuruluyor..."
sudo cp deploy/trading-bot.service /etc/systemd/system/${SERVICE_NAME}.service
sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}

# 5. Başlat
echo "[5/5] Trading bot başlatılıyor..."
sudo systemctl restart ${SERVICE_NAME}

echo ""
echo "========================================="
echo "  ✅ Deploy tamamlandı!"
echo "========================================="
echo ""
echo "Komutlar:"
echo "  Durum   : sudo systemctl status ${SERVICE_NAME}"
echo "  Loglar  : sudo docker logs -f trading-bot"
echo "  Durdur  : sudo systemctl stop ${SERVICE_NAME}"
echo "  Başlat  : sudo systemctl start ${SERVICE_NAME}"
echo "  Dashboard: http://$(hostname -I | awk '{print $1}'):8501"
echo ""
