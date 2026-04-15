FROM python:3.12-slim

WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc && \
    rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları
COPY trading_system_v1/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodları
COPY trading_system_v1/src/ ./src/
COPY trading_system_v1/data/ ./data/

# Veri klasörü (DB burada oluşacak)
RUN mkdir -p /app/data

ENV PYTHONUNBUFFERED=1
ENV TZ=Europe/Istanbul

WORKDIR /app/src

CMD ["python", "main.py"]
