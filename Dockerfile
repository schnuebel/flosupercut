FROM python:3.10-slim

# ---- system deps ----
RUN apt-get update && apt-get install -y \
    ffmpeg \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---- working dir ----
WORKDIR /app

# ---- python deps ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- app code ----
COPY friend_detector.py .

# ---- runtime ----
ENTRYPOINT ["python", "friend_detector.py"]

