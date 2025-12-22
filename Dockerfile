FROM python:3.10-slim

# ---- system deps ----
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---- working dir ----
WORKDIR /app

# ---- python deps ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- app code ----
COPY detect_friend.py .

# ---- runtime ----
ENTRYPOINT ["python", "detect_friend.py"]

