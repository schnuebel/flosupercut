import os
import subprocess
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.io import wavfile

# ---------------- CONFIG ----------------
YOUTUBE_URL = os.environ.get("YOUTUBE_URL")
FRIEND_WAV = os.environ.get("FRIEND_WAV", "/data/friend.wav")

if not YOUTUBE_URL:
    raise ValueError("YOUTUBE_URL environment variable not set")

CHUNK_SECONDS = 20
SIMILARITY_THRESHOLD = 0.75
FULL_EPISODE_THRESHOLD = 0.7   # 70% coverage
MERGE_GAP_SECONDS = 30
PADDING_SECONDS = 10
# ----------------------------------------

encoder = VoiceEncoder()

def run(cmd):
    subprocess.run(cmd, shell=True, check=True)

def download_episode():
    run(f"""
    yt-dlp -f bestvideo+bestaudio --merge-output-format mp4 \
    -o episode.mp4 {YOUTUBE_URL}
    """)

def extract_audio():
    run("ffmpeg -y -i episode.mp4 -ar 16000 -ac 1 episode.wav")

def load_friend_embedding():
    wav = preprocess_wav(FRIEND_WAV)
    return encoder.embed_utterance(wav)

def split_audio():
    sr, data = wavfile.read("episode.wav")
    chunk_size = CHUNK_SECONDS * sr
    chunks = []

    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        if len(chunk) < chunk_size // 2:
            continue
        chunks.append((i / sr, chunk))

    return sr, chunks

def is_friend(chunk, sr, friend_embed):
    wav = chunk.astype(np.float32) / 32768.0
    embed = encoder.embed_utterance(wav)
    similarity = np.dot(friend_embed, embed)
    return similarity >= SIMILARITY_THRESHOLD

def merge_segments(segments):
    merged = []
    for start, end in segments:
        if not merged or start - merged[-1][1] > MERGE_GAP_SECONDS:
            merged.append([start, end])
        else:
            merged[-1][1] = end
    return merged

def cut(start, end, output):
    run(f"""
    ffmpeg -y -i episode.mp4 -ss {start} -to {end} -c copy {output}
    """)

def main():
    print("Downloading episode...")
    download_episode()

    print("Extracting audio...")
    extract_audio()

    print("Loading voice reference...")
    friend_embed = load_friend_embedding()

    print("Analyzing audio...")
    sr, chunks = split_audio()

    friend_segments = []

    for start, chunk in chunks:
        if is_friend(chunk, sr, friend_embed):
            friend_segments.append((start, start + CHUNK_SECONDS))

    if not friend_segments:
        print("Friend not detected.")
        return

    total_friend_time = sum(e - s for s, e in friend_segments)
    episode_length = chunks[-1][0] + CHUNK_SECONDS
    coverage = total_friend_time / episode_length

    print(f"Friend coverage: {coverage:.2%}")

    if coverage >= FULL_EPISODE_THRESHOLD:
        print("Detected FULL EPISODE HOST")
        run("ffmpeg -y -i episode.mp4 -c copy /data/friend_full_episode.mp4")
        return

    print("Detected FIXED SEGMENT")
    merged = merge_segments(friend_segments)
    largest = max(merged, key=lambda x: x[1] - x[0])

    start = max(0, largest[0] - PADDING_SECONDS)
    end = largest[1] + PADDING_SECONDS

    print(f"Cutting segment: {start:.1f}s → {end:.1f}s")
    cut(start, end, "/data/friend_segment.mp4")

if __name__ == "__main__":
    main()

