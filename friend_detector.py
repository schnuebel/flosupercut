import os
import subprocess
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.io import wavfile
import subprocess
import json

# ---------------- CONFIG ----------------
YOUTUBE_URL = os.environ.get("YOUTUBE_URL")
FRIEND_WAV = os.environ.get("FRIEND_WAV", "/data/friend.wav")

if not YOUTUBE_URL:
    raise ValueError("YOUTUBE_URL environment variable not set")

CHUNK_SECONDS = 10
SIMILARITY_THRESHOLD = 0.80
FULL_EPISODE_THRESHOLD = 0.7   
CHAPTER_PERCENT_THRESHOLD = 0.6
# ----------------------------------------

encoder = VoiceEncoder()

def run(cmd):
    subprocess.run(cmd, shell=True, check=True)

def get_chapters():
    cmd = f"yt-dlp --dump-json {YOUTUBE_URL}"
    result = subprocess.check_output(cmd, shell=True)
    info = json.loads(result)
    chapters = info.get("chapters", [])
    if not chapters:
        raise ValueError("No chapters found")
    return [
        {
            "title": c["title"],
            "start": c["start_time"],
            "end": c["end_time"]
        }
        for c in chapters
    ]

def download_episode():
    run(f"""
    yt-dlp -f bestvideo+bestaudio --merge-output-format mp4 \
    -o episode.mp4 {YOUTUBE_URL}
    """)

def extract_audio():
    run("ffmpeg -y -i episode.mp4 -ar 16000 -ac 1 episode.wav")

def extract_chapter_audio(start, end, output_wav):
    run(f"""
    ffmpeg -y -i episode.wav -ss {start} -to {end} {output_wav}
    """)

def chapter_has_friend(wav_path, friend_embed):
    wav = preprocess_wav(wav_path)
    embed = encoder.embed_utterance(wav)
    similarity = np.dot(friend_embed, embed)
    print(f"Similarity: {similarity:.3f}")
    return similarity >= SIMILARITY_THRESHOLD


def load_friend_embedding():
    wav = preprocess_wav(FRIEND_WAV)
    return encoder.embed_utterance(wav)


def cut(start, end, output):
    run(f"""
    ffmpeg -y -i episode.mp4 -ss {start} -to {end} -c copy {output}
    """)

def main():
    download_episode()
    extract_audio()

    friend_embed = load_friend_embedding()
    chapters = get_chapters()

    friend_chapters = []

    for i, ch in enumerate(chapters):
        print(f"\nAnalyzing chapter: {ch['title']}")
        tmp_wav = f"/tmp/chapter_{i}.wav"

        extract_chapter_audio(ch["start"], ch["end"], tmp_wav)

        if chapter_has_friend(tmp_wav, friend_embed):
            friend_chapters.append(ch)

    total = len(chapters)
    detected = len(friend_chapters)
    ratio = detected / total if total else 0

    print(f"\nFriend in {detected}/{total} chapters ({ratio:.1%})")

    # --- FULL EPISODE CASE ---
    if ratio >= CHAPTER_PERCENT_THRESHOLD:
        print("Detected PRIMARY HOST → keeping full episode")
        run("ffmpeg -y -i episode.mp4 -c copy /data/friend_full_episode.mp4")
        return

    # --- PARTIAL APPEARANCE CASE ---
    print("Detected GUEST / PARTIAL HOST")

    start = min(c["start"] for c in friend_chapters)
    end = max(c["end"] for c in friend_chapters)
    start = max(0, start)

    print(f"Cutting segment: {start:.1f}s → {end:.1f}s")
    cut(start, end, "/data/friend_segment.mp4")

if __name__ == "__main__":
    main()

