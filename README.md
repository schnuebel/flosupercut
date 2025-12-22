docker build -t friend-detector .

docker run --rm \
  -e YOUTUBE_URL="https://youtube.com/watch?v=VIDEO_ID" \
  -v "$(pwd)/data:/data" \
  friend-detector


data/
├── friend.wav
├── friend_segment.mp4
└── friend_full_episode.mp4

