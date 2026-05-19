# Meme Face & Hand Reaction Engine

I use **face and hand tracking** with MediaPipe to generate meme-style visual reactions.

> just try to copy the images in media.

## Reference Reactions

Each state shows a image stored in `media/`, triggered by specific gesture + face combination.

| Reaction | Trigger | Reference |
|----------|--------|------------|
| Speed 🐵 | Blink detection | `speed.png` |
| Cry Goblin 🟩 | Fist + thumbs near mouth | `cry.png` |
| Mokney 🤔 | Index finger near mouth + open mouth | `thinking.jpg` |
| Mewing 🤫 | Just mewing | `mewing.png` |
| perturved Cat 😺 | Open mouth without blink | `uuy.jpg` |
| Feared yellow ball 😨 | Mixed hand gesture near mouth | `miedo.jpg` |
| serious donkey 🫏 | Default state | `donkey.png` |

## MediaPipe Models I Used:
FaceLandmarker and HandLandmarker.

## How It Works

1. Webcam frame is captured
2. MediaPipe detects:
   - Face landmarks
   - Hand landmarks
3. rules:
   - Distances between key points
   - Gesture conditions
4. A **state function** selects the active reaction
5. A meme image is rendered as overlay on the frame

## Instalation:

```bash
pip install opencv-python mediapipe numpy
```
or just run with UV
