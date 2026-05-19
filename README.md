# Meme Face & Hand Reaction Engine

I use **face and hand tracking** using MediaPipe to generate meme-style visual reactions.

---

In short:

> It does not recognize emotions — it *imitates vibes*.

---

## 🖼️ Reference Reactions

Each state is inspired by an image stored in `media/`, and triggered by specific gesture + face combinations.

| Reaction | Trigger | Reference |
|----------|--------|------------|
| Speed Mode 🐵 | Blink detection | `speed.png` |
> | Cry Goblin 🟩 | Fist + thumbs near mouth | `cry.png` |
| Mokney 🤔 | Index finger near mouth + open mouth | `thinking.jpg` |
| Mewing 🤫 | Just mewing | `mewing.png` |
| perturved Cat 😺 | Open mouth without blink | `uuy.jpg` |
| Feared yellow ball 😨 | Mixed hand gesture near mouth | `miedo.jpg` |
| serious donkey 🫏 | Default state | `donkey.png` |

---

## MediaPipe Models Used

This project uses MediaPipe Tasks API with two main models:

### FaceLandmarker
Detects and tracks 468 facial landmarks in real time.

---

### HandLandmarker
Detects up to 2 hands with 21 landmarks each.

---

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

---

## 📦 Installation

```bash
pip install opencv-python mediapipe numpy

```bash
or just use UV
