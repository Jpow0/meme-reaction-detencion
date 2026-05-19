import cv2

# =========================================================
# CARGA DE IMÁGENES
# =========================================================

def load_image(name, size=(200, 200)):

    path = rf"media\{name}"

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(path)

    return cv2.resize(img, size)

# =========================================================
# OVERLAY PNG
# =========================================================

def overlay_png(frame, overlay, x, y):

    h, w = overlay.shape[:2]

    if y + h > frame.shape[0] or x + w > frame.shape[1]:
        return

    if overlay.shape[2] == 4:

        alpha = overlay[:, :, 3] / 255.0

        roi = frame[y:y+h, x:x+w]

        for c in range(3):

            roi[:, :, c] = (
                alpha * overlay[:, :, c]
                +
                (1 - alpha) * roi[:, :, c]
            )

    else:

        frame[y:y+h, x:x+w] = overlay

# =========================================================
# LANDMARK ESTÉTICO
# =========================================================

def draw_point(image, landmarks, h, w, idx, color):

    lm = landmarks[idx]

    x = int(lm.x * w)
    y = int(lm.y * h)

    # Glow exterior
    cv2.circle(image, (x, y), 12, color, 1)

    # Anillo
    cv2.circle(image, (x, y), 7, color, 2)

    # Centro
    cv2.circle(image, (x, y), 3, color, -1)

# =========================================================
# HUD
# =========================================================

def draw_hud(image, current_status):

    cv2.rectangle(
        image,
        (20, 20),
        (260, 90),
        (20, 20, 20),
        -1
    )

    cv2.rectangle(
        image,
        (20, 20),
        (260, 90),
        (90, 90, 90),
        2
    )

    cv2.putText(
        image,
        "Face Tracking",
        (35, 50),
        cv2.FONT_HERSHEY_DUPLEX,
        0.75,
        (255,255,255),
        2
    )

    cv2.putText(
        image,
        f"State: {current_status}",
        (35, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (220,220,220),
        2
    )

# =========================================================
# DEBUG PANEL
# =========================================================

def draw_debug_panel(image, estado):

    cv2.rectangle(
        image,
        (20, 110),
        (340, 390),
        (15, 15, 15),
        -1
    )

    cv2.rectangle(
        image,
        (20, 110),
        (340, 390),
        (90, 90, 90),
        2
    )

    cv2.putText(
        image,
        "DEBUG STATES",
        (35, 140),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        (255,255,255),
        2
    )

    estados_debug = [
        ("parpadeo", estado["parpadeo"]),
        ("boca_abierta", estado["boca_abierta"]),
        ("indice_boca", estado["indice_boca"]),
        ("medio_boca", estado["medio_boca"]),
        ("pulgar_boca", estado["pulgar_boca"]),
        ("indice_nariz", estado["indice_nariz"]),
        ("punio cerrado", estado["punio cerrado"]),
    ]

    y_debug = 175

    for nombre, activo in estados_debug:

        color = (0,255,0) if activo else (255,255,255)

        texto = f"[ {'ON' if activo else 'OFF'} ]  {nombre}"

        cv2.putText(
            image,
            texto,
            (35, y_debug),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2
        )

        y_debug += 30

# =========================================================
# DETECTOR DE PUÑO
# =========================================================

def is_fist_closed(hand):

    cerrados = sum(
        hand[d].y > hand[d-2].y
        for d in [8,12,16,20]
    )

    return cerrados >= 4