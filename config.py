# ===================== CONFIG =====================
umbral_parpadeo = 0.35
umbral_boca_abierta = 0.2
umbral_pulgar = 60

debug = True

# ===================== PUNTOS =====================
FACE_LANDMARKS = [
    1,
    33,
    133,
    159,
    145,
    362,
    263,
    386,
    374,
    61,
    291,
    13,
    14,
]  # range(468)
HAND_LANDMARKS = [0, 1, 4, 5, 8, 9, 12, 17, 20]  # range(21)


# ===================== DISTANCIAS =====================
def dist(p1, p2):
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5


def dist_px(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
