# =========================================================
# Librerías
# =========================================================

import cv2
import mediapipe as mp

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

from config import *
from func import *

# =========================================================
# MODELOS
# =========================================================

FACE_MODEL = "models/face_landmarker.task"
HAND_MODEL = "models/hand_landmarker.task"

# =========================================================
# ESCALADO PARA OPTIMIZACIÓN & IMAGENES
# =========================================================

PROCESS_W = 640
PROCESS_H = 360

HAND_DETECTION_EVERY = 2

think_st = load_image("thinking.jpg")
miedo = load_image("miedo.jpg")
speed = load_image("speed.png")
cry = load_image("cry.png")
mewing = load_image("mewing.png")
donkey = load_image("donkey.png")
uuy = load_image("uuy.jpg")

# =========================================================
# MOODELOS DE MEDIAPIPEE 
# =========================================================

face_options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACE_MODEL),
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

hand_options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL),
    num_hands=2
)

face_detector = vision.FaceLandmarker.create_from_options(face_options)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

# =========================================================
# CÁMARA
# =========================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_count = 0

last_hand_result = None

# =========================================================
# LOOP
# =========================================================

while cap.isOpened(): # si la camara funciona

    success, image = cap.read() # se lee un frame

    if not success:
        break

    frame_count += 1

    image = cv2.flip(image, 1) # espejar para naturalidad 

    h, w, _ = image.shape

    # =====================================================
    # INFERENCIA EN BAJA RESOLUCIÓN (fondo)
    # =====================================================

    small = cv2.resize( image, (PROCESS_W, PROCESS_H) ) # se infiere en baja resolución para optimizar

    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB ) # mediapipe usa RGB, OpenCV usa BGR, hay que convertir

    mp_image = mp.Image( image_format=mp.ImageFormat.SRGB, data=rgb ) # formato de imagen para mediapipe

    # =====================================================
    # ESTADO
    # =====================================================

    estado = { # estados según landmarks y distancias, para definir eventos despues
        "parpadeo": False,
        "boca_abierta": False,
        "indice_boca": False,
        "medio_boca": False,
        "pulgar_boca": False,
        "indice_nariz": False,
        "punio cerrado": False,
        "boca_izq": None,
        "boca_der": None,
        "nariz": None
    }

    current_status = "Neutral" # estado base burro shreck (yo)

    # =====================================================
    # FACE
    # =====================================================

    face_result = face_detector.detect(mp_image) # output de mediapipe con landmarks de la cara

    face = None

    if face_result.face_landmarks: # si se detecta cara

        face = face_result.face_landmarks[0] 

        apertura_izq = dist(face[159], face[145]) # distancia entre parpado superior e inferior izquierdo
        ancho_izq = dist(face[33], face[133]) # distancia entre extremos del ojo izquierdo

        apertura_der = dist(face[386], face[374]) # distancia entre parpado superior e inferior derecho
        ancho_der = dist(face[362], face[263]) # distancia entre extremos del ojo derecho

        estado["parpadeo"] = ( # detecta apertura de ojos 
            apertura_izq / ancho_izq < umbral_parpadeo
            and
            apertura_der / ancho_der < umbral_parpadeo
        )

        boca_ratio = ( # ratio de apertura de boca, para escalar umbral segun tamaño de cara
            dist(face[13], face[14])
            /
            dist(face[12], face[291])
        )

        estado["boca_abierta"] = (
            boca_ratio > umbral_boca_abierta
        )

        estado["boca_izq"] = (
            int(face[61].x * w),
            int(face[61].y * h)
        )

        estado["boca_der"] = (
            int(face[291].x * w),
            int(face[291].y * h)
        )

        estado["nariz"] = (
            int(face[1].x * w),
            int(face[1].y * h)
        )

        # DEBUG CARA

        if debug:

            for i in FACE_LANDMARKS:

                lm = face[i]

                x = int(lm.x * w)
                y = int(lm.y * h)

                cv2.circle(image, (x, y), 2, (0,255,0), -1)

    # =====================================================
    # HANDS
    # =====================================================

    hands_detected = []

    hand_result = last_hand_result

    # SOLO detectar manos cada N frames
    # Y solo si hay cara

    if face is not None and frame_count % HAND_DETECTION_EVERY == 0:

        hand_result = hand_detector.detect(mp_image)

        last_hand_result = hand_result

    if ( 
        hand_result
        and
        hand_result.hand_landmarks
        and
        estado["boca_izq"]
    ):

        boca_izq = estado["boca_izq"]
        boca_der = estado["boca_der"]
        nariz = estado["nariz"]

        umbral = dist_px(boca_izq, boca_der) * 0.7

        medios_izq = []
        medios_der = []

        pulgar_izq_hands = []
        pulgar_der_hands = []

        indice_boca = False
        indice_nariz = False

        punio_cerrado = False

        for hand in hand_result.hand_landmarks:# por cada mano detectada

            hands_detected.append(hand) # se guardan para dibujar despues y no perder detección aunque no se detecten en cada frame

            if debug: # DEBUG MANO

                for i in HAND_LANDMARKS:

                    x = int(hand[i].x * w)
                    y = int(hand[i].y * h)

                    cv2.circle(image, (x, y), 3, (255,0,0), -1)

            p_ind = ( # punta del indice
                int(hand[8].x * w),
                int(hand[8].y * h)
            )

            p_med = ( # punta del medio
                int(hand[12].x * w),
                int(hand[12].y * h)
            )

            p_pul = ( # punta del pulgar
                int(hand[4].x * w),
                int(hand[4].y * h)
            )

            if dist_px(p_med, boca_izq) < umbral: # detectar si el medio está a la derecha de la boca
                medios_izq.append(hand)

            if dist_px(p_med, boca_der) < umbral: # detectar si el medio está a la izquierda de la boca
                medios_der.append(hand)

            if dist_px(p_pul, boca_izq) < umbral*2: # detectar si el pulgar está a la izquierd de la boca
                pulgar_izq_hands.append(hand)

            if dist_px(p_pul, boca_der) < umbral*2: # detectar si el pulgar está a la derecha de la boca
                pulgar_der_hands.append(hand)

            if dist_px(p_ind, boca_der) < umbral: # detectar si el indice está a la izquierda de la boca
                indice_boca = True

            if nariz and dist_px(p_ind, nariz) < umbral: # detectar si el indice está cerca de la nariz
                indice_nariz = True

            cerrados = sum( # suma por cada dedo cerrado, no pulgar
                hand[d].y > hand[d-2].y
                for d in [8,12,16,20]
            )

            if cerrados >= 4: # si hay 4 dedos cerrados, se considera puño cerrado
                punio_cerrado = True

        estado["medio_boca"] = ( # detectar si hay medio a ambos lados de la boca, para evento "bola amarilla miedo"
            len(medios_izq) > 0
            and
            len(medios_der) > 0
        )

        estado["pulgar_boca"] = False 

        for h1 in pulgar_izq_hands: # detectar si hay un pulgar a la izquierda de la boca y otro a la derecha, para evento "cry goblin we we"
            for h2 in pulgar_der_hands:

                if h1 is not h2:
                    estado["pulgar_boca"] = True 
        #actualizar estado global con detecciones de manos
        estado["indice_boca"] = indice_boca 
        estado["indice_nariz"] = indice_nariz 
        estado["punio cerrado"] = punio_cerrado

    # =====================================================
    # EVENTOS
    # =====================================================

    if estado["parpadeo"] and not estado["boca_abierta"]: # speed

        current_status = "Speed"

        overlay_png(image, speed, w-250, 120)

        draw_point(image, face, h, w, 159, (0,120,240))
        draw_point(image, face, h, w, 386, (0,120,240))
        draw_point(image, face, h, w, 61, (0,120,240))
        draw_point(image, face, h, w, 291, (0,120,240))

    elif estado["medio_boca"]: # bola amarilla asustada

        current_status = "Bola amarilla miedo"

        overlay_png(image, miedo, w-250, 120)

        for hand in hands_detected:

            for idx in [8, 12, 16]:

                draw_point(
                    image,
                    hand,
                    h,
                    w,
                    idx,
                    (75, 212, 255)
                )

    elif estado["indice_boca"] and estado["boca_abierta"]: # mono pensando

        current_status = "Thinking monkey"

        overlay_png(image, think_st, w-250, 120)

        for hand in hands_detected:

            draw_point(image, hand, h, w, 8, (14, 8, 31))
            draw_point(image, hand, h, w, 5, (14, 8, 31))

        draw_point(
            image,
            face,
            h,
            w,
            13,
            (148, 164, 196)
        )

    elif estado["punio cerrado"] and estado["pulgar_boca"]: # goblin llorando

        current_status = "Cry goblin we we"

        overlay_png(image, cry, w-250, 120)

        for hand in hands_detected:

            for idx in [4, 17, 5]:

                draw_point(
                    image,
                    hand,
                    h,
                    w,
                    idx,
                    (116, 227, 64)
                )

        draw_point(image, face, h, w, 159, (75, 212, 255))
        draw_point(image, face, h, w, 386, (75, 212, 255))

    elif estado["indice_nariz"]: # mewing bola azul

        current_status = "Mewing"

        overlay_png(image, mewing, w-250, 120)

        draw_point(image, face, h, w, 136, (229,30,50))
        draw_point(image, face, h, w, 365, (229,30,50))

        for hand in hands_detected:

            draw_point(
                image,
                hand,
                h,
                w,
                8,
                (248, 244, 241)
            )

    elif estado["boca_abierta"] and not estado["parpadeo"]: # gatito uuy

        current_status = "gatito uuy"

        overlay_png(image, uuy, w-250, 120)

        draw_point(image, face, h, w, 61, (120,120,240))
        draw_point(image, face, h, w, 291, (120,120,240))
        draw_point(image, face, h, w, 13, (120,120,240))
        draw_point(image, face, h, w, 14, (120,120,240))

    else:

        current_status = "Neutral"

        overlay_png(image, donkey, w-250, 120)

    # =====================================================
    # HUD
    # =====================================================

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

    # =====================================================
    # DEBUG PANEL
    # =====================================================

    if debug:

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

    # =====================================================
    # SHOW
    # =====================================================

    cv2.imshow("Face + Hands + Eventos", image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()