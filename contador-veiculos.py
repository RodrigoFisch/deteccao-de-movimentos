
import numpy as np
import cv2
import sys
import os
import time
import validator                 # usa validator.SimpleValidator
from random import randint

# =====================================================================
# CONFIGURAÇÕES VISUAIS (cores, fontes)
# =====================================================================

LINE_IN_COLLOR = (64, 255, 0)
LINE_OUT_COLLOR = (0, 0, 255)
BOUNDING_BOX_COLLOR = (255, 128, 0)
TRACKER_COLOR = (randint (0, 255), randint (0, 255), randint (0, 255))
TEXT_COLOR = (randint (0, 255), randint (0, 255), randint (0, 255))
FONT = cv2.FONT_HERSHEY_SIMPLEX

# =====================================================================
# CONFIGURAÇÃO DE VÍDEO (entrada / saída)
# =====================================================================

VIDEO_SOURCE = "video/cars.mp4"
VIDEO_OUT = "videos/results/result_traffic.mp4"

# Tipos de background subtractor disponíveis
BGS_TYPES = ["GMG", "MOG", "MOG2", "KNN", "CNT"]
BGS_TYPE = BGS_TYPES[2]   # "MOG2"

# =====================================================================
# FUNÇÕES AUXILIARES
# =====================================================================

def getKernerl(KERNEL_TYPE):
    """
    Retorna os kernels usados nos filtros morfológicos.
    Mantém exatamente a sua lógica original.
    """
    if KERNEL_TYPE == "dilation":
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return np.ones((3, 3), np.uint8)


def getFilter(img, filter):
    """
    Pipeline de filtragem:
        closing -> opening -> dilation
    Mantém o comportamento original (equivalente ao 'combine').
    """
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, getKernerl("closing"), iterations=2)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, getKernerl("opening"), iterations=2)
    return cv2.dilate(opening, getKernerl("dilation"), iterations=2)


def getBGSubtractor(BGS_TYPE):
    """
    Escolhe o subtractor (MOG2/KNN).
    """
    if BGS_TYPE == "MOG2":
        return cv2.createBackgroundSubtractorMOG2()
    if BGS_TYPE == "KNN":
        return cv2.createBackgroundSubtractorKNN()

    print("Tipo inválido")
    sys.exit(1)


def getCentroid(x, y, w, h):
    """
    Calcula o centróide do bounding box.
    """
    return x + w//2, y + h//2


# ============================================================
# FUNÇÃO PARA SALVAR IMAGEM DE CADA VEÍCULO CONTADO
# ============================================================

def save_vehicle_image(roi_frame, x, y, w, h, vtype, vid):
    """
    Salva a imagem (recorte) do veículo dentro da ROI, quando ele é CONTADO.
    Parâmetros:
        roi_frame : recorte da ROI (imagem)
        x, y, w, h: bounding box do veículo dentro da ROI
        vtype     : "CAR" ou "TRUCK" (string de exibição)
        vid       : ID numérico do objeto (proveniente do validator)
    """
    # Garantir limites válidos do recorte
    h_roi, w_roi = roi_frame.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w_roi, x + w)
    y1 = min(h_roi, y + h)

    if x1 <= x0 or y1 <= y0:
        return  # bounding inválido, não salva

    crop = roi_frame[y0:y1, x0:x1]

    if crop.size == 0:
        return

    # Pasta de saída
    out_dir = "vehicles"
    os.makedirs(out_dir, exist_ok=True)

    # Nome do arquivo: tipo + id + timestamp
    ts = int(time.time())
    filename = os.path.join(out_dir, f"{vtype}_{vid}_{ts}.jpg")

    cv2.imwrite(filename, crop)
    print(f"[INFO] Veículo salvo: {filename}")


# =====================================================================
# INICIALIZAÇÃO DO VÍDEO E ROI
# =====================================================================

cap = cv2.VideoCapture(VIDEO_SOURCE)
ok, frame = cap.read()

if not ok:
    print("Erro ao abrir o vídeo de entrada.")
    sys.exit(1)

# Selecionar ROI manualmente
bbox = cv2.selectROI(frame, False)
(w1, h1, w2, h2) = bbox  # x, y, width, height

# Cálculo da área e filtros
frameArea = w2 * h2
minArea = int(frameArea / 250)
maxArea = 15000

# Instancia background subtractor
bg = getBGSubtractor(BGS_TYPE)

# Instancia o validator (conta e identifica veículos)
# Dica: ajuste 'truck_area_threshold' conforme seu vídeo
validator_instance = validator.SimpleValidator(
    min_area=minArea,
    truck_area_threshold=5000
)

# =====================================================================
# LOOP PRINCIPAL
# =====================================================================

frame_index = 0

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    # Recorte correto da ROI (sem step acidental)
    roi = frame[h1:h1 + h2, w1:w1 + w2]

    # Subtração de fundo + limpeza
    fgmask = bg.apply(roi)
    fgmask = getFilter(fgmask, "combine")

    # Encontrar contornos dentro da ROI
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Filtragem básica de ruído
        if minArea < area <= maxArea:

            # Bounding box e centróide
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = getCentroid(x, y, w, h)

            # --------------------------
            #  PASSA PARA O VALIDATOR
            #  (agora retorna também o ID do objeto)
            # --------------------------
            vtype, counted, vid = validator_instance.register(cx, cy, int(area))

            # Label para exibição
            label = "TRUCK" if vtype == "truck" else ("CAR" if vtype != "ignore" else "IGNORE")

            # Salvar imagem somente quando o veículo é contado (ou seja, entrou)
            if counted and vtype != "ignore" and vid is not None:
                # Salva o recorte do veículo dentro da ROI
                save_vehicle_image(roi, x, y, w, h, label, vid)

            # Desenha a detecção no ROI
            cv2.rectangle(roi, (x, y), (x+w, y+h), BOUNDING_BOX_COLLOR, 2)
            cv2.putText(roi, label, (x, y-5), FONT, 0.7, (255,255,255), 2)

    # Devolve o ROI processado para o frame original
    frame[h1:h1 + h2, w1:w1 + w2] = roi

    # Obtém contagem atual
    cars, trucks = validator_instance.get_counts()

    # Exibe contagem no frame principal
    cv2.putText(frame, f"Cars Entered: {cars}", (20, 50), FONT, 1, (0,255,0), 2)
    cv2.putText(frame, f"Trucks Entered: {trucks}", (20, 100), FONT, 1, (0,165,255), 2)

    # Mostrar telas
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", fgmask)

    # Encerrar no Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
