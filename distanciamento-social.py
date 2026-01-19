import numpy as np
import cv2
import sys

# Cores e fontes para anotações visuais na tela
TEXT_COLOR = (0, 255, 0)
TRACKER_COLOR = (255, 0, 0)

#quando identificado duas pessoas proximas o contorno com a cor definida
WARNING_COLLOR = (24,201,255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Caminho do vídeo e tipo de background subtractor (requer opencv-contrib se usar GMG/MOG)
VIDEO_SOURCE = "video/distanciamento.mp4"
BGS_TYPES = ["GMG", "MOG", "MOG2", "KNN", "CNT"]
BGS_TYPES = BGS_TYPES[1]

def getKernerl(KERNEL_TYPE):
    # Retorna kernels para operações morfológicas:
    # - dilation: estrutura elíptica (melhor para crescer regiões)
    # - opening/closing: quadrado 3x3 de uns (remoção de ruído/fechamento de buracos)
    if KERNEL_TYPE == "dilation":
        kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    if KERNEL_TYPE == "opening":
        kernel =  np.ones((3, 5), np.uint8)
    if KERNEL_TYPE == "closing":
        kernel =  np.ones((11, 11), np.uint8)

    return kernel

def getFilter(img, filter):
    # Encadeia filtros morfológicos para limpar a máscara de fundo
    if filter == "closing":
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, getKernerl("closing"), iterations=2)
    if filter == "opening":
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, getKernerl("opening"), iterations=2)
    if filter == "dilation":
        return cv2.dilate(img, getKernerl("dilation"), iterations=2)
    if filter == "combine":
        # Pipeline: closing -> opening -> dilation (preenche falhas, remove ruído e expande)
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, getKernerl("closing"), iterations=2)
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, getKernerl("opening"), iterations=2)
        return cv2.dilate(opening, getKernerl("dilation"), iterations=2)

def getBGSubtractor(BGS_TYPE):
    # Seleciona o algoritmo de subtração de fundo
    # Observação: GMG e MOG ficam em cv2.bgsegm (pacote opencv-contrib-python)
    if BGS_TYPE == "GMG":
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if BGS_TYPE == "MOG":
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    if BGS_TYPE == "MOG2":
        return cv2.createBackgroundSubtractorMOG2(detectShadows=False, varThreshold=100)
    if BGS_TYPE == "KNN":
        return cv2.createBackgroundSubtractorKNN()
    if BGS_TYPE == "CNT":
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    print("Detector inválido")
    sys.exit(1)

# Abre o vídeo de entrada
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("Erro: vídeo não encontrado!")
    sys.exit(1)

# Instancia o subtractor escolhido
bg_subtractor = getBGSubtractor(BGS_TYPES)
minArea = 400  # área mínima do contorno para considerar "movimento"
maxArea = 800


#controla somente o tamanho das janelas de exibição
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.namedWindow("BG Mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame", 800, 450)    # ajuste o tamanho aqui
cv2.resizeWindow("BG Mask", 800, 450)
cv2.moveWindow("Frame", 50, 50)        # opcional: posição das janelas
cv2.moveWindow("BG Mask", 900, 50)


def main():
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("Fim do vídeo.")
            break

        # Reduz resolução para acelerar processamento
        frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)
        bg_mask = bg_subtractor.apply(frame)
        bg_mask = getFilter(bg_mask, 'combine')
        bg_mask = cv2.medianBlur(bg_mask, 5)

        #extrração dos contornos
        (contours, hierarchy) = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Regra simples: só considera movimentos com área mínima
            if area >= minArea:
                x, y, w, h = cv2.boundingRect(cnt)

                #Alternativas visuais (descomente o que quiser ver/testar):
                cv2.drawContours(frame, cnt, 1, TEXT_COLOR, 10)
                cv2.drawContours(frame, cnt, 1, (255, 255, 255), 1)

                if area >= maxArea:
                    cv2.rectangle(frame, (x, y), (x + 120, y - 13), (49, 49, 49), -1)
                    cv2.putText(frame, 'Aviso distancimaneto', (x, y -2), FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.drawContours(frame, [cnt], -1, WARNING_COLLOR, 2)
                    cv2.drawContours(frame, [cnt], -1, (255, 255, 255), 1)

        # Combina frame original com máscara (útil para visualização do que foi mantido)
        result = cv2.bitwise_and(frame, frame, mask=bg_mask)


        # Janelas de visualização
        cv2.imshow("Frame", frame)
        cv2.imshow("BG Mask", result)

        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

main()
