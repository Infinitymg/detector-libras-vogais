"""Detecção das mãos com o MediaPipe.

Usa a API nova do MediaPipe (a "Tasks"). A API antiga (mp.solutions) saiu das
versões recentes, então um pip install novo já não traz mais ela.
Na primeira vez o modelo de mãos (uns 8 MB) é baixado pra modelos/.
"""
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_MODELO_MAO = RAIZ / "modelos" / "hand_landmarker.task"
URL_MODELO_MAO = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# pares de pontos que se ligam no "esqueleto" da mão (punho, polegar, indicador...)
CONEXOES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def garantir_modelo():
    """Baixa o modelo de mãos se ainda não estiver na pasta modelos/."""
    if CAMINHO_MODELO_MAO.exists():
        return
    CAMINHO_MODELO_MAO.parent.mkdir(exist_ok=True)
    temporario = CAMINHO_MODELO_MAO.with_suffix(".part")
    print("Baixando o modelo de mãos do MediaPipe (só na primeira vez)...")
    try:
        urllib.request.urlretrieve(URL_MODELO_MAO, temporario)
        temporario.replace(CAMINHO_MODELO_MAO)
    except Exception as erro:
        temporario.unlink(missing_ok=True)
        raise RuntimeError(
            "Não consegui baixar o modelo de mãos. Baixa na mão por este link e salva como "
            f"{CAMINHO_MODELO_MAO}:\n{URL_MODELO_MAO}"
        ) from erro


class DetectorDeMaos:
    def __init__(self, confianca=0.7, max_maos=2):
        garantir_modelo()
        opcoes = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(CAMINHO_MODELO_MAO)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_maos,
            min_hand_detection_confidence=confianca,
            min_hand_presence_confidence=confianca,
            min_tracking_confidence=confianca,
        )
        self._detector = vision.HandLandmarker.create_from_options(opcoes)
        self._inicio = time.monotonic()
        self._ultimo_ms = -1

    def detectar(self, img_bgr):
        """Devolve uma lista de mãos. Cada mão é uma lista de 21 pontos (com .x e .y)."""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ms = int((time.monotonic() - self._inicio) * 1000)
        ms = max(ms, self._ultimo_ms + 1)  # o MediaPipe exige tempo sempre crescente
        self._ultimo_ms = ms
        return self._detector.detect_for_video(imagem, ms).hand_landmarks

    def fechar(self):
        self._detector.close()


def desenhar_mao(img, mao):
    """Desenha o esqueleto da mão em cima da imagem."""
    altura, largura = img.shape[:2]
    pontos = [(int(p.x * largura), int(p.y * altura)) for p in mao]
    for a, b in CONEXOES:
        cv2.line(img, pontos[a], pontos[b], (0, 255, 0), 2)
    for ponto in pontos:
        cv2.circle(img, ponto, 4, (0, 0, 255), -1)
