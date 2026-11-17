"""Reconhece ao vivo, pela webcam, a vogal que você faz com a mão.

Precisa do modelo treinado antes (python scripts/main.py). Aperta 'q' pra sair.
"""
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib

from landmarks import normalizar, pontos_da_mao
from maos import DetectorDeMaos, desenhar_mao

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_MODELO = RAIZ / "modelos" / "modelo_libras.pkl"

LIMIAR_CONFIANCA = 0.5  # abaixo disso aparece "?" em vez de chutar uma letra
JANELA = 7  # vota entre as últimas 7 previsões, senão a letra fica piscando


def prever(modelo, mao):
    """Devolve (letra, confiança) pra uma mão."""
    x = normalizar(pontos_da_mao(mao))
    probabilidades = modelo.predict_proba(x)[0]
    i = probabilidades.argmax()
    return str(modelo.classes_[i]), float(probabilidades[i])


def escrever(img, texto, y, cor=(255, 255, 255)):
    # o putText do OpenCV não desenha acento, por isso o texto na tela vai sem
    cv2.putText(img, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)


def main():
    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError("Não achei o modelo. Treina antes: python scripts/main.py")
    modelo = joblib.load(CAMINHO_MODELO)

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Não foi possível abrir a câmera. Veja se ela está conectada e livre.")

    detector = DetectorDeMaos()
    recentes = deque(maxlen=JANELA)

    try:
        while True:
            ok, img = camera.read()
            if not ok:
                break

            # espelha, igual na coleta (os dois têm que bater)
            img = cv2.flip(img, 1)
            maos = detector.detectar(img)

            texto = "Mostre uma mao"
            if len(maos) == 1:
                desenhar_mao(img, maos[0])
                letra, confianca = prever(modelo, maos[0])
                recentes.append(letra if confianca >= LIMIAR_CONFIANCA else "?")
                escolhida = Counter(recentes).most_common(1)[0][0]
                texto = f"Letra: {escolhida}  ({confianca:.0%})"
            else:
                recentes.clear()
                if len(maos) > 1:
                    texto = "Use so uma mao"

            escrever(img, texto, 40, (0, 255, 0))
            escrever(img, "q sai", img.shape[0] - 15)
            cv2.imshow("Detector de Libras - vogais", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        detector.fechar()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
