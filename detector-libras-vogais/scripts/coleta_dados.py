"""Coleta de exemplos pela webcam.

Põe a mão na frente da câmera, faz a vogal e aperta a tecla correspondente
(a, e, i, o, u). Cada aperto salva os 42 números da mão (x e y dos 21 pontos)
mais a letra em dados/dataset.csv. Aperta 'q' pra sair.
"""
import csv
from pathlib import Path

import cv2

from landmarks import COLUNAS, pontos_da_mao
from maos import DetectorDeMaos, desenhar_mao

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_CSV = RAIZ / "dados" / "dataset.csv"
LETRAS = "AEIOU"


def contar_exemplos(caminho=CAMINHO_CSV):
    """Quantos exemplos já existem de cada letra."""
    contagem = {letra: 0 for letra in LETRAS}
    if caminho.exists():
        with open(caminho, newline="", encoding="utf-8") as f:
            for linha in csv.DictReader(f):
                if linha.get("label") in contagem:
                    contagem[linha["label"]] += 1
    return contagem


def salvar_exemplo(letra, pontos, caminho=CAMINHO_CSV):
    """Adiciona uma linha no CSV (cria o arquivo com cabeçalho se for novo)."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    arquivo_novo = not caminho.exists() or caminho.stat().st_size == 0
    with open(caminho, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if arquivo_novo:
            writer.writerow(COLUNAS + ["label"])
        writer.writerow(list(pontos) + [letra])


def escrever(img, texto, y, cor=(255, 255, 255)):
    # o putText do OpenCV não desenha acento, por isso o texto na tela vai sem
    cv2.putText(img, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)


def main():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Não foi possível abrir a câmera. Veja se ela está conectada e livre.")

    detector = DetectorDeMaos()
    contagem = contar_exemplos()
    aviso, frames_aviso = "", 0

    print("Como usar:")
    print("1. Põe uma mão só na frente da câmera.")
    print("2. Faz a vogal (A, E, I, O ou U).")
    print("3. Aperta a tecla da letra no teclado pra salvar.")
    print("4. Aperta 'q' pra sair.")

    try:
        while True:
            ok, img = camera.read()
            if not ok:
                break

            # espelha a imagem. A detecção faz a mesma coisa, os dois têm que bater
            img = cv2.flip(img, 1)
            maos = detector.detectar(img)
            for mao in maos:
                desenhar_mao(img, mao)

            resumo = "  ".join(f"{letra}:{n}" for letra, n in contagem.items())
            escrever(img, f"Exemplos  {resumo}", 25)
            escrever(img, "a/e/i/o/u salva  |  q sai", img.shape[0] - 15)
            if frames_aviso > 0:
                escrever(img, aviso, 55, (0, 255, 255))
                frames_aviso -= 1

            cv2.imshow("Coleta de dados - vogais", img)
            tecla = cv2.waitKey(1) & 0xFF

            if tecla == ord("q"):
                break
            if tecla < 128 and chr(tecla).upper() in LETRAS:
                letra = chr(tecla).upper()
                if len(maos) == 1:
                    salvar_exemplo(letra, pontos_da_mao(maos[0]))
                    contagem[letra] += 1
                    aviso = f"Salvo: {letra} ({contagem[letra]})"
                else:
                    aviso = "Precisa de exatamente 1 mao na imagem"
                frames_aviso = 30
    finally:
        camera.release()
        detector.fechar()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
