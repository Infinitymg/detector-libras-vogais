"""Funções que a coleta, o treino e a detecção usam em comum.

Ficam num arquivo só pra os três fazerem exatamente a mesma conta. Se a coleta
e a detecção tirarem os pontos de jeitos diferentes, o modelo começa a errar.
"""
import numpy as np

N_PONTOS = 21  # o MediaPipe devolve 21 pontos por mão
COLUNAS_X = [f"x{i}" for i in range(N_PONTOS)]
COLUNAS_Y = [f"y{i}" for i in range(N_PONTOS)]
COLUNAS = COLUNAS_X + COLUNAS_Y  # mesma ordem do dataset.csv

PULSO = 0
BASE_DEDO_MEDIO = 9


def pontos_da_mao(mao):
    """Pega os 21 pontos de uma mão (cada um com .x e .y) e devolve [x0..x20, y0..y20].

    Os valores vêm crus, de 0 a 1 dentro da imagem.
    """
    xs = [p.x for p in mao]
    ys = [p.y for p in mao]
    return xs + ys


def normalizar(dados):
    """Normaliza um ou vários exemplos (formato x0..x20, y0..y20).

    O pulso vira a origem e tudo é dividido pela distância entre o pulso e a
    base do dedo médio. Assim o modelo olha pro formato da mão, e não pra onde
    ela está na tela nem pra quão perto da câmera ela fica.
    """
    dados = np.atleast_2d(np.asarray(dados, dtype=float))
    xs = dados[:, :N_PONTOS]
    ys = dados[:, N_PONTOS:]

    dx = xs - xs[:, [PULSO]]
    dy = ys - ys[:, [PULSO]]

    tamanho = np.sqrt(dx[:, [BASE_DEDO_MEDIO]] ** 2 + dy[:, [BASE_DEDO_MEDIO]] ** 2)
    tamanho[tamanho == 0] = 1.0  # evita dividir por zero

    return np.hstack([dx / tamanho, dy / tamanho])
