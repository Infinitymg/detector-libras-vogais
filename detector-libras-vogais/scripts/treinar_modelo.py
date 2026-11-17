"""Treina o modelo que reconhece as vogais a partir de dados/dataset.csv.

Passos: lê o dataset, normaliza os pontos da mão, avalia com validação cruzada,
treina o modelo final com todos os exemplos e salva em modelos/.
"""
import json
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split

from landmarks import COLUNAS, normalizar

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_CSV = RAIZ / "dados" / "dataset.csv"
PASTA_MODELOS = RAIZ / "modelos"


def novo_modelo():
    # Sem busca de hiperparâmetros de propósito: com 100 exemplos ela escolheria
    # os parâmetros mais por sorte do que por mérito.
    return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)


def carregar_dados():
    if not CAMINHO_CSV.exists():
        raise FileNotFoundError(
            f"Não achei {CAMINHO_CSV}. Roda a coleta antes (python scripts/main.py --coletar)."
        )
    df = pd.read_csv(CAMINHO_CSV)

    faltando = [c for c in COLUNAS + ["label"] if c not in df.columns]
    if faltando:
        raise ValueError(f"Faltam colunas no dataset: {faltando}")

    df = df.dropna().reset_index(drop=True)
    X = df[COLUNAS].astype(float).to_numpy()  # dá erro se tiver valor que não é número
    y = df["label"].astype(str).to_numpy()
    return X, y


def validacao_cruzada(X, y, n_dobras):
    cv = RepeatedStratifiedKFold(n_splits=n_dobras, n_repeats=10, random_state=42)
    notas = cross_val_score(novo_modelo(), X, y, cv=cv)
    return notas.mean(), notas.std()


def main():
    X_cru, y = carregar_dados()
    por_letra = Counter(y)
    letras = sorted(por_letra)
    print(f"{len(y)} exemplos. Por letra: {dict(sorted(por_letra.items()))}")

    menor = min(por_letra.values())
    if menor < 5:
        raise ValueError(f"Precisa de pelo menos 5 exemplos por letra (a menor tem {menor}).")
    n_dobras = 5

    X = normalizar(X_cru)

    # Compara pontos crus x normalizados só pra ver se a normalização ajuda
    media_crua, _ = validacao_cruzada(X_cru, y, n_dobras)
    media, desvio = validacao_cruzada(X, y, n_dobras)
    print(f"\nValidação cruzada ({n_dobras} dobras x 10 repetições):")
    print(f"  pontos crus:   {media_crua:.1%}")
    print(f"  normalizados:  {media:.1%} (desvio {desvio:.1%})")

    # Divisão 80/20 só pra ver o desempenho letra por letra
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_pred = novo_modelo().fit(X_treino, y_treino).predict(X_teste)
    print("\nRelatório por letra (20% dos exemplos separados pra teste):")
    print(classification_report(y_teste, y_pred, zero_division=0))
    print(f"Matriz de confusão (linhas = letra real, colunas = letra prevista, ordem {letras}):")
    print(confusion_matrix(y_teste, y_pred, labels=letras))

    # Modelo final: treina com todos os exemplos
    modelo = novo_modelo().fit(X, y)
    PASTA_MODELOS.mkdir(exist_ok=True)
    joblib.dump(modelo, PASTA_MODELOS / "modelo_libras.pkl")

    metadata = {
        "classes": letras,
        "n_exemplos": int(len(y)),
        "exemplos_por_letra": dict(sorted(por_letra.items())),
        "modelo": "RandomForestClassifier(n_estimators=200)",
        "normalizacao": "pulso como origem, dividido pela distância pulso -> base do dedo médio",
        "validacao_cruzada_normalizado": round(float(media), 4),
        "validacao_cruzada_desvio": round(float(desvio), 4),
        "validacao_cruzada_pontos_crus": round(float(media_crua), 4),
        "sklearn_version": sklearn.__version__,
    }
    caminho_meta = PASTA_MODELOS / "metadata_modelo_libras.json"
    caminho_meta.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nTreinamento concluído. Modelo salvo em modelos/modelo_libras.pkl")


if __name__ == "__main__":
    main()
