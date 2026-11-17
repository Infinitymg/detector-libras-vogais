"""Roda o fluxo do projeto: (coleta) -> valida o dataset -> treina o modelo.

Exemplos:
    python scripts/main.py              treina com o dataset que já existe
    python scripts/main.py --coletar    abre a webcam pra coletar mais exemplos antes
    python scripts/main.py --detectar   depois de treinar, abre o detector ao vivo
"""
import argparse
import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PASTA_SCRIPTS = RAIZ / "scripts"
ARQUIVO_DATASET = RAIZ / "dados" / "dataset.csv"
LETRAS_ESPERADAS = {"A", "E", "I", "O", "U"}
MINIMO_POR_LETRA = 10


def avisar_se_sem_venv():
    # só avisa, não trava: o venv é recomendado, mas dá pra rodar sem ele
    em_venv = sys.prefix != sys.base_prefix
    if not em_venv:
        print("Aviso: o ambiente virtual (venv) não está ativo. Recomendo ativar antes de instalar as dependências.\n")


def rodar_script(nome, descricao):
    print(f"--- {descricao} ---")
    caminho = PASTA_SCRIPTS / nome
    if not caminho.exists():
        print(f"Erro: não encontrei {caminho}")
        return False
    try:
        # cwd na raiz do projeto, assim os caminhos não dependem de onde o terminal estava
        subprocess.run([sys.executable, str(caminho)], check=True, cwd=RAIZ)
    except subprocess.CalledProcessError as erro:
        print(f"Falha ao executar {nome}: {erro}\n")
        return False
    print(f"{descricao}: ok\n")
    return True


def validar_dados():
    print("--- Validando o dataset ---")
    if not ARQUIVO_DATASET.exists() or ARQUIVO_DATASET.stat().st_size == 0:
        print(f"Erro: {ARQUIVO_DATASET} não existe ou está vazio.")
        return False

    with open(ARQUIVO_DATASET, newline="", encoding="utf-8") as f:
        contagem = Counter(linha["label"] for linha in csv.DictReader(f) if linha.get("label"))

    faltando = LETRAS_ESPERADAS - set(contagem)
    if faltando:
        print(f"Erro: faltam exemplos das letras {sorted(faltando)}")
        return False

    print("Exemplos por letra:", dict(sorted(contagem.items())))
    poucos = sorted(letra for letra, n in contagem.items() if n < MINIMO_POR_LETRA)
    if poucos:
        print(f"Aviso: menos de {MINIMO_POR_LETRA} exemplos em {poucos}. O resultado pode ficar fraco.")
    print()
    return True


def main():
    parser = argparse.ArgumentParser(description="Detector de Libras (vogais): coleta, treino e detecção.")
    parser.add_argument("--coletar", action="store_true", help="abre a webcam pra coletar mais exemplos antes de treinar")
    parser.add_argument("--detectar", action="store_true", help="abre o detector ao vivo depois de treinar")
    args = parser.parse_args()

    avisar_se_sem_venv()

    # sem dataset não tem o que treinar, então a coleta roda mesmo sem a opção
    if args.coletar or not ARQUIVO_DATASET.exists():
        if not rodar_script("coleta_dados.py", "Coleta de dados"):
            sys.exit(1)

    if not validar_dados():
        sys.exit(1)

    if not rodar_script("treinar_modelo.py", "Treinamento do modelo"):
        sys.exit(1)

    if args.detectar:
        rodar_script("detectar.py", "Detector ao vivo")

    print("Pronto.")


if __name__ == "__main__":
    main()
