# Detector de Libras — Vogais

Projeto em grupo feito para a **Amostra Tech** da Uniube. O sistema reconhece as vogais **A, E, I, O, U** do alfabeto de Libras usando a webcam: o MediaPipe encontra os 21 pontos da mão e um modelo de Machine Learning (Random Forest) decide qual vogal está sendo feita.

## Como funciona

1. **Coleta** — a webcam mostra a mão com os 21 pontos; a cada tecla apertada (`a`, `e`, `i`, `o`, `u`) o exemplo é salvo em `dados/dataset.csv`.
2. **Normalização** — o pulso vira a origem e tudo é dividido pela distância entre o pulso e a base do dedo médio. Assim o modelo olha o *formato* da mão e não onde ela está na tela, nem quão perto da câmera.
3. **Treino** — Random Forest (scikit-learn), avaliado com validação cruzada.
4. **Detecção ao vivo** — mostra a letra prevista e a confiança. Abaixo de 50% de confiança aparece `?`, e a letra é votada entre as últimas 7 previsões para não ficar piscando.

## Resultados

Dataset: 100 exemplos (20 por letra). Validação cruzada com 5 dobras repetida 10 vezes:

| Entrada | Acurácia |
|---|---|
| Pontos crus | ≈ 74% |
| Pontos normalizados | ≈ 99% |

A normalização foi a melhora que mais fez diferença.

**Limitações (importante):**
- Os 100 exemplos foram coletados em uma única sessão, então o ~99% é um número otimista. Em outra pessoa, outra luz ou outra câmera o resultado tende a ser menor.
- Só reconhece vogais estáticas (sem movimento) e com uma mão só.
- Pra melhorar: coletar com várias pessoas, ângulos e iluminações, e avaliar separando por pessoa.

## Estrutura

```
detector-libras-vogais/
├── dados/
│   └── dataset.csv          # exemplos coletados (x0..x20, y0..y20, label)
├── scripts/
│   ├── main.py              # roda o fluxo: (coleta) -> valida -> treina -> (detecta)
│   ├── coleta_dados.py      # coleta pela webcam
│   ├── treinar_modelo.py    # treina e avalia o modelo
│   ├── detectar.py          # reconhecimento ao vivo
│   ├── landmarks.py         # funções em comum (extrair e normalizar pontos)
│   └── maos.py              # detecção das mãos com o MediaPipe
├── requirements.txt
└── README.md
```

## Como rodar

Precisa de Python 3.10 ou mais novo.

```bash
git clone https://github.com/Infinitymg/detector-libras-vogais.git
cd detector-libras-vogais

python -m venv venv
# Windows:  venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

pip install -r requirements.txt
```

```bash
python scripts/main.py              # treina com o dataset que já vem no projeto
python scripts/main.py --detectar   # treina e abre o detector ao vivo (precisa de webcam)
python scripts/main.py --coletar    # abre a webcam pra coletar mais exemplos antes de treinar
```

Na primeira vez que a câmera for usada, o modelo de mãos do MediaPipe (uns 8 MB) é baixado sozinho para a pasta `modelos/`. Precisa de internet nessa hora.

> O MediaPipe removeu a API antiga (`mp.solutions`) das versões recentes. Este projeto usa a API nova (*Tasks*), por isso funciona com uma instalação nova.

## Tecnologias

Python · MediaPipe · OpenCV · scikit-learn · pandas · NumPy

## Equipe

- Matheus Marques Gomes — [@Infinitymg](https://github.com/Infinitymg)
- _Nome do integrante_ — _o que fez_
- _Nome do integrante_ — _o que fez_

Projeto apresentado na Amostra Tech — Uniube, curso de Sistemas de Informação.
