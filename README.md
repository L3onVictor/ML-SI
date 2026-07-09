# ML-SI
 Projeto - Aprendizagem de máquina

# ☕ Sistema Inteligente de Classificação — Qualidade do Café

Projeto de classificação da qualidade de cafés (Arábica) utilizando **KNN** e **Árvore de Decisão**, com base em dados sensoriais do Coffee Quality Institute (CQI).

## 📁 Estrutura do Projeto

Apenas para exemplificação, não precisa seguir a risca

```
├── data/
│   └── df_arabica_clean                # CSV original (df_arabica_clean.csv)
├── notebooks/               # Análise exploratória e experimentação
├── preprocessing/           # Scripts de limpeza e transformação dos dados
├── models/                  # Treinamento e modelos salvos (.pkl)
├── evaluation/               # Métricas, matrizes de confusão, gráficos
├── requirements.txt
└── README.md
```

## 🔧 Pré-requisitos

- Python 3.10+ instalado

## 🚀 Setup do Ambiente

### 1. Clonar o repositório
```bash
git clone <url-do-repositorio>
cd <nome-do-repositorio>
```

### 2. Criar o ambiente virtual

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

> ✅ Você saberá que o ambiente está ativo quando `(venv)` aparecer no início da linha do terminal.

### 3. Instalar as dependências
```bash
pip install -r requirements.txt
```

### 4. Adicionar o dataset

Houve mudança no dataset pois o antigo havia apenas 206 linhas, apesar de estar melhor estruturado. Ele pode ser encontrado no [Kagle](https://www.kaggle.com/datasets/fatihb/coffee-quality-data-cqi), sendo a tabela `df_arabica_clean.csv` que se encontra no arquivo ```data/older_data```.

Baixe o `arabica_coffee_full_table.csv` (fonte: [Coffee Quality Data CQI - Kaggle](https://www.kaggle.com/datasets/erwinhmtang/coffee-quality-institute-reviews-may2023?select=arabica_coffee_full_table.csv)) e coloque em:
```
data/raw/arabica_coffee_full_table.csv
```

## ▶️ Executando o projeto

```bash
jupyter notebook
```
Ou, se estiver usando scripts `.py`:
```bash
python preprocessing/preprocess.py
python models/train.py
python evaluation/evaluate.py
```

## 📦 Desativar o ambiente virtual (quando terminar)
```bash
deactivate
```

## 🧠 Algoritmos utilizados

- **KNN (K-Nearest Neighbors)** — testado com K = 3, 5, 7 e distâncias Euclidiana/Manhattan
- **Árvore de Decisão** — avaliação de `max_depth` para controle de overfitting

## 📊 Métricas avaliadas

Acurácia, Precisão, Recall, F1-Score e Matriz de Confusão para cada modelo.