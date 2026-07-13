# ML-SI

# ☕ Sistema Inteligente de Classificação de Café

Este projeto reúne um fluxo completo de análise, treinamento e uso de modelos de machine learning para classificar cafés em categorias como Tradicional, Superior e Gourmet.

A aplicação principal está em [app.py](app.py) e usa modelos treinados com base em dois conjuntos de dados:

- Tabela A: sem notas sensoriais
- Tabela B: com notas sensoriais

## 📁 Estrutura do projeto

```text
ML-SI/
├── app.py                  # Interface em Streamlit para classificação
├── requirements.txt       # Dependências do projeto
├── README.md               # Documentação geral
├── data/
│   ├── old_data/           # Dados antigos e versões anteriores
│   ├── processed/          # Arquivos CSV prontos para treino e uso
│   └── raw/                # Dados brutos originais
├── models/                 # Modelos treinados em joblib e metadados em JSON
└── notebooks/              # Notebooks de EDA, experimentação e treinamento
```

## 🧠 O que o projeto faz

- Explora os dados em notebooks para entender as features e a distribuição das classes.
- Treina modelos de classificação com diferentes abordagens.
- Disponibiliza uma interface interativa para o usuário inserir características de um café e obter uma previsão.

## 🔧 Pré-requisitos

- Python 3.10 ou superior
- pip
- Ambiente virtual recomendado

## 🚀 Como configurar o ambiente

No terminal, na raiz do projeto:

### 1. Criar o ambiente virtual

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Linux/Mac:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar as dependências

```bash
pip install -r requirements.txt
```

## ▶️ Como rodar a interface

A interface é executada com Streamlit:

```bash
streamlit run app.py
```

Se quiser usar o interpretador do ambiente virtual explicitamente:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Depois disso, abra o endereço local mostrado no terminal, normalmente:

```text
http://localhost:8501
```

## 🖱️ Como usar a interface

1. Escolha o tipo de entrada:
   - Sem notas sensoriais
   - Com notas sensoriais
2. Selecione o modelo:
   - KNN
   - Random Forest
3. Preencha os campos com as características do café.
4. Clique em “Classificar café” para ver a previsão e as probabilidades por classe.

## 📦 Modelos e dados

Os modelos são salvos automaticamente na pasta [models](models) quando a interface é usada pela primeira vez. Se os arquivos ainda não existirem, a aplicação gera os modelos e os armazena para uso futuro.

## 📓 Notebooks

Os notebooks em [notebooks](notebooks) representam as etapas do projeto:

- EDA inicial e exploração dos dados
- Preparação dos datasets processados
- Treinamento de modelos com KNN
- Treinamento de modelos com árvores e random forest

## 🧪 Observação

A aplicação usa as mesmas estruturas de features que foram empregadas durante o treinamento, por isso a escolha entre as duas tabelas influencia diretamente quais campos aparecem na tela.