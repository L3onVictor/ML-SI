import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

DATASETS = {
    "Tabela A": {
        "file": DATA_DIR / "coffee_classificado.csv",
        "numeric": [
            "Moisture",
            "Category_One_Defects",
            "Category_Two_Defects",
            "Quakers",
            "Altitude",
        ],
        "categorical": ["Variety", "Processing_Method", "Country_of_Origin", "Color"],
    },
    "Tabela B": {
        "file": DATA_DIR / "coffee_classificado_sensoriais.csv",
        "numeric": [
            "Moisture",
            "Category_One_Defects",
            "Category_Two_Defects",
            "Quakers",
            "Altitude",
            "Aroma",
            "Flavor",
            "Aftertaste",
            "Acidity",
            "Body",
            "Balance",
            "Uniformity",
            "Clean_Cup",
            "Sweetness",
        ],
        "categorical": ["Variety", "Processing_Method", "Country_of_Origin", "Color"],
    },
}

MODEL_TYPES = ["KNN", "Random Forest"]
DATASET_CHOICES = {
    "Sem notas sensoriais": "Tabela A",
    "Com notas sensoriais": "Tabela B",
}
CLASS_LABELS = ["Tradicional", "Superior", "Gourmet"]


def build_pipeline(model_type: str, numeric_features: list[str], categorical_features: list[str]):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    if model_type == "KNN":
        classifier = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
    else:
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            class_weight="balanced",
            random_state=42,
        )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def train_and_save_model(dataset_name: str, model_type: str):
    config = DATASETS[dataset_name]
    df = pd.read_csv(config["file"])
    X = df[config["numeric"] + config["categorical"]]
    y = df["Classe"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_pipeline(model_type, config["numeric"], config["categorical"])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    model_path = MODELS_DIR / f"{dataset_name.lower().replace(' ', '_')}_{model_type.lower().replace(' ', '_')}.joblib"
    joblib.dump(model, model_path)

    meta = {
        "dataset": dataset_name,
        "model_type": model_type,
        "accuracy": round(float(accuracy), 4),
        "features": config["numeric"] + config["categorical"],
    }
    with open(MODELS_DIR / f"{model_path.stem}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return model_path


def load_model(dataset_name: str, model_type: str):
    model_path = MODELS_DIR / f"{dataset_name.lower().replace(' ', '_')}_{model_type.lower().replace(' ', '_')}.joblib"
    meta_path = MODELS_DIR / f"{model_path.stem}.json"

    if not model_path.exists() or not meta_path.exists():
        model_path = train_and_save_model(dataset_name, model_type)

    return joblib.load(model_path)


def get_dataset_options(dataset_name: str):
    config = DATASETS[dataset_name]
    df = pd.read_csv(config["file"])
    options = {}
    for col in config["categorical"]:
        values = sorted([str(v) for v in df[col].dropna().unique()])
        options[col] = values
    return options


def build_input_frame(dataset_name: str, values: dict):
    config = DATASETS[dataset_name]
    row = {}
    for col in config["numeric"]:
        row[col] = float(values[col])
    for col in config["categorical"]:
        row[col] = values[col]
    return pd.DataFrame([row])


st.set_page_config(page_title="Classificador de Café", page_icon="☕", layout="wide")
st.title("☕ Classificador de Café")
st.markdown("Insira os atributos do seu café e veja a previsão do modelo selecionado.")

with st.sidebar:
    st.header("Configuração")
    dataset_choice = st.selectbox(
        "Escolha o tipo de entrada",
        list(DATASET_CHOICES.keys()),
        help="Sem notas sensoriais usa a Tabela A; com notas sensoriais usa a Tabela B.",
    )
    dataset_name = DATASET_CHOICES[dataset_choice]
    model_type = st.selectbox("Escolha o tipo de ML", MODEL_TYPES)
    st.caption("Tabela A: sem notas sensoriais. Tabela B: com notas sensoriais.")

options = get_dataset_options(dataset_name)

st.subheader(f"Modelo selecionado: {model_type} - {dataset_name}")

with st.form("coffee_form"):
    st.markdown("Preencha os atributos do café abaixo para gerar a previsão.")
    col1, col2 = st.columns(2)

    values = {}
    with col1:
        for col in DATASETS[dataset_name]["numeric"]:
            values[col] = st.number_input(col, value=0.0, step=0.1)

    with col2:
        for col in DATASETS[dataset_name]["categorical"]:
            values[col] = st.selectbox(col, options[col])

    submitted = st.form_submit_button("Classificar café")

if submitted:
    with st.spinner("Carregando modelo e fazendo a previsão..."):
        model = load_model(dataset_name, model_type)
        input_df = build_input_frame(dataset_name, values)
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

    st.success("Previsão concluída")
    st.metric("Classe prevista", prediction)

    prob_df = pd.DataFrame(
        {
            "Classe": CLASS_LABELS,
            "Probabilidade": probabilities,
        }
    ).sort_values("Probabilidade", ascending=False)

    st.bar_chart(prob_df.set_index("Classe"))
    st.dataframe(prob_df.round(4), use_container_width=True)

    st.caption("As entradas são processadas usando o mesmo tipo de pré-processamento adotado nos notebooks.")
