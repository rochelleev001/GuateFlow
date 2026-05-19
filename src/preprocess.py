"""
preprocess.py
Pipeline de limpieza, codificación y escalado del dataset GuateFlow.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib, os

def load_and_preprocess(raw_path="data/raw/trafico_guatemala.csv"):
    df = pd.read_csv(raw_path)

    # --- Codificación One-Hot de variables categóricas ---
    df_enc = pd.get_dummies(df, columns=["zona","dia_semana","clima"], drop_first=False)

    # --- Codificación de la variable objetivo ---
    le = LabelEncoder()
    df_enc["nivel_trafico_enc"] = le.fit_transform(df_enc["nivel_trafico"])

    # --- Separar X / y ---
    feature_cols = [c for c in df_enc.columns if c not in ["nivel_trafico","nivel_trafico_enc"]]
    X = df_enc[feature_cols].astype(float)
    y = df_enc["nivel_trafico_enc"]

    # --- Escalar columna hora ---
    scaler = StandardScaler()
    X["hora"] = scaler.fit_transform(X[["hora"]])

    # --- Split train / test ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # --- Guardar artefactos ---
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv",   index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv",   index=False)
    joblib.dump(le,      "model/label_encoder.pkl")
    joblib.dump(scaler,  "model/scaler.pkl")
    joblib.dump(list(X.columns), "model/feature_names.pkl")

    print(f"Preprocesamiento completo. Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, le, scaler, list(X.columns)

if __name__ == "__main__":
    load_and_preprocess()
