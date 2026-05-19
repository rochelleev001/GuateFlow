"""
train.py
Entrena y compara 4 clasificadores. Guarda el mejor modelo (Gradient Boosting).
Uso: python src/train.py
"""
import pandas as pd
import numpy as np
import joblib, os
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from preprocess import load_and_preprocess

def train():
    X_train, X_test, y_train, y_test, le, scaler, features = load_and_preprocess()

    modelos = {
        "Gradient Boosting":  GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                                          max_depth=4, random_state=42),
        "Random Forest":      RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
        "Decision Tree":      DecisionTreeClassifier(max_depth=8, random_state=42),
        "Logistic Regression":LogisticRegression(max_iter=1000, random_state=42),
    }

    resultados = []
    mejor_f1   = 0
    mejor_modelo = None

    print("\n" + "="*60)
    print("  COMPARATIVA DE MODELOS — GuateFlow")
    print("="*60)

    for nombre, clf in modelos.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro")
        resultados.append({"Modelo": nombre, "Accuracy": acc, "F1-Score (macro)": f1})
        print(f"\n{nombre}")
        print(f"  Accuracy : {acc:.4f}   F1-macro : {f1:.4f}")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_modelo = clf

    # --- Guardar el mejor modelo ---
    os.makedirs("model", exist_ok=True)
    joblib.dump(mejor_modelo, "model/guateflow_model.pkl")
    print("\n" + "="*60)
    print(f"  Modelo guardado: model/guateflow_model.pkl")
    print(f"  Mejor F1-Score : {mejor_f1:.4f}")
    print("="*60)

    # --- Tabla resumen ---
    df_res = pd.DataFrame(resultados).sort_values("F1-Score (macro)", ascending=False)
    print("\nResumen:\n", df_res.to_string(index=False))

if __name__ == "__main__":
    train()
