"""
predict.py
Realiza una predicción individual usando el modelo guardado.
Uso:
  python src/predict.py --zona "Zona 10" --hora 8 --dia "Lunes" --clima "Lluvia" --evento 0
"""
import argparse, joblib, pandas as pd, numpy as np, os

ZONAS  = ["Zona 1","Zona 4","Zona 10","Zona 12","Calzada Roosevelt","Periférico"]
DIAS   = ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]
CLIMAS = ["Soleado","Nublado","Lluvia"]

def predecir(zona, hora, dia, clima, evento):
    model    = joblib.load("model/guateflow_model.pkl")
    le       = joblib.load("model/label_encoder.pkl")
    scaler   = joblib.load("model/scaler.pkl")
    features = joblib.load("model/feature_names.pkl")

    # Construir fila con las mismas columnas del entrenamiento
    row = {f: 0 for f in features}
    hora_scaled = scaler.transform([[hora]])[0][0]
    row["hora"] = hora_scaled
    row["evento_especial"] = int(evento)
    for z in ZONAS:
        key = f"zona_{z}"
        if key in row: row[key] = 1 if z == zona else 0
    for d in DIAS:
        key = f"dia_semana_{d}"
        if key in row: row[key] = 1 if d == dia else 0
    for c in CLIMAS:
        key = f"clima_{c}"
        if key in row: row[key] = 1 if c == clima else 0

    X = pd.DataFrame([row])[features]
    pred_enc = model.predict(X)[0]
    proba    = model.predict_proba(X)[0]
    nivel    = le.inverse_transform([pred_enc])[0]

    print(f"\n{'='*40}")
    print(f"  GuateFlow — Predicción de Tráfico")
    print(f"{'='*40}")
    print(f"  Zona    : {zona}")
    print(f"  Hora    : {hora}:00")
    print(f"  Día     : {dia}")
    print(f"  Clima   : {clima}")
    print(f"  Evento  : {'Sí' if evento else 'No'}")
    print(f"{'='*40}")
    print(f"  NIVEL PREDICHO : {nivel.upper()}")
    print(f"{'='*40}")
    for i, cls in enumerate(le.classes_):
        print(f"    {cls:10s}: {proba[i]*100:.1f}%")
    return nivel

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GuateFlow — Predictor de tráfico")
    p.add_argument("--zona",   default="Zona 10",  choices=ZONAS)
    p.add_argument("--hora",   type=int, default=8)
    p.add_argument("--dia",    default="Lunes",    choices=DIAS)
    p.add_argument("--clima",  default="Soleado",  choices=CLIMAS)
    p.add_argument("--evento", type=int, default=0, choices=[0,1])
    args = p.parse_args()
    predecir(args.zona, args.hora, args.dia, args.clima, args.evento)
