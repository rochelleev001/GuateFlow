"""
app.py
API REST con Flask que expone el modelo GuateFlow.
Uso: python src/app.py
Endpoint: POST /predict   Body JSON: {"zona","hora","dia_semana","clima","evento_especial"}
"""
from flask import Flask, request, jsonify
import joblib, pandas as pd, os, sys

app = Flask(__name__)

# Cargar artefactos al iniciar
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model    = joblib.load(os.path.join(BASE, "model/guateflow_model.pkl"))
le       = joblib.load(os.path.join(BASE, "model/label_encoder.pkl"))
scaler   = joblib.load(os.path.join(BASE, "model/scaler.pkl"))
features = joblib.load(os.path.join(BASE, "model/feature_names.pkl"))

ZONAS  = ["Zona 1","Zona 4","Zona 10","Zona 12","Calzada Roosevelt","Periférico"]
DIAS   = ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]
CLIMAS = ["Soleado","Nublado","Lluvia"]

def build_row(zona, hora, dia, clima, evento):
    row = {f: 0 for f in features}
    row["hora"]            = scaler.transform([[hora]])[0][0]
    row["evento_especial"] = int(evento)
    for z in ZONAS:
        k = f"zona_{z}";        row[k] = 1 if z == zona  else 0
    for d in DIAS:
        k = f"dia_semana_{d}";  row[k] = 1 if d == dia   else 0
    for c in CLIMAS:
        k = f"clima_{c}";       row[k] = 1 if c == clima  else 0
    return pd.DataFrame([row])[features]

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "GuateFlow API activa", "version": "1.0"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data   = request.get_json(force=True)
        zona   = data.get("zona","Zona 10")
        hora   = int(data.get("hora", 8))
        dia    = data.get("dia_semana","Lunes")
        clima  = data.get("clima","Soleado")
        evento = int(data.get("evento_especial", 0))

        X        = build_row(zona, hora, dia, clima, evento)
        pred_enc = model.predict(X)[0]
        proba    = model.predict_proba(X)[0]
        nivel    = le.inverse_transform([pred_enc])[0]

        return jsonify({
            "nivel_trafico": nivel,
            "probabilidades": {cls: round(float(p)*100,1)
                               for cls, p in zip(le.classes_, proba)},
            "inputs": {"zona":zona,"hora":hora,"dia_semana":dia,
                       "clima":clima,"evento_especial":evento}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
