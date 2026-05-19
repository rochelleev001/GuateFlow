"""
generate_dataset.py
Genera el dataset simulado de tráfico para Ciudad de Guatemala.
Ejecutar una sola vez antes de entrenar el modelo.
"""
import pandas as pd
import numpy as np
import os

SEED = 42
np.random.seed(SEED)

ZONAS = ["Zona 1", "Zona 4", "Zona 10", "Zona 12", "Calzada Roosevelt", "Periférico"]
DIAS  = ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]
CLIMAS = ["Soleado", "Nublado", "Lluvia"]

N = 5000

def asignar_nivel(zona, hora, dia, clima, evento):
    score = 0
    if hora in range(6,10) or hora in range(16,20): score += 3
    elif hora in range(10,16): score += 1
    else: score -= 1
    if dia in ["Lunes","Martes","Miércoles","Jueves","Viernes"]: score += 2
    elif dia == "Sábado": score += 1
    if zona in ["Calzada Roosevelt","Zona 1","Periférico"]: score += 2
    elif zona in ["Zona 10","Zona 12"]: score += 1
    if clima == "Lluvia": score += 2
    elif clima == "Nublado": score += 1
    if evento == 1: score += 2
    score += np.random.randint(-1,2)
    if score <= 3: return "Bajo"
    elif score <= 6: return "Moderado"
    else: return "Alto"

rows = []
for _ in range(N):
    zona   = np.random.choice(ZONAS)
    hora   = np.random.randint(0,24)
    dia    = np.random.choice(DIAS)
    clima  = np.random.choice(CLIMAS, p=[0.55,0.30,0.15])
    evento = np.random.choice([0,1], p=[0.90,0.10])
    nivel  = asignar_nivel(zona, hora, dia, clima, evento)
    rows.append([zona, hora, dia, clima, evento, nivel])

df = pd.DataFrame(rows, columns=["zona","hora","dia_semana","clima","evento_especial","nivel_trafico"])
os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/trafico_guatemala.csv", index=False)
print(f"Dataset generado: {len(df)} registros")
print(df["nivel_trafico"].value_counts())
