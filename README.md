# GuateFlow — Predictor de Congestión Vial

> Sistema de predicción de tráfico urbano basado en Machine Learning para zonas clave de Ciudad de Guatemala.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikitlearn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)
![License](https://img.shields.io/badge/License-MIT-green)
![Estado](https://img.shields.io/badge/Estado-MVP%20Funcional-brightgreen)

---

##  Descripción del Proyecto

**GuateFlow** es un sistema inteligente que predice el nivel de congestión vial en zonas críticas de Ciudad de Guatemala (Zona 1, Zona 10, Zona 12, Calzada Roosevelt, entre otras), utilizando técnicas de clasificación con Machine Learning.

El modelo toma como entrada variables como la hora del día, el día de la semana, condiciones climáticas y la ocurrencia de eventos especiales, y devuelve una predicción del nivel de tráfico: **Bajo**, **Moderado** o **Alto**.

Desarrollado como proyecto de seminario para el curso de **Inteligencia Artificial — Primer Semestre 2026**, Universidad Rafael Landívar, Facultad de Ingeniería.

---

##  Problema que Resuelve

La congestión vial en Ciudad de Guatemala genera pérdidas estimadas de horas productivas diarias para miles de ciudadanos. La falta de herramientas predictivas accesibles impide que conductores, empresas de logística y autoridades municipales tomen decisiones anticipadas sobre rutas y horarios.

**GuateFlow** propone una solución práctica, de bajo costo y escalable para predecir el tráfico **antes de que ocurra**.

---

##  Estructura del Proyecto

```
GuateFlow/
│
├── data/
│   ├── raw/
│   │   └── trafico_guatemala.csv       # Dataset con 5,000 registros
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── model/
│   ├── guateflow_model.pkl             # Modelo entrenado (Gradient Boosting)
│   ├── label_encoder.pkl               # Codificador de etiquetas
│   ├── scaler.pkl                      # Escalador de variables numéricas
│   └── feature_names.pkl              # Nombres de columnas del modelo
│
├── src/
│   ├── generate_dataset.py             # Genera el dataset simulado
│   ├── preprocess.py                   # Limpieza y codificación de datos
│   ├── train.py                        # Entrena y compara los 4 modelos
│   ├── predict.py                      # Predicción individual por terminal
│   └── app.py                          # API REST con Flask
│
├── demo/
│   └── GuateFlow_Demo.html             # Demo web interactiva (sin instalación)
│
├── docs/
│   └── GuateFlow_Documento_Seminario.pdf
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

##  Técnicas de IA Utilizadas

| Técnica | Detalle |
|---|---|
| Clasificación supervisada | Predicción multiclase: Bajo / Moderado / Alto |
| Gradient Boosting | Modelo ganador — F1-Score: **77.9%** |
| Random Forest | F1-Score: 71.2% |
| Decision Tree | F1-Score: 73.3% |
| Logistic Regression | F1-Score: 50.6% |
| One-Hot Encoding | Codificación de variables categóricas |
| StandardScaler | Escalado de la variable numérica `hora` |

---

##  Variables del Modelo

| Variable | Tipo | Descripción |
|---|---|---|
| `zona` | Categórica | Zona geográfica (Zona 1, 4, 10, 12, Calzada Roosevelt, Periférico) |
| `hora` | Numérica | Hora del día en formato 24h (0–23) |
| `dia_semana` | Categórica | Día de la semana (Lunes a Domingo) |
| `clima` | Categórica | Condición climática (Soleado / Nublado / Lluvia) |
| `evento_especial` | Binaria | 1 si hay evento especial (partido, marcha, feria); 0 si no |
| `nivel_trafico` | **Target** | **Bajo / Moderado / Alto** — variable a predecir |

---

##  Instrucciones de Instalación y Uso

### Requisitos previos

- Python 3.10 o superior
- pip
- Git

---

### 1. Clonar el repositorio

```bash
git clone https://github.com/rochelleev001/GuateFlow.git
cd GuateFlow
```

---

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

### 3. Generar el dataset

```bash
python src/generate_dataset.py
```

Esto crea el archivo `data/raw/trafico_guatemala.csv` con 5,000 registros simulados.

---

### 4. Entrenar el modelo

```bash
python src/train.py
```

Este script:
- Preprocesa los datos automáticamente
- Entrena y compara 4 clasificadores
- Guarda el mejor modelo en `model/guateflow_model.pkl`
- Imprime la tabla comparativa de resultados en consola

---

### 5. Hacer una predicción por terminal

```bash
python src/predict.py --zona "Calzada Roosevelt" --hora 8 --dia "Lunes" --clima "Lluvia" --evento 0
```

**Parámetros disponibles:**

| Parámetro | Opciones |
|---|---|
| `--zona` | `"Zona 1"`, `"Zona 4"`, `"Zona 10"`, `"Zona 12"`, `"Calzada Roosevelt"`, `"Periférico"` |
| `--hora` | Número entero del `0` al `23` |
| `--dia` | `"Lunes"`, `"Martes"`, `"Miércoles"`, `"Jueves"`, `"Viernes"`, `"Sábado"`, `"Domingo"` |
| `--clima` | `"Soleado"`, `"Nublado"`, `"Lluvia"` |
| `--evento` | `0` (sin evento) o `1` (hay evento especial) |

---

### 6. Abrir la demo interactiva

No requiere servidor ni instalación adicional. Solo abre el archivo directamente en tu navegador:

```
demo/GuateFlow_Demo.html
```

Haz doble clic sobre el archivo desde el explorador de archivos. La demo incluye:

- Predictor interactivo con selector de parámetros
- Perfil de tráfico de 24 horas (gráfico)
- Mapa de niveles estimados por zona
- Importancia de variables del modelo

---

### 7. Levantar la API REST (opcional)

```bash
python src/app.py
```

La API queda disponible en `http://localhost:5000`

**Endpoint:**

```
POST /predict
Content-Type: application/json
```

**Ejemplo de petición:**

```json
{
  "zona": "Zona 10",
  "hora": 17,
  "dia_semana": "Viernes",
  "clima": "Lluvia",
  "evento_especial": 0
}
```

**Respuesta:**

```json
{
  "nivel_trafico": "Alto",
  "probabilidades": {
    "Alto": 72.3,
    "Moderado": 21.4,
    "Bajo": 6.3
  },
  "inputs": {
    "zona": "Zona 10",
    "hora": 17,
    "dia_semana": "Viernes",
    "clima": "Lluvia",
    "evento_especial": 0
  }
}
```

---

##  Resultados del Modelo

| Modelo | Accuracy | F1-Score (macro) |
|---|---|---|
| **Gradient Boosting**  | **77.8%** | **77.9%** |
| Decision Tree | 73.5% | 73.3% |
| Random Forest | 72.1% | 71.2% |
| Logistic Regression | 52.7% | 50.6% |

> El modelo **Gradient Boosting** fue seleccionado como modelo final por su mejor desempeño en todas las métricas. La clase `Moderado` es la más difícil de predecir al estar en la frontera entre las otras dos clases.

---

##  Modelo de Monetización

| Fase | Estrategia | Horizonte |
|---|---|---|
| Corto plazo | API gratuita para municipalidades y proyectos académicos | 0–6 meses |
| Mediano plazo | Suscripción SaaS para empresas de logística y transporte (Q300–Q800/mes) | 6–18 meses |
| Largo plazo | Integración con plataformas de navegación vía API licenciada | 18+ meses |

---

##  Información

**Universidad Rafael Landívar — Facultad de Ingeniería**
**Curso:** Inteligencia Artificial — Primer Semestre 2026

---

##  Licencia

Este proyecto está bajo la licencia MIT. Ver archivo `LICENSE` para más detalles.
