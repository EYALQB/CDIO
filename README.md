# 📘 Proyecto CDIO - Grupo 6

Este repositorio contiene el desarrollo completo del proyecto **Modelado y predicción de la línea de costa (2017–2025)**, realizado por el **Grupo 6 de CDIO**.

El objetivo del trabajo es analizar la evolución temporal de la línea de costa en el tramo Castelldefels–Gavà–El Prat mediante series temporales, identificar tendencias y patrones estacionales, y generar modelos predictivos que permitan estimar el comportamiento costero durante el primer semestre de 2025.

---

## 🗂️ Estructura del proyecto
├── data/
│ ├── shoreline_distances_castefa_gava_prat_2017_2024.csv.zip # Datos originales (2017–2024)
│ ├── shoreline_distances_castefa_gava_prat_h1_2025_ref2017.csv # Observaciones reales 2025
│
├── outputs/
│ ├── predictions_2025_H1.csv # Predicciones generadas por los modelos
│ ├── validation_summary.txt # Métricas de validación (RMSE, MAE, PICP)
│ └── figures/ # Gráficas de resultados y validación
│
├── src/
│ ├── data_preparation.py # Limpieza, filtrado y agregación de datos
│ ├── exploratory_analysis.py # Análisis exploratorio y visualización
│ ├── model_fitting.py # Ajuste de modelos base y mejorados (step y sigmoid)
│ ├── model_evaluation.py # Evaluación comparativa de modelos
│ ├── forecasting.py # Generación del pronóstico enero–junio 2025
│ └── validation_discussion.py # Validación final y discusión de resultados
│
├── memoria_proyecto_CDIO_Grupo6.pdf # Informe final (memoria del proyecto)
└── README.md # Este archivo


---

## ⚙️ Ejecución

Cada paso del proyecto puede ejecutarse de forma independiente desde el directorio raíz:

```bash
python src/data_preparation.py
python src/exploratory_analysis.py
python src/model_fitting.py
python src/model_evaluation.py
python src/forecasting.py
python src/validation_discussion.py

Los resultados y gráficas se almacenan automáticamente en la carpeta outputs/.


