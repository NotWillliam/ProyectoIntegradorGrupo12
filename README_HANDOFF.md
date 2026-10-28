# Predicción de deserción de clientes — Retail farmacéutico

Tesis de Maestría en Inteligencia Artificial. Predicción de deserción
(churn) **no contractual** de clientes de un retail farmacéutico,
a partir de datos transaccionales reales, para que
Marketing priorice una lista limitada de clientes a contactar.

**Modelo de producción: LightGBM (variante A2, categóricas nativas).**
PR-AUC de test = 0,6875, precision@10% = 0,84, lift@10% = 2,09 — supera con
claridad a la regresión logística (0,6701) y a los baselines de negocio
(0,63-0,64). Ver `docs/arquitectura.md` para el diseño completo y
`docs/optimizacion.md` para cómo se llegó a esa configuración.

## Estructura

```
PROYECTO_ESTRUCTURADO/
├── docs/            documentación (planificación, EDA, arquitectura, optimización, ética, manual de usuario)
├── data/            raw/ (dataset crudo) y processed/ (splits ya preprocesados)
├── notebooks/       01_exploracion -> 02_preprocesamiento -> 03_modelado -> 04_optimizacion -> 05_evaluacion
├── src/             módulos reutilizables (data_processing, model, train, evaluate, utils, preprocesamiento)
├── models/          best_model.pkl (LightGBM A2) + versiones anteriores
├── app/             (todavía no construida — ver nota abajo)
├── tests/           pruebas de src/ con pytest
└── results/         figures/, metrics/, reports/
```

Esta carpeta es una **reorganización** (26/09/2026) del trabajo ya hecho en
la carpeta de trabajo original (un nivel arriba: `01_desercion_dataset.ipynb`,
`02_desercion_eda_prep.ipynb`, `03_desercion_modelado.ipynb`,
`preprocesamiento.py`, `datos/`, `artefactos/`, `salidas/`). Los notebooks de
aquí contienen las mismas celdas ya ejecutadas (no resultados fabricados de
nuevo), reorganizadas en la estructura estándar de un proyecto de ML.
`notebooks/05_evaluacion.ipynb` es la excepción: es un notebook **nuevo**,
que recalcula y muestra la evaluación final quedándose **solo con LightGBM**
(la comparación completa contra la logística y los baselines queda en
`03_modelado.ipynb` / `04_optimizacion.ipynb`).

**La recolección de datos** (conexión a SQL Server, el SQL simplificado de 9
CTEs) **no** se reconstruyó dentro de esta carpeta: no forma parte de
exploración/preprocesamiento/modelado/optimización/evaluación, que fue lo
pedido. Sigue viviendo en `01_desercion_dataset.ipynb`, un nivel arriba.

## `app/` — interfaz local

La interfaz Streamlit carga `models/best_model.pkl` mediante
`app/prediction_service.py`. Permite ejecutar una demostración, cargar CSV,
configurar el tamaño de campaña, descargar el ranking y consultar un cliente.
Se inicia desde la raíz con `streamlit run app/app.py`.

## Qué falta antes de dar por cerrado el modelado

- **T3.5 — Calibración** de probabilidades (Platt scaling / isotonic
  regression + Brier score).
- **T3.8 — Análisis de supervivencia**, el diferenciador académico del
  trabajo: incorporar a los clientes censurados (15,82%) como información
  parcial en vez de excluirlos.

Ver `docs/planificacion.md` y `PLAN_DE_ACCION.md` (Proyecto de Claude "MODELO
DE PREDICCIÓN DE DESERCIÓN") para el detalle completo, actualizado, de cada
tarea.

## Cómo empezar

```bash
pip install -r requirements.txt
python -m pytest tests/          # 21 pruebas, deben pasar
python src/train.py              # reentrena LightGBM A2 (~10 s)
python src/evaluate.py           # evalúa sobre test, reproduce las cifras de arriba
```

O abrir los notebooks en `notebooks/` en orden. Más detalle en
`docs/manual_usuario.md`.
