# Sistema Inteligente de Recomendación para Mantenimiento Predictivo

## Resumen

Herramienta que analiza parámetros operativos de máquinas para predecir fallos y sugerir acciones preventivas con justificación basada en modelos y reglas físicas.

https://mantenimiento-predictivo.streamlit.app/

## Características

- Predicción de fallo global y modos (TWF, HDF, PWF, OSF, RNF)
- Interpretabilidad con SHAP en UI
- Motor de recomendaciones accionables basado en reglas
- Interfaz Streamlit intuitiva con 4 pestañas
- **Sistema de feedback manual** para marcar si ocurrió o no un fallo
- **Combinación automática** de predicciones con feedback para reentrenamiento
- **Versionado automático de modelos** con historial y metadata completa

## Estructura

```
ProyectoFinalSI/
├── ai4i2020.csv                  # Dataset original
├── requirements.txt              # Dependencias
├── README.md                     # Esta guía
├── DOCUMENTACION_COMPLETA.md     # Documentación consolidada completa
├── app/
│   └── streamlit_app.py          # Interfaz de usuario
├── src/
│   ├── data/                     # Carga y preprocesamiento
│   └── ml/                       # Train, recommendation, SHAP, combine_feedback
├── models/                       # Modelos entrenados
│   └── versions/                 # Historial de versiones
├── logs/                         # Predicciones y feedback
└── tests/                        # Pruebas unitarias
```

## Instalación

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Inicio Rápido

### 1. Entrenar Modelo

```powershell
python -m src.ml.train
```

### 2. Ejecutar Interfaz de Usuario

```powershell
streamlit run app/streamlit_app.py
```

Abrir navegador en `http://localhost:8501`

## Uso del Sistema

### Hacer Predicciones

1. Ajustar parámetros en barra lateral (temperatura, RPM, torque, desgaste)
2. Clic en **"Calcular Predicción y Recomendaciones"**
3. Revisar índice de riesgo y recomendaciones priorizadas
4. **Pestaña SHAP**: Ver explicabilidad de la predicción
5. **Pestaña Histórico**: Ver evolución temporal

### Sistema de Feedback Manual

Después de cada predicción y operación real:

1. En sección **"📝 Feedback Post-Operación"**:
   - Clic en **"✅ No hubo fallo"** si operación exitosa
   - Clic en **"🚨 Si hubo fallo"** si ocurrió un fallo
2. Sistema actualiza automáticamente `logs/predicciones.csv` con valor de `Machine failure` (0 o 1)

### Mejora Continua del Modelo

**Flujo automático:**

```
1. Hacer predicción → se guarda en logs/predicciones.csv con Machine failure = None
2. Marcar feedback → actualiza misma fila con Machine failure = 0 o 1
```


## Próximas Mejoras

- Dashboard de análisis feedback vs predicciones
- Alertas automáticas por email para fallos críticos
- API REST para integración con otros sistemas
- Aplicación móvil para técnicos de campo

## Cita del Dataset

"Explainable Artificial Intelligence for Predictive Maintenance Applications"  
S. Matzka, Third International Conference on Artificial Intelligence for Industries (AI4I 2020)
