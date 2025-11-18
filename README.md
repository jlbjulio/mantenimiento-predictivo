# Sistema Inteligente de Recomendación para Mantenimiento Predictivo

## Resumen

Herramienta que analiza parámetros operativos de máquinas para predecir fallos y sugerir acciones preventivas con justificación basada en modelos y reglas físicas.

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
3. Reentrenamiento automático se ejecuta en segundo plano

### Mejora Continua del Modelo

**Flujo automático:**

```
1. Hacer predicción → se guarda en logs/predicciones.csv con Machine failure = None
2. Marcar feedback → actualiza misma fila con Machine failure = 0 o 1
3. Sistema automático combina y reentrena en segundo plano
```

El sistema automáticamente:

- Combina predicciones con feedback (solo filas con `Machine failure` definido)
- Genera dataset etiquetado en `data/additional/feedback_labeled_*.csv` (14 columnas)
- Entrena con datos originales + feedback real
- Crea nueva versión timestamped en `models/versions/`
- Guarda metadata completa (AUC, fecha, muestras)
- **NO requiere operaciones manuales** - todo es automático

## Documentación Completa

Ver **`DOCUMENTACION_COMPLETA.md`** para:

- Alcance y objetivos SMART
- Análisis exploratorio de datos (EDA)
- Arquitectura y desarrollo del sistema
- Manual de usuario detallado
- Guía de despliegue y mantenimiento
- Sistema de feedback y mejora continua

## Recomendaciones del Sistema

El motor de recomendaciones utiliza 5 reglas principales:

1. **Desgaste ≥ 200 min** → Reemplazo urgente herramienta
2. **Delta térmico < 9K + RPM < 1400** → Riesgo disipación térmica
3. **Potencia < 3500W o > 9000W** → Ajustar torque/velocidad
4. **Strain elevado según tipo** → Reducir carga operativa
5. **Probabilidad ≥ 0.6** → Inspección preventiva inmediata

## 🚀 Deploy en Streamlit Cloud

Para desplegar esta aplicación en la nube:

```powershell
# 1. Verificar que todo está listo
python verify_deploy.py

# 2. Inicializar Git (si no lo has hecho)
git init
git add .
git commit -m "Initial commit"

# 3. Subir a GitHub
git remote add origin https://github.com/TU_USUARIO/mantenimiento-predictivo.git
git push -u origin main

# 4. Ir a https://share.streamlit.io/
# 5. Crear nueva app apuntando a: app/streamlit_app.py
```

**Guía completa**: Ver `DEPLOY.md` para instrucciones detalladas

## Próximas Mejoras

- Dashboard de análisis feedback vs predicciones
- Alertas automáticas por email para fallos críticos
- API REST para integración con otros sistemas
- Aplicación móvil para técnicos de campo

## Cita del Dataset

"Explainable Artificial Intelligence for Predictive Maintenance Applications"  
S. Matzka, Third International Conference on Artificial Intelligence for Industries (AI4I 2020)
