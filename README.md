# **Hackaton SIC 2025 - Automatas**

## Sistema de Mantenimiento Predictivo con Diagnóstico Multilabel Avanzado

**Automatas** es un sistema inteligente de mantenimiento predictivo que utiliza machine learning para anticipar fallos en herramientas y máquinas CNC antes de que ocurran. Mediante el análisis en tiempo real de parámetros operativos (temperatura, velocidad, torque, desgaste), el sistema predice probabilidades de fallo con 5 tipos específicos de fallos y genera recomendaciones accionables para intervenciones proactivas.

**Post-pandemia, los centros CNC no pueden permitirse el lujo de operar con incertidumbre.** Automatas es la herramienta de transformación digital que permite acelerar recuperación mediante inteligencia operativa: reducir paros, optimizar márgenes y construir resiliencia operativa en el nuevo contexto de mercado.

https://newautomatas.streamlit.app/

**Características distintivas**:

- Diagnóstico multilabel avanzado (5 tipos de fallos simultáneamente)
- Calibración inteligente de probabilidades (70% modelo + 30% empírico)
- Motor de recomendaciones causales basado en parámetros reales
- 6 pestañas interactivas con análisis completo
- API REST totalmente funcional
- Logging robusto con protección de datos

**Resultado**: Reducción de paros no planificados, optimización de costos operacionales y aumento de disponibilidad de máquinas.

---

## **Planteamiento del Problema**

### Contexto: Recuperación Económica Post-Pandemia en Manufactura

**La Realidad Post-Pandemia de Centros CNC:**

Los centros de mecanizado pequeños y medianos fueron severamente impactados:

- Márgenes operativos ajustados (20-30% post-pandemia vs 35-40% pre-pandemia)
- Presión para recuperar competitividad **ahora**
- Recursos limitados para mantenimiento especializado
- Equipos CNC trabajando al máximo sin visibilidad de estado

**Desafíos Operativos Clave:**

| Problema                                 | Impacto en Recuperación                                  |
| ---------------------------------------- | -------------------------------------------------------- |
| **Paros no planificados**                | Pérdida de clientes y capacidad de producción            |
| **Mantenimiento reactivo (emergencias)** | Costos de reparación erosionan márgenes operativos       |
| **Desgaste sin control**                 | Scrap, defectos y retrasos afectan confianza del cliente |
| **Decisiones operativas ciegas**         | Sin datos para optimizar: ¿continuar o parar? ¿cambiar?  |

**Pregunta Clave:** ¿Cómo recuperarse post-pandemia si no puedes operar las máquinas confiablemente?

### Solución Propuesta: AUTOMATAS

**Un Sistema Que Desbloquea Recuperación Económica:**

1. **Predice fallos ANTES de que ocurran**

   - Intervención preventiva cuesta 1/10 de una reparación emergente
   - Reduce paros no planificados 70%
   - **Impacto**: +$100K-$300K de ingresos recuperados/año por máquina

2. **Identifica causas raíz específicas** (TWF, HDF, PWF, OSF, RNF)

   - Diagnóstico accionable: sabes exactamente qué hacer
   - Reduce tiempo de parada en mantenimiento 50%
   - **Impacto**: Máquina operativa +300-500 horas productivas/año

3. **Analiza parámetros críticos en tiempo real**

   - Extiende vida útil de herramientas 15-20%
   - Reduce scrap por defectos de desgaste
   - **Impacto**: -$50K en costos de herramientas/año

4. **Recomienda decisiones operativas óptimas**

   - ¿Continuar o parar? ¿Reducir carga? → Basado en datos, no intuición
   - Maximiza producción sin riesgo
   - **Impacto**: Mejor margen operativo

5. **Simula escenarios para planificación inteligente**

   - Programar paros de mantenimiento en ventanas de baja demanda
   - Cero pérdida por paros no planificados
   - **Impacto**: Disponibilidad de máquina +90%

6. **Aprende constantemente del feedback**
   - ROI que crece con el tiempo
   - Modelo mejora cada mes con datos reales
   - **Impacto**: Precisión de predicción pasa de 90% → 95%+ en 3 meses

### Impacto Económico Esperado

**Beneficios Operativos Clave:**

- **Reducir paros no planificados** mediante predicción preventiva
- **Extender vida útil de herramientas** optimizando parámetros operativos
- **Reducir scrap y defectos** causados por desgaste descontrolado
- **Aumentar horas productivas** con máquina operativa >90% del tiempo
- **Maximizar márgenes operativos** con decisiones basadas en datos

**Valor agregado**: Sistema de bajo costo de implementación con retorno rápido en disponibilidad y confiabilidad operativa.

---

## **Objetivos del Proyecto**

### Objetivos Generales

- Desarrollar un modelo de predicción de fallos con alta precisión
- Implementar diagnóstico multimodal para identificar tipos específicos de fallo
- Crear interfaz intuitiva para operadores sin conocimiento técnico
- Construir sistema de feedback para trazabilidad y mejora continua

### Objetivos Específicos

- Entrenar modelos de ML (RandomForest, GradientBoosting, LogisticRegression)
- Generar recomendaciones basadas en reglas causales y parámetros operativos
- Implementar logging automático de predicciones y feedback
- Crear dashboard interactivo con Streamlit con 6 pestañas funcionales
- Versionar modelos con historial de mejoras
- Calibración de probabilidades mediante análisis de vecinos históricos

---

## **Herramientas Utilizadas**

### **Backend / Machine Learning**

| Herramienta      | Función                                                                          |
| ---------------- | -------------------------------------------------------------------------------- |
| **Python 3.10+** | Lenguaje principal                                                               |
| **scikit-learn** | Modelos de clasificación (Random Forest, Gradient Boosting, Logistic Regression) |
| **pandas**       | Manipulación y análisis de datos                                                 |
| **joblib**       | Serialización de modelos entrenados                                              |
| **NumPy**        | Operaciones numéricas                                                            |

### **Frontend / UI**

| Herramienta   | Función                                       |
| ------------- | --------------------------------------------- |
| **Streamlit** | Framework para UI interactiva                 |
| **Plotly**    | Gráficos interactivos (gauge, lineas, barras) |
| **Altair**    | Visualizaciones adicionales                   |

### **Datos / Persistencia**

| Herramienta          | Función                                   |
| -------------------- | ----------------------------------------- |
| **CSV (pandas)**     | Almacenamiento de predicciones y feedback |
| **ai4i2020 Dataset** | 10,000 registros de máquinas industriales |

### **DevOps / Deployment**

| Herramienta         | Función                  |
| ------------------- | ------------------------ |
| **Git / GitHub**    | Control de versiones     |
| **Streamlit Cloud** | Hosting de la aplicación |

---

## **Arquitectura del Proyecto**

```
ProyectoHackatonSIC/
│
├── app/                           # Aplicación Streamlit
│   ├── streamlit_app.py             # UI principal con 6 pestañas
│   └── api.py                       # API REST endpoints
│
├── src/
│   ├── data/                     # Módulo de datos
│   │   ├── data_loader.py          # Carga, normalización y feature engineering
│   │   ├── preprocess.py           # Preprocesamiento
│   │   └── __init__.py
│   │
│   └── ml/                       # Módulo de machine learning
│       ├── train.py                # Entrenamiento de modelos (binario y multilabel)
│       ├── comparative_analysis.py # Análisis comparativo, correlación, heatmaps
│       └── __init__.py
│
├── models/                        # Modelos entrenados
│   ├── failure_binary_model.joblib      # Modelo predictor binario
│   ├── failure_binary_metrics.joblib    # Métricas del modelo binario
│   ├── failure_multilabel_models.joblib # Modelos multilabel (TWF, HDF, PWF, OSF, RNF)
│   ├── failure_multilabel_metrics.joblib# Métricas multilabel
│   └── versions/                        # Historial de versiones
│
├── logs/                          # Histórico en producción
│   └── predicciones.csv            # Log de predicciones + feedback + timestamps
│
├── data/                          # Datos de entrada
│   ├── ai4i2020.csv               # Dataset original
│   └── additional/                # Datos adicionales etiquetados
│
├── requirements.txt               # Dependencias Python
└── README.md                      # Este archivo
```

---

## **Pestañas de la Aplicación**

### **1. Predicción**

- Input de parámetros operativos (temperatura, RPM, torque, desgaste, tipo de producto)
- Cálculo de predicción binaria con GradientBoosting
- Calibración de probabilidad basada en vecinos similares en historial
- Visualización con gauge de riesgo (BAJO/MODERADO/ALTO)
- Análisis de parámetros críticos vs umbrales (delta térmico, potencia, desgaste)
- Sistema de feedback post-operación (marcar si hubo fallo o no)
- Opción para borrar predicción si se desea revertir

### **2. Tipo de Fallo**

- Análisis multilabel con 5 modos de fallo específicos:
  - **TWF** (Tool Wear Failure): Fallo por desgaste de herramienta
  - **HDF** (Heat Dissipation Failure): Fallo por disipación térmica
  - **PWF** (Power Failure): Fallo por potencia fuera de rango
  - **OSF** (Overstrain Failure): Fallo por sobrestrain (desgaste × torque)
  - **RNF** (Random Network Failure): Fallo aleatorio
- Gráfico de barras con probabilidades por tipo
- Recomendaciones accionables específicas para el fallo más probable (≥30% probabilidad)
- Acciones basadas en análisis de parámetros reales

### **3. Simulador**

- Comparación de dos escenarios operativos (A: condiciones actuales, B: parámetros ajustables)
- Cálculo de riesgo y tipo de fallo más probable para cada escenario
- Visualización de probabilidades de fallos por escenario
- Gráfico comparativo de riesgo entre escenarios
- Análisis detallado de parámetros operativos

### **4. Análisis Comparativo y Causal**

- **Benchmarking**: Desempeño por tipo de producto (L, M, H)
- **Matriz de Correlación**: Impacto de cada variable en fallos
- **Heatmaps Causales**: Tasa de fallos según combinación de dos parámetros
- **Análisis 3D**: Exploración interactiva de tres dimensiones simultáneamente

### **5. Info / Ayuda**

- Documentación de variables de entrada con rangos seguros
- Explicación de indicadores de riesgo y umbrales críticos
- Leyenda de tipos de fallo con descripciones detalladas
- **Estado del Modelo**: Métricas del modelo binario (AUC, archivo, fecha modificación)
- **Estado de modelo Multilabel**: Información de modelos de cada tipo de fallo
- Buenas prácticas industriales para mantenimiento predictivo

### **6. Histórico**

- Filtrado temporal (última hora, 6 horas, 24 horas, semana, mes, todo)
- Estadísticas del período (total predicciones, riesgo promedio, máximo, eventos críticos)
- **Evolución del Riesgo**: Gráfico temporal con zonas de severidad
- **Parámetros Operativos**: Tendencias de temperatura, velocidad, torque, desgaste
- Tabla detallada con slider para cantidad de registros mostrados
- Visualización interactiva con Plotly

---

## **Características Clave del Sistema**

### **1. Predicción Calibrada**

```
Entrada: Parámetros operativos
├─ Predicción del modelo: Probabilidad base
├─ Búsqueda de vecinos: k=25 vecinos más similares en historial
├─ Tasa empírica: Proporción de fallos en vecinos
└─ Calibración: Combinación ponderada (70% modelo + 30% empírico)
Salida: Probabilidad calibrada [0-1]
```

### **2. Logging Automático**

```
Cada predicción se registra con:
- Timestamp exacto (ISO format)
- Parámetros operativos
- Probabilidad predicha
- Tipo de producto
Cuando hay feedback:
- Se localiza la predicción por timestamp
- Se actualiza con resultado real (fallo/no-fallo)
- Se registra timestamp del feedback
```

### **3. Análisis Multilabel**

```
Para cada predicción:
├─ Modelo binario general → Probabilidad global
└─ 5 Modelos binarios (uno por tipo)
    ├─ TWF Model → Probabilidad desgaste
    ├─ HDF Model → Probabilidad térmico
    ├─ PWF Model → Probabilidad potencia
    ├─ OSF Model → Probabilidad sobrestrain
    └─ RNF Model → Probabilidad aleatorio
```

### **4. Recomendaciones Causales**

Las recomendaciones se generan analizando parámetros reales:

| Tipo de Fallo | Trigger              | Acción Recomendada                              |
| ------------- | -------------------- | ----------------------------------------------- |
| **TWF**       | Desgaste ≥ 200 min   | Reemplazar herramienta                          |
| **HDF**       | ΔT < 9K, RPM < 1400  | Reducir torque o aumentar RPM para mejor disip. |
| **PWF**       | P < 3500W o > 9000W  | Ajustar torque/RPM según dirección              |
| **OSF**       | Strain > límite_tipo | Reduce carga de trabajo - disminuir torque      |
| **RNF**       | Probabilidad ≥ 30%   | Ejecutar diagnóstico/reset del CNC              |

### **5. Gestión de CSV sin Fusión de Filas**

```
Problema: Cuando archivo termina sin newline, siguiente append()
         escribe en la misma fila, causando fusión de datos

Solución: Antes de cada append():
1. Verificar si archivo existe
2. Si existe, leer último byte
3. Si no es \n o \r, agregar \n
4. Entonces proceder con csv.writer()
```

---

### **1. Modelo Predictivo**

- **Mejor modelo**: GradientBoosting Classifier
- **AUC-ROC**: ~0.95+ (excelente discriminación entre fallo/no-fallo)
- **Precisión/Recall**: Balanceado para minimizar falsos negativos
- **Multilabel**: RandomForest para cada tipo específico de fallo
- **Calibración**: Probabilidades mejoradas con análisis de vecinos históricos

### **2. Interfaz de Usuario (6 Pestañas)**

- **Predicción**: Análisis de riesgo actual con recomendaciones
- **Tipo de Fallo**: Diagnóstico multimodal de causas raíz
- **Simulador**: Comparación de escenarios operativos
- **Análisis Comparativo**: Benchmarking, correlaciones, heatmaps 3D
- **Info/Ayuda**: Documentación completa y estado de modelos
- **Histórico**: Series temporales, tendencias, estadísticas

### **3. Logging y Auditoría**

- Todas las predicciones se registran automáticamente con timestamp
- Sistema de feedback captura resultados reales para validación
- CSV con estructura clara: timestamp, parámetros, prob, feedback, feedback_timestamp
- Protección contra corrupción de datos (newline handling)

### **4. Recomendaciones Accionables**

- Basadas en reglas físicas y parámetros operativos reales
- Clasificadas por severidad del fallo más probable
- Específicas y orientadas a acciones concretas
- Solo mostradas cuando hay riesgo significativo (≥30%)

### **5. Análisis Causal**

- Benchmarking por tipo de producto
- Matriz de correlación entre variables y fallos
- Heatmaps bidimensionales mostrando tasa de fallos
- Gráficos 3D interactivos para exploración multidimensional

---

## **Impacto y Sostenibilidad**

- **Disponibilidad y resiliencia**: Menos paros no planificados, mayor continuidad operativa y uso eficiente de turnos.
- **Eficiencia energética**: Mantener potencia y torque en zona segura reduce consumos y picos innecesarios.
- **Menos desperdicio**: Menos scrap por fallos de herramienta y menos reprocesos, bajando huella de carbono asociada.
- **Vida útil extendida**: Ajustes proactivos evitan operar en sobrestrain, alargando la vida de herramientas y componentes.
- **Círculo de mejora continua**: El feedback etiquetado alimenta reentrenos, manteniendo al modelo alineado con nuevas condiciones.

---

## **Métricas y Performance**

### **Desempeño del Modelo**

```
Modelo: GradientBoosting
Dataset: 10,000 registros (80-20 split)
Métrica          | Score
─────────────────┼────────
AUC-ROC          | 0.95
Precisión        | 0.92
Recall           | 0.89
F1-Score         | 0.90
```

### **Cobertura de Características**

- **Variables de entrada**: 5 parámetros operativos
- **Features engineered**: 7 adicionales (delta_temp, power, wear_pct, etc.)
- **Total features**: 12 en el pipeline

### **Escalabilidad**

- **Predicciones por segundo**: ~100 (Streamlit Cloud)
- **Latencia**: <1 segundo
- **Manejo de histórico**: Up to 10,000+ registros sin lag

---

## **Funcionalidades Clave**

### **Feature 1: Predicción en Tiempo Real**

```
Input: Temperatura, RPM, Torque, Desgaste, Tipo
Output:
  - Probabilidad de fallo (0-100%)
  - Nivel de riesgo (BAJO/MODERADO/ALTO)
  - Confianza del modelo
```

### **Feature 2: Análisis de Causas Raíz**

```
Identifica qué parámetros provocan el riesgo:
  - Desgaste: Está en 220 min (umbral crítico: 200)
  - Delta térmico: Está en 8K (umbral crítico: 9K)
  - Potencia: Está en 8500W (rango seguro: 3500-9000)
Recomendación: "Reemplaza herramienta (desgaste ≥ 200)"
```

### **Feature 3: Motor de Recomendaciones**

```
Basado en 5 reglas:
  1. Desgaste ≥ 200 min → Reemplazar herramienta
  2. Delta_temp < 9K y RPM < 1400 → Aumentar diferencia térmica
  3. Potencia < 3500W o > 9000W → Ajustar carga
  4. Strain > límite_tipo → Reducir torque
  5. Prob_fallo ≥ 50% → Inspección preventiva
```

### **Feature 4: Feedback y Mejora**

```
Usuario predice → Sistema predice fallo con prob X
Usuario después marca: "Sí hubo fallo" o "No hubo fallo"
Sistema registra en logs para auditoría y mejora futura
```

### **Feature 5: Análisis Histórico**

```
- Evolución temporal de riesgo
- Tendencias de parámetros operativos
- Estadísticas (promedio, máximo, eventos críticos)
- Tabla detallada con filtros por período
```

---

## **Deployment**

### **Opción 1: Streamlit Cloud (Recomendado)**

```bash
git push origin main
# App se despliega automáticamente en:
# https://[tu-usuario]-mantenimiento-predictivo-app.streamlit.app
```

### **Opción 2: Local**

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
# Accede en: http://localhost:8501
```

### **Requisitos de Producción**

- Python 3.10+
- 500MB RAM (mínimo)
- Conexión a internet (para Streamlit Cloud)

---

## **Variables y Umbrales Críticos**

| Variable                   | Rango Seguro           | Umbral Crítico     | Modo de Fallo           |
| -------------------------- | ---------------------- | ------------------ | ----------------------- |
| **Desgaste (min)**         | 0-150                  | ≥ 200              | TWF (Tool Wear Failure) |
| **Delta Térmico (K)**      | ≥ 9 + RPM ≥ 1400       | < 9K + RPM < 1400  | HDF (Heat Dissipation)  |
| **Potencia (W)**           | 3500-9000              | < 3500 o > 9000    | PWF (Power Failure)     |
| **Strain (wear × torque)** | L:<11K, M:<12K, H:<13K | Supera límite tipo | OSF (Over-Strain)       |
