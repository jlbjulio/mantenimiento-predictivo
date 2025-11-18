# Mapeo de Columnas: Predicciones → Dataset de Entrenamiento

## 📊 Columnas del Dataset Original (ai4i2020.csv)

```
UDI, Product ID, Type, Air temperature [K], Process temperature [K],
Rotational speed [rpm], Torque [Nm], Tool wear [min], Machine failure,
TWF, HDF, PWF, OSF, RNF
```

## 🔄 Transformación desde predicciones.csv

### Columnas que vienen directamente del CSV de predicciones:

| Columna Destino             | Origen en predicciones.csv | Notas                             |
| --------------------------- | -------------------------- | --------------------------------- |
| **Type**                    | `type`                     | Copiado directamente (L/M/H)      |
| **Air temperature [K]**     | `air_temp_k`               | Copiado directamente              |
| **Process temperature [K]** | `process_temp_k`           | Copiado directamente              |
| **Rotational speed [rpm]**  | `rot_speed_rpm`            | Copiado directamente              |
| **Torque [Nm]**             | `torque_nm`                | Copiado directamente              |
| **Tool wear [min]**         | `tool_wear_min`            | Copiado directamente              |
| **Machine failure**         | `Machine failure`          | Etiqueta real del usuario (0 o 1) |

### Columnas generadas automáticamente (valores por defecto):

| Columna Destino | Valor Asignado             | Razón                                                         |
| --------------- | -------------------------- | ------------------------------------------------------------- |
| **UDI**         | Sequential ID (1, 2, 3...) | ID único secuencial para nuevos registros                     |
| **Product ID**  | `{Type}99999`              | Placeholder: H99999, M99999, L99999                           |
| **TWF**         | `0`                        | Tool Wear Failure - No se puede inferir sin lógica de negocio |
| **HDF**         | `0`                        | Heat Dissipation Failure - No se puede inferir                |
| **PWF**         | `0`                        | Power Failure - No se puede inferir                           |
| **OSF**         | `0`                        | Overstrain Failure - No se puede inferir                      |
| **RNF**         | `0`                        | Random Failure - No se puede inferir                          |

### Columnas internas (NO guardadas en CSV final):

Estas columnas se usan internamente para tracking y deduplicación, pero **NO** aparecen en `feedback_labeled_*.csv`:

| Columna Interna         | Origen               | Uso                                     |
| ----------------------- | -------------------- | --------------------------------------- |
| `_prediction_prob`      | `prob`               | Probabilidad predicha por el modelo     |
| `_prediction_timestamp` | `timestamp`          | Timestamp de la predicción (para dedup) |
| `_feedback_timestamp`   | `feedback_timestamp` | Timestamp del feedback del usuario      |

Estas columnas metadata sí aparecen en `feedback_full_*.csv` para análisis posteriores.

## ⚠️ Limitaciones Importantes

### 1. Tipos de Fallo Específicos (TWF, HDF, PWF, OSF, RNF)

**Problema:** El sistema actual solo captura `Machine failure` (fallo binario general), NO los tipos específicos de fallo.

**Valores por defecto:** Todos se llenan con `0` (no fallo).

**Implicación:** El modelo NO aprenderá a distinguir entre tipos de fallo específicos usando datos de feedback. Solo aprenderá "fallo general" vs "sin fallo".

**Solución futura:** Para capturar tipos de fallo específicos, necesitarías:

- Modificar la UI para preguntar QUÉ tipo de fallo ocurrió
- Implementar lógica de inferencia basada en parámetros operativos
- O calcular automáticamente según umbrales (ejemplo: si `tool_wear_min >= 200` → `TWF = 1`)

### 2. UDI y Product ID

**UDI:** Se genera secuencialmente (1, 2, 3...) en cada archivo nuevo. No hay continuidad entre archivos.

**Product ID:** Se genera como placeholder `{Type}99999`. No corresponde a un producto real del dataset original.

**Implicación:** Estas columnas son decorativas en los datos de feedback. El modelo las ignora en el preprocesador (se excluyen explícitamente en `train.py`).

## 📁 Archivos Generados

### `feedback_labeled_TIMESTAMP.csv`

Contiene **solo** las columnas necesarias para entrenamiento (formato idéntico a ai4i2020.csv):

```
UDI,Product ID,Type,Air temperature [K],Process temperature [K],
Rotational speed [rpm],Torque [Nm],Tool wear [min],Machine failure,
TWF,HDF,PWF,OSF,RNF
```

### `feedback_full_TIMESTAMP.csv`

Contiene todas las columnas anteriores MÁS las columnas metadata:

```
...(todas las anteriores)..., _prediction_prob, _prediction_timestamp, _feedback_timestamp
```

## ✅ Verificación

Para confirmar que el formato es correcto:

```python
import pandas as pd

# Leer dataset original
original = pd.read_csv('ai4i2020.csv')
print("Columnas originales:", list(original.columns))

# Leer último feedback
feedback = pd.read_csv('data/additional/feedback_labeled_LATEST.csv')
print("Columnas feedback:", list(feedback.columns))

# Verificar match exacto
assert list(original.columns) == list(feedback.columns), "¡Columnas no coinciden!"
print("✅ Las columnas coinciden perfectamente")
```

## 🔄 Flujo Completo

1. **Usuario hace predicción** → Se guarda en `logs/predicciones.csv`
2. **Usuario da feedback** → Se actualiza `Machine failure` en el mismo CSV
3. **Combinación automática** → `combine_predictions_with_feedback()` extrae filas con feedback
4. **Normalización** → Se mapean columnas y se añaden valores por defecto
5. **Guardado** → Se genera `feedback_labeled_*.csv` con formato idéntico a dataset original
6. **Reentrenamiento** → `train.py` carga `ai4i2020.csv` + todos los `data/additional/*.csv`

## 🎯 Recomendación

Si necesitas que el modelo aprenda tipos de fallo específicos (TWF, HDF, etc.), considera:

- Implementar lógica automática de inferencia basada en umbrales físicos
- Añadir campos en la UI para que el usuario indique el tipo de fallo
- Usar el modelo multilabel existente y añadir esas etiquetas al feedback
