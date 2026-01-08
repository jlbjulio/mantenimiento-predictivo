import streamlit as st
import joblib
import pandas as pd
import os
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import altair as alt
import csv
import shutil
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

st.set_page_config(
    page_title="Automatas - Mantenimiento Predictivo para Centros de Mecanizado",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/jlbjulio/Automatas',
        'Report a bug': 'https://github.com/jlbjulio/Automatas/issues',
        'About': '# Automatas - Mantenimiento Predictivo para Centros de Mecanizado\nVersion 1.0 - Nov 2025'
    }
)

MODEL_PATH = 'models/failure_binary_model.joblib'
METRICS_PATH = 'models/failure_binary_metrics.joblib'
PRED_LOG = 'logs/predicciones.csv'
MULTILABEL_MODEL_PATH = 'models/failure_multilabel_models.joblib'
MULTILABEL_METRICS_PATH = 'models/failure_multilabel_metrics.joblib'

from src.data.data_loader import engineer_features, normalize_columns, load_dataset, find_dataset
from src.ml.comparative_analysis import (
    analyze_by_product_type, calculate_correlation_matrix, calculate_failure_rates_by_bins,
    create_benchmark_comparison, create_correlation_heatmap,
    create_failure_heatmap, create_3d_scatter
)


def _to_original_schema(df_row: pd.DataFrame, base_file: str):
    """Convierte una fila con esquema normalizado al esquema original de ai4i2020.csv.
    
    Esta función mapea columnas normalizadas (ej: air_temp_k) a las columnas originales del dataset
    (ej: 'Air temperature [K]'). Es útil para mantener compatibilidad con el dataset base.
    
    Args:
        df_row (pd.DataFrame): Fila de datos con columnas normalizadas o Series
        base_file (str): Ruta al archivo CSV base para obtener esquema original
    
    Returns:
        dict: Diccionario con columnas del esquema original mapeadas desde los valores de entrada
    """
    try:
        import pandas as pd
        head = pd.read_csv(base_file, nrows=0)
        orig_cols = list(head.columns)
        mapping = {
            'UDI': 'uid',
            'Product ID': 'product_id',
            'Type': 'type',
            'Air temperature [K]': 'air_temp_k',
            'Process temperature [K]': 'process_temp_k',
            'Rotational speed [rpm]': 'rot_speed_rpm',
            'Torque [Nm]': 'torque_nm',
            'Tool wear [min]': 'tool_wear_min',
            'Machine failure': 'machine_failure',
            'TWF': 'twf',
            'HDF': 'hdf',
            'PWF': 'pwf',
            'OSF': 'osf',
            'RNF': 'rnf'
        }
        out = {}
        if isinstance(df_row, pd.DataFrame):
            if df_row.shape[0] >= 1:
                row_series = df_row.iloc[0]
            else:
                row_series = pd.Series([])
        else:
            row_series = df_row
        for c in orig_cols:
            norm = mapping.get(c, c)
            val = row_series.get(norm, None) if norm in row_series.index else row_series.get(c, None)
            if pd.notna(val):
                out[c] = val
            else:
                out[c] = ''
        return out
    except Exception:
        try:
            return df_row.to_dict(orient='records')[0]
        except Exception:
            return {}


def _align_input_with_pipeline(df_row: pd.DataFrame, pipeline):
    """Alinea las columnas de entrada con las esperadas por el preprocesador del pipeline.
    
    Verifica que el DataFrame de entrada tenga todas las columnas requeridas por el pipeline.
    Añade columnas faltantes con valores NA o cadenas vacías (según el tipo) y reordena
    las columnas para coincidir exactamente con el orden esperado por el modelo.
    
    Args:
        df_row (pd.DataFrame): DataFrame con características de entrada
        pipeline: Pipeline de scikit-learn que contiene el preprocesador
    
    Returns:
        pd.DataFrame: DataFrame con columnas alineadas y ordenadas según el pipeline
    """
    import pandas as pd
    try:
        pre = pipeline.named_steps.get('pre', None)
        expected_cols = []
        if pre is not None and hasattr(pre, 'transformers_'):
            try:
                for _name, _tr, _cols in pre.transformers_:
                    if _cols is None:
                        continue
                    
                    if isinstance(_cols, slice):
                        expected_cols.extend(list(df_row.columns[_cols]))
                        continue
                    
                    try:
                        iter_cols = list(_cols)
                        if len(iter_cols) > 0 and isinstance(iter_cols[0], int):
                            expected_cols.extend([df_row.columns[i] for i in iter_cols if isinstance(i, int) and i < len(df_row.columns)])
                        else:
                            expected_cols.extend([c for c in iter_cols if isinstance(c, str)])
                        continue
                    except Exception:
                        pass
                    if isinstance(_cols, str):
                        expected_cols.append(_cols)
            except Exception:
                expected_cols = []
        if pre is not None and hasattr(pre, 'feature_names_in_'):
            expected_cols = list(pre.feature_names_in_)
        elif not expected_cols:
            if pre is not None and hasattr(pre, 'feature_names_in_'):
                expected_cols = list(pre.feature_names_in_)
            elif hasattr(pipeline, 'feature_names_in_'):
                expected_cols = list(pipeline.feature_names_in_)
            else:
                expected_cols = list(df_row.columns)
    except Exception:
        expected_cols = list(df_row.columns)

    import numpy as np
    CANONICAL_NUMERIC = [
        'air_temp_k','process_temp_k','rot_speed_rpm','torque_nm','tool_wear_min',
        'delta_temp_k','omega_rad_s','power_w','wear_pct','prob','prediction_prob'
    ]
    for c in expected_cols:
        if c not in df_row.columns:
            if c in CANONICAL_NUMERIC or 'prob' in c.lower():
                df_row[c] = np.nan
            else:
                df_row[c] = ''
    extras = [c for c in df_row.columns if c not in expected_cols]
    ordered_cols = expected_cols + extras
    return df_row[ordered_cols]

def prepare_feature_row(user_data: dict) -> pd.DataFrame:
    """Convierte parámetros de entrada de usuario en un DataFrame con características ingenieridas.
    
    Toma un diccionario con los 5 parámetros operativos básicos (temperatura, RPM, torque, desgaste, tipo)
    y aplica ingeniería de características para generar nuevas columnas derivadas (delta térmico,
    potencia, razones de desgaste, etc.) que se usan en el modelo de predicción.
    
    Args:
        user_data (dict): Diccionario con claves: air_temp_k, process_temp_k, rot_speed_rpm, 
                         torque_nm, tool_wear_min, type
    
    Returns:
        pd.DataFrame: Una fila de datos con características originales e ingenieridas
    """
    import pandas as pd
    import numpy as np
    
    # Crear fila con datos de entrada
    df_row = pd.DataFrame([{
        'air_temp_k': float(user_data.get('air_temp_k', 300.0)),
        'process_temp_k': float(user_data.get('process_temp_k', 310.0)),
        'rot_speed_rpm': float(user_data.get('rot_speed_rpm', 1500.0)),
        'torque_nm': float(user_data.get('torque_nm', 40.0)),
        'tool_wear_min': float(user_data.get('tool_wear_min', 50.0)),
        'type': user_data.get('type', 'L'),
    }])
    
    try:
        df_row = engineer_features(df_row)
    except Exception:
        pass
    
    return df_row

def predict_instance(model, user_data: dict):
    """Realiza predicción de probabilidad de fallo para un conjunto de parámetros operativos.
    
    Prepara las características del usuario, alinea con el pipeline, y genera predicción binaria
    con probabilidad asociada. Maneja alineación automática de columnas y érores en preprocesamiento.
    
    Args:
        model: Modelo entrenado (pipeline con preprocesador y clasificador)
        user_data (dict): Parámetros operativos del usuario
    
    Returns:
        tuple: (predicción_binaria, probabilidad_fallo) donde predicción ∈ {0, 1} y 
               probabilidad ∈ [0, 1]
    """
    df_row = prepare_feature_row(user_data)
    try:
        df_row_aligned = _align_input_with_pipeline(df_row, model)
    except Exception:
        df_row_aligned = df_row
    prob = model.predict_proba(df_row_aligned)[0][1]
    pred = model.predict(df_row_aligned)[0]
    return pred, prob


def predict_modes_multilabel(multi_models, user_data: dict):
    """Calcula probabilidades de cada tipo de fallo (TWF, HDF, PWF, OSF, RNF) usando modelos multilabel.
    
    Para cada modo de fallo se entrena un clasificador binario independiente. Esta función
    invoca todos los clasificadores y retorna las probabilidades individuales para permitir
    diagnóstico multimodal de las causas raíz del fallo.
    
    Args:
        multi_models: Diccionario de modelos cargados {nombre_modo: modelo} o None
        user_data (dict): Parámetros operativos del usuario
    
    Returns:
        dict: Probabilidades por modo de fallo {MODO: probabilidad}, ej: {'TWF': 0.45, 'HDF': 0.12, ...}
              Si multi_models es None, retorna dict vacío {}
    """
    if multi_models is None:
        return {}
    df_row = prepare_feature_row(user_data)
    probs = {}
    for label, mdl in multi_models.items():
        try:
            df_al = _align_input_with_pipeline(df_row.copy(), mdl)
            probs[label.upper()] = float(mdl.predict_proba(df_al)[0][1])
        except Exception:
            probs[label.upper()] = None
    return probs

def calibrated_probability(user_data: dict, model_prob: float, k_neighbors: int = 25, alpha: float = 0.7) -> float:
    """Calibra la probabilidad del modelo combinando predicción con tasa empírica de vecinos similares.
    
    Usa un enfoque híbrido: busca los k vecinos más cercanos en el historial de predicciones
    con feedback etiquetado, calcula su tasa de fallo empírica, y mezcla esa tasa con la
    probabilidad del modelo usando un parámetro alpha de confianza.
    
    Nótese: no hay reentrenamiento. Solo usa feedback válido existente en logs/predicciones.csv.
    
    Args:
        user_data (dict): Parámetros operativos actuales
        model_prob (float): Probabilidad predicha por el modelo [0, 1]
        k_neighbors (int): Cantidad de vecinos similares a considerar en historial (default 25)
        alpha (float): Peso de confianza en el modelo [0, 1] (default 0.7 → 70% modelo, 30% empírico)
    
    Returns:
        float: Probabilidad calibrada en rango [0, 1]
    """
    try:
        if not os.path.exists(PRED_LOG):
            return float(model_prob)
        hist = pd.read_csv(PRED_LOG, engine='python', on_bad_lines='warn')
        # Filtrar registros con feedback etiquetado (Machine failure y feedback_timestamp válidos)
        valid = hist.copy()
        valid = valid[valid['Machine failure'].isin([0.0, 1.0, 0, 1])]
        valid = valid[pd.to_datetime(valid.get('feedback_timestamp', ''), errors='coerce').notna()]
        if valid.empty:
            return float(model_prob)
        # Construir matriz de características
        feats = ['air_temp_k','process_temp_k','rot_speed_rpm','torque_nm','tool_wear_min']
        # Normalizar características numéricas a escalas comparables
        def _norm(col, x):
            try:
                c = valid[col].astype(float)
                mn, mx = c.min(), c.max()
                if mx > mn:
                    return (x - mn) / (mx - mn)
                return 0.5
            except Exception:
                return 0.5
        q = {f: float(user_data.get(f)) for f in feats}
        qn = {f: _norm(f, q[f]) for f in feats}
        dists = []
        for idx, row in valid.iterrows():
            try:
                rn = {f: _norm(f, float(row.get(f))) for f in feats}
                dist = 0.0
                for f in feats:
                    dist += (qn[f] - rn[f])**2
                type_pen = 0.0 if str(row.get('type')) == str(user_data.get('type')) else 0.15
                dists.append((dist + type_pen, float(row.get('Machine failure', 0.0))))
            except Exception:
                continue
        if not dists:
            return float(model_prob)
        dists.sort(key=lambda x: x[0])
        neighbors = dists[:max(1, k_neighbors)]
        empirical = sum([mf for _, mf in neighbors]) / len(neighbors)
        calib = float(alpha) * float(model_prob) + (1.0 - float(alpha)) * float(empirical)
        calib = max(0.0, min(1.0, calib))
        return calib
    except Exception:
        return float(model_prob)

@st.cache_resource
def load_models():
    """Carga los modelos de clasificación binaria y multilabel desde archivos joblib en disco.
    
    Intenta cargar dos modelos:
    1. Modelo binario: Predice presencia/ausencia de fallo (sí/no)
    2. Modelos multilabel: Clasificadores individuales para cada tipo de fallo (TWF, HDF, PWF, OSF, RNF)
    
    Si alguno falla, emite una advertencia pero continua con los que se carguen correctamente.
    El caching garantiza que los modelos se cargan solo una vez por sesión de Streamlit.
    
    Returns:
        tuple: (modelo_binario, diccionario_modelos_multilabel) donde cada elemento es None si falla su carga
    """
    model = None
    multi = None
    
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
    except Exception:
        st.warning('No se pudo cargar el modelo binario')
    
    try:
        if os.path.exists(MULTILABEL_MODEL_PATH):
            multi = joblib.load(MULTILABEL_MODEL_PATH)
    except Exception:
        st.warning('No se pudo cargar los modelos multilabel')
    
    return model, multi

@st.cache_resource
def best_model_name():
    """Extrae el nombre del mejor modelo entrenado a partir del archivo de métricas.
    
    Lee el archivo de métricas guardado durante el entrenamiento y busca el modelo
    con el AUC más alto. Si no hay métricas disponibles, retorna 'GradientBoosting' por defecto.
    
    Returns:
        str: Nombre del mejor modelo o 'GradientBoosting' por defecto
    """
    try:
        if os.path.exists(METRICS_PATH):
            metrics = joblib.load(METRICS_PATH)
            aucs = {}
            if isinstance(metrics, dict) and 'aucs' in metrics:
                aucs = metrics['aucs']
            elif isinstance(metrics, dict):
                for model_name, model_metrics in metrics.items():
                    if isinstance(model_metrics, dict) and 'auc' in model_metrics:
                        aucs[model_name] = model_metrics['auc']
            if aucs:
                best = max(aucs.items(), key=lambda x: x[1])
                return best[0]
    except Exception:
        pass
    return 'GradientBoosting'

def load_metrics_status():
    status = {}
    status['best_model'] = 'GradientBoosting'
    
    if os.path.exists(MODEL_PATH):
        status['model_file'] = MODEL_PATH
        status['last_modified'] = str(pd.to_datetime(os.path.getmtime(MODEL_PATH), unit='s'))
    else:
        status['message'] = 'Entrene una vez el modelo base con el dataset original.'
    return status

def load_multilabel_status():
    """Retorna el estado y métricas del modelo multilabel en formato JSON."""
    status = {}
    status['model_used'] = 'RandomForest'
    
    if os.path.exists(MULTILABEL_MODEL_PATH):
        status['model_file'] = MULTILABEL_MODEL_PATH
        status['last_modified'] = str(pd.to_datetime(os.path.getmtime(MULTILABEL_MODEL_PATH), unit='s'))
    else:
        status['message'] = 'No hay modelo multilabel entrenado. Entrena failure_multilabel_models.joblib.'
    
    if os.path.exists(MULTILABEL_METRICS_PATH):
        try:
            metrics = joblib.load(MULTILABEL_METRICS_PATH)
            status['metrics_available'] = True
            if isinstance(metrics, dict):
                status['labels'] = sorted(list(metrics.keys()))
        except Exception:
            status['metrics_available'] = False
    
    return status

def log_prediction(data: dict, prob: float, pred: int, machine_failure: int = None, feedback_timestamp: pd.Timestamp = None):
    """Registra una predicción en el archivo CSV de log con manejo de actualizaciones por feedback.
    
    Implementa un sistema de dos pasos para registrar predicciones y actualizar con feedback:
    1. Primer llamado (sin feedback): Crea nueva fila en el CSV con timestamp de predicción
    2. Segundo llamado (con feedback): Localiza la fila anterior por timestamp exacto y actualiza
       los campos Machine failure y feedback_timestamp con el feedback del usuario.
    
    Manejador especial: Respeta la bandera suppress_logging para evitar registros durante
    operaciones de limpieza (ej: cuando el usuario borra la última predicción).
    
    Args:
        data (dict): Parámetros operativos. Debe incluir 'prediction_timestamp' para deduplicación
        prob (float): Probabilidad de fallo predicha [0, 1]
        pred (int): Predicción binaria (0 = no fallo, 1 = fallo)
        machine_failure (int, optional): Feedback del usuario (0/1) o None si aún no hay feedback
        feedback_timestamp (pd.Timestamp, optional): Marca temporal del feedback o None
    
    Returns:
        None
    """
    try:
        if st.session_state.get('suppress_logging', False):
            return
    except Exception:
        pass
    header = ['timestamp','air_temp_k','process_temp_k','rot_speed_rpm','torque_nm','tool_wear_min','type','pred','prob','Machine failure','feedback_timestamp']
    write_header = not os.path.exists(PRED_LOG)
    # Asegurar que el encabezado CSV incluya las columnas nuevas; si faltan, agregarlas sin reescribir los datos existentes
    full_header = ['timestamp','air_temp_k','process_temp_k','rot_speed_rpm','torque_nm','tool_wear_min','type','pred','prob','Machine failure','feedback_timestamp']
    if os.path.exists(PRED_LOG):
        try:
            with open(PRED_LOG, 'r', newline='') as f:
                first_line = f.readline().strip()
            existing_cols = [c.strip() for c in first_line.split(',') if c.strip()]
            missing = [c for c in full_header if c not in existing_cols]
            if missing:
                # Reescribir CSV con nuevo encabezado y anexar filas antiguas con valores vacíos para columnas faltantes
                df_old = pd.read_csv(PRED_LOG, engine='python', on_bad_lines='warn')
                for c in missing:
                    df_old[c] = ''
                df_old.to_csv(PRED_LOG, index=False, columns=full_header)
                write_header = False
        except Exception:
            # Mejor esfuerzo; ignorar errores
            pass
    def _write(ts):
        # Normalizar timestamps a ISO segundos para evitar problemas de parseo
        try:
            ts_norm = pd.to_datetime(ts, errors='coerce')
            if pd.notna(ts_norm):
                ts = ts_norm.strftime('%Y-%m-%dT%H:%M:%S')
        except Exception:
            pass
        fb_ts = ''
        if feedback_timestamp is not None:
            try:
                fb_ts_dt = pd.to_datetime(feedback_timestamp, errors='coerce')
                if pd.notna(fb_ts_dt):
                    fb_ts = fb_ts_dt.strftime('%Y-%m-%dT%H:%M:%S')
                else:
                    fb_ts = str(feedback_timestamp)
            except Exception:
                fb_ts = str(feedback_timestamp)
        # Asegurar que el archivo termina con newline antes de escribir
        if os.path.exists(PRED_LOG):
            with open(PRED_LOG, 'rb') as f:
                f.seek(-1, 2)  # Ir al último byte
                if f.read(1) not in (b'\n', b'\r'):
                    with open(PRED_LOG, 'a') as f_fix:
                        f_fix.write('\n')
        with open(PRED_LOG, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow([ts, data['air_temp_k'], data['process_temp_k'], data['rot_speed_rpm'], data['torque_nm'], data['tool_wear_min'], data['type'], pred, f"{prob:.4f}", machine_failure if machine_failure is not None else '', fb_ts])
    # Aceptar timestamp proporcionado si existe
    # Actualizar (deduplicación) SOLO cuando hay feedback: buscar por timestamp exacto y actualizar campos
    try:
        if (machine_failure is not None or feedback_timestamp is not None) and os.path.exists(PRED_LOG):
            existing = pd.read_csv(PRED_LOG)
            if 'timestamp' in existing.columns and not existing.empty:
                try:
                    provided_raw = data.get('prediction_timestamp')
                    provided_dt = pd.to_datetime(provided_raw, errors='coerce')
                    if pd.isna(provided_dt):
                        raise ValueError('Invalid prediction timestamp')
                    provided_str = provided_dt.strftime('%Y-%m-%dT%H:%M:%S')
                    mask = (existing['timestamp'].astype(str) == provided_str)
                    if mask.any():
                        idxs = existing[mask].index
                        if machine_failure is not None:
                            existing.loc[idxs, 'Machine failure'] = float(machine_failure)
                        if feedback_timestamp is not None:
                            try:
                                fb_ts_dt = pd.to_datetime(feedback_timestamp, errors='coerce')
                                fb_val = fb_ts_dt.strftime('%Y-%m-%dT%H:%M:%S') if pd.notna(fb_ts_dt) else str(feedback_timestamp)
                            except Exception:
                                fb_val = str(feedback_timestamp)
                            existing.loc[idxs, 'feedback_timestamp'] = fb_val
                        existing.to_csv(PRED_LOG, index=False)
                        return
                except Exception:
                    pass
    except Exception:
        pass
    if isinstance(data.get('prediction_timestamp'), (str, pd.Timestamp)):
        _write(str(data.get('prediction_timestamp')))
    else:
        _write(pd.Timestamp.utcnow())
    
 

def remove_last_prediction_row() -> bool:
    """Elimina la última fila registrada en el archivo CSV de predicciones.
    
    Operación atómica: Lee el CSV, remueve la última fila, y reescribe el archivo.
    Usada cuando el usuario desea cancelar/deshacer la última predicción registrada.
    
    Returns:
        bool: True si se eliminó exitosamente, False si hubo error o el log está vacío
    """
    try:
        if not os.path.exists(PRED_LOG):
            return False
        df = pd.read_csv(PRED_LOG, engine='python', on_bad_lines='warn')
        if df.empty or len(df) == 0:
            return False
        df = df.iloc[:-1].copy()
        df.to_csv(PRED_LOG, index=False)
        return True
    except Exception as e:
        return False


def ensure_pred_log_has_history(model, count: int = 500):
    """Puebla el archivo de log de predicciones con un historial inicial del dataset base.
    
    Util para bootstrapping: Si el archivo de log no existe o tiene pocas filas (< count),
    carga muestras del dataset base ai4i2020.csv, genera predicciones para ellas, y las
    agrega al log. De esta manera, el sistema de calibración de probabilidades (que busca
    vecinos similares en historial) tiene suficientes ejemplos para funcionar bien.
    
    Args:
        model: Modelo entrenado para generar predicciones
        count (int): Número mínimo de filas esperadas en el log (default 500)
    
    Returns:
        None. Modifica el archivo logs/predicciones.csv como efecto secundario.
    """
    try:
        import numpy as np
        if os.path.exists(PRED_LOG):
            df_exist = pd.read_csv(PRED_LOG)
            if len(df_exist) >= count:
                return
        base = load_dataset()
        if base is None or base.empty:
            return
        sample = base.sample(n=min(count, len(base)), replace=True).reset_index(drop=True)
        rows = []
        for _, row in sample.iterrows():
            data = {
                'air_temp_k': row.get('air_temp_k'),
                'process_temp_k': row.get('process_temp_k'),
                'rot_speed_rpm': row.get('rot_speed_rpm'),
                'torque_nm': row.get('torque_nm'),
                'tool_wear_min': row.get('tool_wear_min'),
                'type': row.get('type')
            }
            df_row = prepare_feature_row(data)
            try:
                prob = model.predict_proba(df_row)[0][1]
                pred = int(model.predict(df_row)[0])
            except Exception:
                prob = 0.0
                pred = 0
            rows.append([pd.Timestamp.utcnow(), data['air_temp_k'], data['process_temp_k'], data['rot_speed_rpm'], data['torque_nm'], data['tool_wear_min'], data['type'], pred, f"{prob:.4f}"])
        write_header = not os.path.exists(PRED_LOG)
        header = ['timestamp','air_temp_k','process_temp_k','rot_speed_rpm','torque_nm','tool_wear_min','type','pred','prob']
        # Asegurar que el archivo termina con newline antes de escribir
        if os.path.exists(PRED_LOG) and not write_header:
            with open(PRED_LOG, 'rb') as f:
                f.seek(-1, 2)  # Ir al último byte
                if f.read(1) not in (b'\n', b'\r'):
                    with open(PRED_LOG, 'a') as f_fix:
                        f_fix.write('\n')
        with open(PRED_LOG, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            for r in rows:
                w.writerow(r)
    except Exception:
        # Ignorar errores de siembra; el historial puede estar incompleto
        return

def log_feedback(timestamp: str, actual_failure: int):
    """Registra feedback de un usuario sobre una predicción previa (compatibilidad hacia atrás).
    
    Busca una fila existente por timestamp de predicción y actualiza sus campos Machine failure
    y feedback_timestamp. Si no existe una entrada previa, crea una entrada mínima con los datos
    de feedback disponibles.
    
    Esta función proporciona compatibilidad hacia atrás con código más antiguo que pudía
    registrar feedback sin tener una predicción previa guardada.
    
    Args:
        timestamp (str): Timestamp de la predicción para la cual se registra feedback
        actual_failure (int): Feedback del usuario (0 = no fallo, 1 = fallo)
    
    Returns:
        None
    """
    try:
        # Intento de localizar una fila de predicción existente y actualizarla
        if os.path.exists(PRED_LOG):
            df = pd.read_csv(PRED_LOG)
            ts = str(timestamp)
            if 'timestamp' in df.columns and ts in df['timestamp'].astype(str).tolist():
                        # Asegurar que columnas de texto sean dtype object para evitar warnings de deprecación
                for _c in ['feedback_timestamp']:
                    if _c in df.columns:
                        try:
                            df[_c] = df[_c].astype('object')
                        except Exception:
                            pass
                        df.loc[df['timestamp'] == ts, 'Machine failure'] = actual_failure
                df.loc[df['timestamp'] == ts, 'feedback_timestamp'] = str(pd.Timestamp.utcnow())
                df.to_csv(PRED_LOG, index=False)
                return
        # Si no se encuentra una predicción previa, crear una entrada mínima
        dummy = {
            'air_temp_k': None, 'process_temp_k': None, 'rot_speed_rpm': None,
            'torque_nm': None, 'tool_wear_min': None, 'type': None,
            'pred': None, 'prob': None
        }
        data = {**dummy, 'prediction_timestamp': pd.Timestamp.utcnow()}
        log_prediction(data, prob=0.0, pred=0, machine_failure=actual_failure, feedback_timestamp=pd.Timestamp.utcnow())
    except Exception:
        pass

def bulk_log(df: pd.DataFrame):
    """Registra en lote múltiples predicciones al CSV de log sin feedback individual.
    
    Función de utilidad para añadir histórico masivamente. Itera sobre el DataFrame y
    escribe cada fila al log de predicciones. Últil para importar datos históricos o
    simular multiple predicciones de una sola vez.
    
    Requiere: DataFrame debe tener columna 'failure_prob' para las probabilidades.
    
    Args:
        df (pd.DataFrame): DataFrame con columnas: air_temp_k, process_temp_k, rot_speed_rpm,
                          torque_nm, tool_wear_min, type, failure_prob
    
    Returns:
        None. Modifica logs/predicciones.csv como efecto secundario.
    """
    if 'failure_prob' not in df.columns:
        return
    write_header = not os.path.exists(PRED_LOG)
    header = ['timestamp','air_temp_k','process_temp_k','rot_speed_rpm','torque_nm','tool_wear_min','type','pred','prob','Machine failure','feedback_timestamp']
    # Asegurar que el archivo termina con newline antes de escribir
    if os.path.exists(PRED_LOG) and not write_header:
        with open(PRED_LOG, 'rb') as f:
            f.seek(-1, 2)  # Ir al último byte
            if f.read(1) not in (b'\n', b'\r'):
                with open(PRED_LOG, 'a') as f_fix:
                    f_fix.write('\n')
    with open(PRED_LOG, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        for _, row in df.iterrows():
            w.writerow([pd.Timestamp.utcnow(), row.get('air_temp_k'), row.get('process_temp_k'), row.get('rot_speed_rpm'), row.get('torque_nm'), row.get('tool_wear_min'), row.get('type'), 'NA', f"{row.get('failure_prob'):.4f}", '', ''])


def main():
    st.title('Automatas - Mantenimiento Predictivo para Centros de Mecanizado')
    st.caption('Plataforma de inteligencia predictiva para CNC. Diagnóstico multimodal de fallos, simulación de escenarios y feedback inteligente para optimizar mantenimiento preventivo en planta.')
    model, multi = load_models()
    # Inicializar estado de sesión para almacenar última predicción
    try:
        if 'last_prediction_data' not in st.session_state:
            st.session_state.last_prediction_data = None
    except Exception:
        # st.session_state puede no estar disponible fuera del runtime de Streamlit
        pass
    # Reentrenamiento deshabilitado: no hay bandera ni umbral

    base_df = load_dataset()
    def dyn_range(col, pad=0.05):
        if col not in base_df.columns:
            return (0.0, 1.0)
        mn, mx = base_df[col].min(), base_df[col].max()
        span = mx - mn
        return float(mn - span*pad), float(mx + span*pad)

    # Mantener dyn_range como referencia; no limitar las entradas a los límites del dataset
    a_min, a_max = dyn_range('air_temp_k')
    p_min, p_max = dyn_range('process_temp_k')
    r_min, r_max = dyn_range('rot_speed_rpm')
    tq_min, tq_max = dyn_range('torque_nm')
    w_min, w_max = dyn_range('tool_wear_min')

    st.sidebar.header('Entrada de Parámetros Operativos')
    # Verificar si hay una predicción pendiente para desactivar los inputs
    has_pending_prediction = st.session_state.get('last_prediction_data') is not None
    if has_pending_prediction:
        st.sidebar.warning('⚠️ Confirma el feedback de la predicción actual para modificar parámetros.')
    # Usar number_input en lugar de sliders para permitir valores fuera de los límites del dataset
    air_temp = st.sidebar.number_input('Temperatura ambiente [K]', value=300.0, step=0.1, format="%.1f", disabled=has_pending_prediction)
    process_temp = st.sidebar.number_input('Temperatura de proceso [K]', value=310.0, step=0.1, format="%.1f", disabled=has_pending_prediction)
    rot_speed = st.sidebar.number_input('Velocidad de rotación [rpm]', value=1500.0, step=1.0, format="%.0f", disabled=has_pending_prediction)
    torque = st.sidebar.number_input('Torque [Nm]', value=40.0, step=0.1, format="%.1f", disabled=has_pending_prediction)
    wear = st.sidebar.number_input('Desgaste herramienta [min]', value=50.0, step=1.0, format="%.0f", disabled=has_pending_prediction)
    prod_type = st.sidebar.selectbox('Tipo de producto', ['L','M','H'], disabled=has_pending_prediction)
    st.sidebar.markdown("""
**Tipos de Producto:**
- **L (Low)**: Calidad baja, mayor tolerancia a strain (≤11,000)
- **M (Medium)**: Calidad media, tolerancia moderada (≤12,000)
- **H (High)**: Calidad alta, menor tolerancia strain (≤13,000)
""")

    tab_pred, tab_modes, tab_sim, tab_analysis, tab_info, tab_hist = st.tabs([
        "Predicción", "Tipo de fallo", "Simulador", "Análisis Comparativo y Causal", "Info / Ayuda", "Histórico"
    ])

    with tab_pred:
        st.subheader('Escenario Actual de Operación')
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)
        col1.metric('Temp. Ambiente (K)', f"{air_temp:.1f}")
        col2.metric('Temp. Proceso (K)', f"{process_temp:.1f}", f"Δ {(process_temp-air_temp):.1f} K")
        col3.metric('Rotación (rpm)', f"{rot_speed:.0f}")
        col4.metric('Torque (Nm)', f"{torque:.1f}")
        col5.metric('Desgaste (min)', f"{wear:.0f}", f"{wear/240*100:.1f}%")
        col6.metric('Tipo', prod_type)
        st.markdown('<hr/>', unsafe_allow_html=True)
        
        # Desactivar botón de predicción si hay una predicción pendiente sin feedback
        has_pending_prediction = st.session_state.get('last_prediction_data') is not None
        if has_pending_prediction:
            st.info('⚠️ Debes confirmar el feedback de la predicción actual antes de calcular una nueva o modificar parámetros.')
        
        if st.button('Calcular Predicción y Recomendaciones', disabled=has_pending_prediction):
            st.session_state.feedback_given = False
            st.session_state.suppress_logging = False
            data = {
                'air_temp_k': air_temp,
                'process_temp_k': process_temp,
                'rot_speed_rpm': rot_speed,
                'torque_nm': torque,
                'tool_wear_min': wear,
                'type': prod_type
            }
            pred, prob = predict_instance(model, data)
            prob_cal = calibrated_probability(data, prob, k_neighbors=25, alpha=0.7)
            now_ts = pd.Timestamp.utcnow()
            st.session_state.last_prediction_data = {**data, 'pred': int(pred), 'prob': float(prob), 'prob_cal': float(prob_cal), 'prediction_timestamp': now_ts}
            log_prediction(
                data={**data, 'prediction_timestamp': now_ts},
                prob=float(prob),
                pred=int(pred),
                machine_failure=None,
                feedback_timestamp=None
            )
            st.rerun()

        # Mostrar resultado y acciones de feedback si hay predicción activa en la sesión
        if st.session_state.get('last_prediction_data') is not None:
            data_view = st.session_state.get('last_prediction_data')
            prob_v = float(data_view.get('prob', 0.0))
            prob_cal_v = float(data_view.get('prob_cal', prob_v))
            pred_v = int(data_view.get('pred', 0))
            st.subheader('Resultado de Predicción')
            risk_label = 'ALTO' if prob_cal_v>=0.6 else ('MODERADO' if prob_cal_v>=0.3 else 'BAJO')
            st.markdown(f"### Riesgo de fallo: **{risk_label}** - **{prob_cal_v:.2f}**")
            st.markdown(f"Modelo seleccionado: **{best_model_name()}**")

            # Alertas visuales rápidas
            alert_badges = []
            if prob_cal_v >= 0.6:
                alert_badges.append(('ALTO RIESGO', '#c62828'))
            elif prob_cal_v >= 0.3:
                alert_badges.append(('RIESGO MODERADO', '#ff8f00'))
            if alert_badges:
                chips = " ".join([f"<span style='padding:6px 10px;border-radius:12px;background:{c};color:white;font-weight:700;margin-right:6px;'>" \
                                   f"{txt}</span>" for txt, c in alert_badges])
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.success('Condiciones dentro de rango seguro.')

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_cal_v*100,
                title={'text': 'Índice de Riesgo (%)'},
                gauge={
                    'axis': {'range': [0,100]},
                    'steps': [
                        {'range':[0,30],'color':'#2e7d32'},
                        {'range':[30,60],'color':'#ffb300'},
                        {'range':[60,100],'color':'#c62828'}
                    ],
                    'threshold': {'line': {'color': '#c62828', 'width': 4}, 'thickness': 0.75, 'value': prob_cal_v*100}
                }
            ))
            st.plotly_chart(gauge, use_container_width=True)

            delta_temp = float(data_view['process_temp_k']) - float(data_view['air_temp_k'])
            omega = float(data_view['rot_speed_rpm']) * 2 * 3.141592653589793 / 60
            power = float(data_view['torque_nm']) * omega
            wear_val = float(data_view['tool_wear_min'])

            fig_metrics = make_subplots(
                rows=1, cols=3,
                subplot_titles=["Delta Térmico (K)", "Potencia (W)", "Desgaste Herramienta (%)"],
                horizontal_spacing=0.12,
                specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
            )
            delta_color = '#d32f2f' if delta_temp < 9 else '#1976d2'
            fig_metrics.add_trace(go.Bar(
                name='ΔT', x=['Delta Térmico'], y=[delta_temp], marker_color=delta_color,
                text=[f"{delta_temp:.1f} K"], textposition='outside', textfont=dict(size=16, color='black'), width=0.5
            ), row=1, col=1)
            fig_metrics.add_hline(y=9, line_color='red', line_width=3, line_dash='dash',
                                  annotation_text='Umbral Crítico: 9K', annotation_position='top right', annotation_font_size=12,
                                  row=1, col=1)
            power_color = '#d32f2f' if (power < 3500 or power > 9000) else '#6a1b9a'
            fig_metrics.add_trace(go.Bar(
                name='Potencia', x=['Potencia'], y=[power], marker_color=power_color,
                text=[f"{power:.0f} W"], textposition='outside', textfont=dict(size=16, color='black'), width=0.5
            ), row=1, col=2)
            fig_metrics.add_hrect(y0=3500, y1=9000, line_width=2, fillcolor='rgba(76,175,80,0.2)', line_color='green',
                                  annotation_text='Zona Segura', annotation_position='top left', annotation_font_size=12, row=1, col=2)
            wear_pct = wear_val/240.0*100
            wear_color = '#d32f2f' if wear_val >= 200 else ('#ff8f00' if wear_val >= 150 else '#00838f')
            fig_metrics.add_trace(go.Bar(
                name='Desgaste', x=['Desgaste'], y=[wear_pct], marker_color=wear_color,
                text=[f"{wear_pct:.1f}% ({wear_val:.0f} min)"], textposition='outside', textfont=dict(size=16, color='black'), width=0.5
            ), row=1, col=3)
            fig_metrics.add_hline(y=200/240*100, line_color='red', line_width=3, line_dash='dash',
                                  annotation_text='Umbral Crítico: 200 min', annotation_position='top right', annotation_font_size=12,
                                  row=1, col=3)
            fig_metrics.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
            fig_metrics.update_xaxes(showticklabels=False)
            fig_metrics.update_layout(showlegend=False, height=500, font=dict(size=14),
                                      title_text="Análisis de Parámetros Críticos vs Umbrales de Seguridad", title_font_size=18)
            st.plotly_chart(fig_metrics, use_container_width=True)
            st.caption("""**Interpretación de colores:** 
            - **Rojo**: Parámetro en zona crítica - requiere acción inmediata
            - **Naranja**: Parámetro en zona de precaución - monitorear de cerca
            - **Azul/Morado**: Parámetro en rango operativo normal
            """)

            st.markdown('<hr/>', unsafe_allow_html=True)
            st.subheader('Feedback Post-Operación')
            feedback_given = st.session_state.get('feedback_given', False)
            if feedback_given:
                st.info('✓ Ya se registró feedback para esta predicción. Haz una nueva predicción para continuar.')
            else:
                st.caption('Marque si el fallo ocurrió o no después de esta operación para mejorar el modelo.')

            col_fb1, col_fb2, col_fb3 = st.columns([2, 2, 3])
            with col_fb1:
                if st.button('✅ No ocurrió fallo', key='no_fail', disabled=feedback_given):
                    pred_stored = st.session_state.get('last_prediction_data')
                    if pred_stored:
                        log_prediction(
                            data=pred_stored,
                            prob=float(pred_stored.get('prob', 0.0)),
                            pred=int(pred_stored.get('pred', 0)),
                            machine_failure=0,
                            feedback_timestamp=pd.Timestamp.utcnow()
                        )
                        st.session_state.feedback_given = True
                        # Limpiar la UI de predicción activa y evitar registrar nuevamente la misma predicción
                        st.session_state.last_prediction_data = None
                        st.session_state.suppress_logging = True
                        # No combinar feedback ni reentrenar; solo registrar y continuar
                        st.rerun()
                    else:
                        st.error('⚠️ No hay predicción activa. Primero realiza una predicción.')
            with col_fb2:
                if st.button('❌ Sí ocurrió fallo', key='fail', disabled=feedback_given):
                    pred_stored = st.session_state.get('last_prediction_data')
                    if pred_stored:
                        log_prediction(
                            data=pred_stored,
                            prob=float(pred_stored.get('prob', 0.0)),
                            pred=int(pred_stored.get('pred', 0)),
                            machine_failure=1,
                            feedback_timestamp=pd.Timestamp.utcnow()
                        )
                        st.session_state.feedback_given = True
                        # Limpiar la UI de predicción activa y evitar registrar nuevamente la misma predicción
                        st.session_state.last_prediction_data = None
                        st.session_state.suppress_logging = True
                        # No combinar feedback ni reentrenar; solo registrar y continuar
                        st.rerun()
                    else:
                        st.error('⚠️ No hay predicción activa. Primero realiza una predicción.')

            if st.button('Borrar predicción actual', key='clear_pred'):
                # Limpiar sesión primero
                st.session_state.last_prediction_data = None
                st.session_state.suppress_logging = True
                
                # Intentar borrar la última fila
                removed = remove_last_prediction_row()
                
                if removed:
                    st.success('🗑️ Última fila eliminada de predicciones correctamente.')
                    st.rerun()
                else:
                    st.warning('No hay predicciones registradas a borrar.')



    with tab_modes:
        st.subheader('Análisis de Tipos de Fallo Detectados')
        st.caption('Identifica cuáles tipos de fallo tienen mayor probabilidad en las condiciones actuales y muestra acciones prioritarias para reducirlos.')

        if multi is None:
            st.warning('Los modelos multilabel no están cargados. Entrena o agrega failure_multilabel_models.joblib para habilitar este panel.')
        else:
            base_data = st.session_state.get('last_prediction_data') or {
                'air_temp_k': air_temp,
                'process_temp_k': process_temp,
                'rot_speed_rpm': rot_speed,
                'torque_nm': torque,
                'tool_wear_min': wear,
                'type': prod_type
            }
            mode_probs = predict_modes_multilabel(multi, base_data)
            # Filtrar valores None y verificar si hay valores válidos
            valid_mode_probs = {k: v for k, v in mode_probs.items() if v is not None}
            
            if not valid_mode_probs:
                st.info('No se pudieron calcular probabilidades por modo.')
            else:
                labels = list(valid_mode_probs.keys())
                values = [valid_mode_probs[lbl] for lbl in labels]
                colors = ['#c62828' if v >= 0.6 else ('#ff8f00' if v >= 0.3 else '#2e7d32') for v in values]
                fig_modes = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, text=[f"{v:.2f}" for v in values], textposition='auto'))
                fig_modes.update_yaxes(range=[0, 1], tickformat='.0%')
                fig_modes.update_layout(title='Probabilidad por tipo de fallo', height=420)
                st.plotly_chart(fig_modes, use_container_width=True)

                # Mostrar recomendaciones para TODOS los fallos con probabilidad >= 0.3
                max_prob = max(valid_mode_probs.values()) if valid_mode_probs else 0
                
                if max_prob >= 0.3:
                    st.markdown('#### Acciones específicas según tus parámetros')
                    
                    # Calcular parámetros para análisis
                    delta_temp = base_data['process_temp_k'] - base_data['air_temp_k']
                    omega_rad_s = base_data['rot_speed_rpm'] * 2 * 3.141592653589793 / 60
                    power_w = base_data['torque_nm'] * omega_rad_s
                    wear_pct = base_data['tool_wear_min'] / 240.0
                    
                    # Filtrar y ordenar fallos con probabilidad >= 0.3
                    significant_failures = [(lbl, prob) for lbl, prob in valid_mode_probs.items() if prob >= 0.3]
                    significant_failures.sort(key=lambda x: x[1], reverse=True)  # Ordenar por probabilidad descendente
                    
                    # Mostrar recomendaciones para TODOS los fallos significativos
                    for lbl, prob_val in significant_failures:
                        risk_level = '🔴 ALTO RIESGO' if prob_val >= 0.6 else '🟠 RIESGO MODERADO'
                        
                        if lbl == 'TWF':
                            st.warning(f"**{lbl} - {prob_val:.1%} {risk_level}**")
                            st.markdown(f"Tu desgaste es **{base_data['tool_wear_min']:.0f} min** (límite crítico: 200 min)")
                            st.markdown(f"**ACCIÓN:** Reemplaza la herramienta")
                            
                        elif lbl == 'HDF':
                            st.warning(f"**{lbl} - {prob_val:.1%} {risk_level}**")
                            st.markdown(f"Tu delta térmico es **{delta_temp:.1f} K** (mínimo recomendado: 9 K)")
                            if delta_temp < 9:
                                st.markdown(f"**ACCIÓN:** Reduce torque o aumenta RPM para mejorar disipación térmica")
                                
                        elif lbl == 'PWF':
                            st.warning(f"**{lbl} - {prob_val:.1%} {risk_level}**")
                            st.markdown(f"Tu potencia es **{power_w:.0f} W** (rango seguro: 3,500-9,000 W)")
                            if power_w < 3500:
                                st.markdown(f"**ACCIÓN:** Aumenta torque o mantén RPM constante")
                            elif power_w > 9000:
                                st.markdown(f"**ACCIÓN:** Reduce torque o aumenta RPM")
                                
                        elif lbl == 'OSF':
                            st.warning(f"**{lbl} - {prob_val:.1%} {risk_level}**")
                            st.markdown(f"Desgaste: **{base_data['tool_wear_min']:.0f} min** ({wear_pct*100:.0f}%) | Torque: **{base_data['torque_nm']:.1f} Nm**")
                            st.markdown(f"**ACCIÓN:** Reduce carga de trabajo - disminuye torque")
                            
                        elif lbl == 'RNF':
                            st.warning(f"**{lbl} - {prob_val:.1%} {risk_level}**")
                            st.markdown("**ACCIÓN:** Ejecuta diagnóstico/reset de la máquina CNC. Verifica conexiones de sensores.")
                        
                        st.divider()  # Separador visual entre fallos


    with tab_sim:
        st.subheader('Simulador de escenarios')
        st.caption('Compara escenarios para decidir ajustes operativos.')
        
        colA, colB = st.columns(2)
        
        with colA:
            st.markdown('### Escenario A')
            st.info('Condiciones actuales de operación')
            air_temp_a = st.number_input('Temp. Ambiente [K] - A', value=air_temp, step=0.1, format="%.1f", disabled=True)
            process_temp_a = st.number_input('Temp. Proceso [K] - A', value=process_temp, step=0.1, format="%.1f", disabled=True)
            rot_speed_a = st.number_input('Velocidad [rpm] - A', value=rot_speed, step=1.0, format="%.0f", disabled=True)
            torque_a = st.number_input('Torque [Nm] - A', value=torque, step=0.1, format="%.1f", disabled=True)
            wear_a = st.number_input('Desgaste [min] - A', value=wear, step=1.0, format="%.0f", disabled=True)
            type_a = st.selectbox('Tipo de producto - A', ['L','M','H'], index=['L','M','H'].index(prod_type), disabled=True)
            
        with colB:
            st.markdown('### Escenario B')
            st.info('Ingresa los parámetros operativos para el escenario B')
            air_temp_b = st.number_input('Temp. Ambiente [K] - B', value=298.0, step=0.1, format="%.1f", key='air_b')
            process_temp_b = st.number_input('Temp. Proceso [K] - B', value=308.0, step=0.1, format="%.1f", key='proc_b')
            rot_speed_b = st.number_input('Velocidad [rpm] - B', value=1200.0, step=1.0, format="%.0f", key='rpm_b')
            torque_b = st.number_input('Torque [Nm] - B', value=35.0, step=0.1, format="%.1f", key='tq_b')
            wear_b = st.number_input('Desgaste [min] - B', value=75.0, step=1.0, format="%.0f", key='wear_b')
            type_b = st.selectbox('Tipo de producto - B', ['L','M','H'], index=0, key='type_b')

        scen_a = {
            'air_temp_k': air_temp_a,
            'process_temp_k': process_temp_a,
            'rot_speed_rpm': rot_speed_a,
            'torque_nm': torque_a,
            'tool_wear_min': wear_a,
            'type': type_a
        }
        
        scen_b = {
            'air_temp_k': air_temp_b,
            'process_temp_k': process_temp_b,
            'rot_speed_rpm': rot_speed_b,
            'torque_nm': torque_b,
            'tool_wear_min': wear_b,
            'type': type_b
        }

        def _score_scenario(name, data_row):
            """Calcula probabilidad y riesgo similar a la pestaña Predicción"""
            pred_s, prob_s = predict_instance(model, data_row)
            prob_cal_s = calibrated_probability(data_row, prob_s, k_neighbors=15, alpha=0.7)
            modes_s = predict_modes_multilabel(multi, data_row) if multi is not None else {}
            valid_modes = {k: v for k, v in modes_s.items() if v is not None}
            top_mode = max(valid_modes.items(), key=lambda x: x[1])[0] if valid_modes else 'N/A'
            
            return {
                'Escenario': name,
                'Prob_fallo': prob_cal_s,
                'Modo_top': top_mode,
                'temp_amb': data_row['air_temp_k'],
                'temp_proc': data_row['process_temp_k'],
                'rpm': data_row['rot_speed_rpm'],
                'torque': data_row['torque_nm'],
                'desgaste': data_row['tool_wear_min'],
                'tipo': data_row['type']
            }

        results = [
            _score_scenario('Escenario A', scen_a),
            _score_scenario('Escenario B', scen_b)
        ]
        
        # Función auxiliar para mostrar modos de fallo de un escenario
        def _show_scenario_failure_modes(scenario_name, scenario_data):
            mode_probs = predict_modes_multilabel(multi, scenario_data)
            valid_mode_probs = {k: v for k, v in mode_probs.items() if v is not None}
            
            if valid_mode_probs:
                labels = list(valid_mode_probs.keys())
                values = [valid_mode_probs[lbl] for lbl in labels]
                colors = ['#c62828' if v >= 0.6 else ('#ff8f00' if v >= 0.3 else '#2e7d32') for v in values]
                
                fig_modes = go.Figure(go.Bar(
                    x=labels, y=values, marker_color=colors, 
                    text=[f"{v:.1%}" for v in values], 
                    textposition='auto'
                ))
                fig_modes.update_yaxes(range=[0, 1], tickformat='.0%')
                fig_modes.update_layout(title=f'Tipo de fallo más probable - {scenario_name}', height=350, showlegend=False)
                
                return fig_modes, valid_mode_probs
            return None, {}
        
        # Mostrar modos de fallo para ambos escenarios
        st.markdown('---')
        st.subheader('Análisis de Tipo de fallo más Probable por Escenario')
        st.caption('Probabilidad de cada tipo de fallo en cada escenario')
        
        col_modes1, col_modes2 = st.columns(2)
        
        with col_modes1:
            fig_a, modes_a = _show_scenario_failure_modes('Escenario A', scen_a)
            if fig_a:
                st.plotly_chart(fig_a, use_container_width=True)
        
        with col_modes2:
            fig_b, modes_b = _show_scenario_failure_modes('Escenario B', scen_b)
            if fig_b:
                st.plotly_chart(fig_b, use_container_width=True)
        
        # Gráfico de riesgo por escenario
        fig_sim = go.Figure()
        for r in results:
            risk_label = 'ALTO' if r['Prob_fallo']>=0.6 else ('MODERADO' if r['Prob_fallo']>=0.3 else 'BAJO')
            color = '#c62828' if r['Prob_fallo']>=0.6 else ('#ff8f00' if r['Prob_fallo']>=0.3 else '#2e7d32')
            fig_sim.add_trace(go.Bar(
                name=r['Escenario'], 
                x=['Riesgo'], 
                y=[r['Prob_fallo']], 
                text=[f"{r['Prob_fallo']:.2%}<br>{risk_label}"], 
                textposition='auto',
                marker_color=color
            ))
        fig_sim.update_yaxes(range=[0,1], tickformat='.0%')
        fig_sim.update_layout(barmode='group', title='Comparativa de riesgo por escenario', height=420)
        st.plotly_chart(fig_sim, use_container_width=True)
        
        # Mostrar detalles de riesgo por escenario (igual que en Predicción)
        st.markdown('---')
        st.subheader('Análisis Detallado por Escenario')
        
        cols_detail = st.columns(2)
        for col, result in zip(cols_detail, results):
            with col:
                st.markdown(f"### {result['Escenario']}")
                risk_label = 'ALTO' if result['Prob_fallo']>=0.6 else ('MODERADO' if result['Prob_fallo']>=0.3 else 'BAJO')
                color = '#c62828' if result['Prob_fallo']>=0.6 else ('#ff8f00' if result['Prob_fallo']>=0.3 else '#2e7d32')
                
                st.metric('Riesgo', f"{result['Prob_fallo']:.1%}", f"{risk_label}")
                st.markdown(f"**Tipo de fallo más probable:** {result['Modo_top']}")
                st.markdown(f"**Tipo:** {result['tipo']}")
                st.markdown(f"- RPM: {result['rpm']:.0f}")
                st.markdown(f"- Torque: {result['torque']:.2f} Nm")
                st.markdown(f"- Temp. proceso: {result['temp_proc']:.1f} K")
                st.markdown(f"- Desgaste: {result['desgaste']:.0f} min")


    with tab_analysis:
        st.subheader('Análisis Comparativo y Causal de Fallos')
        st.caption('Benchmarking entre tipos de producto, correlaciones, heatmaps y reglas de decisión interpretables.')
        
        # Cargar datos históricos
        log_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'predicciones.csv')
        
        if not os.path.exists(log_path):
            st.warning('⚠️ No hay datos históricos. Primero genera predicciones para análisis.')
        else:
            try:
                hist = pd.read_csv(log_path, engine='python', on_bad_lines='warn')
                
                if hist.empty or len(hist) < 10:
                    st.info('Se necesitan al menos 10 predicciones registradas para análisis significativo.')
                else:
                    st.markdown('---')
                    st.markdown('### Benchmarking: Desempeño por Tipo de Producto')
                    
                    stats_by_type = analyze_by_product_type(hist)
                    
                    if stats_by_type:
                        # Tabla de estadísticas
                        col1, col2, col3 = st.columns(3)
                        
                        for i, (prod_type, stats) in enumerate(sorted(stats_by_type.items())):
                            col = [col1, col2, col3][i % 3]
                            with col:
                                st.markdown(f"#### Tipo **{prod_type}**")
                                st.metric('Predicciones', stats['total_predictions'])
                                st.metric('Fallos Registrados', stats['actual_failures'])
                                st.metric('Tasa de Fallos', f"{stats['failure_rate']:.1f}%")
                                st.metric('Riesgo Promedio', f"{stats['avg_prob_risk']:.1%}")
                                st.metric('Desgaste Prom.', f"{stats['avg_wear']:.0f} min")
                                st.metric('RPM Promedio', f"{stats['avg_rpm']:.0f}")
                        
                        # Gráfico comparativo
                        st.markdown('')
                        fig_bench = create_benchmark_comparison(stats_by_type)
                        if fig_bench:
                            st.plotly_chart(fig_bench, use_container_width=True)
                            st.caption('**Interpretación:** Compara tasas de fallos, riesgo promedio y desgaste entre tipos de producto.')
                    
                    st.markdown('---')
                    st.markdown('### Matriz de Correlación: Parámetros vs Fallos')
                    st.caption('Identifica qué variables operativas tienen mayor impacto en los fallos.')
                    
                    corr_df = calculate_correlation_matrix(hist)
                    
                    if not corr_df.empty:
                        # Mostrar solo las correlaciones con Machine failure
                        if 'Machine failure' in corr_df.columns:
                            failure_corr = corr_df[['Machine failure']].sort_values('Machine failure', ascending=False)
                            
                            st.markdown('#### Correlación con Fallos de Máquina')
                            col_table, col_desc = st.columns([1.5, 1])
                            
                            with col_table:
                                st.dataframe(
                                    failure_corr.style.format('{:.3f}').background_gradient(
                                        cmap='RdYlGn', subset=['Machine failure']
                                    ),
                                    use_container_width=True
                                )
                            
                            with col_desc:
                                st.markdown("""
                                **Interpretación:**
                                - **Valores cercanos a +1**: Variable aumenta cuando hay fallo
                                - **Valores cercanos a -1**: Variable disminuye cuando hay fallo
                                - **Valores cercanos a 0**: Poca o nula relación con fallos
                                """)
                        
                        # Heatmap completo
                        st.markdown('')
                        fig_corr = create_correlation_heatmap(corr_df)
                        if fig_corr:
                            st.plotly_chart(fig_corr, use_container_width=True)
                    else:
                        st.info('No hay datos suficientes para calcular correlación.')
                    
                    st.markdown('---')
                    st.markdown('### Heatmaps Causales: Fallos por Parámetros')
                    st.caption('Visualiza la tasa de fallos según combinación de dos parámetros.')
                    
                    col_heat1, col_heat2 = st.columns(2)
                    
                    with col_heat1:
                        param1_heat = st.selectbox(
                            'Parámetro 1 (Eje Y):',
                            ['tool_wear_min', 'air_temp_k', 'rot_speed_rpm', 'torque_nm'],
                            key='param1_heat'
                        )
                    
                    with col_heat2:
                        param2_heat = st.selectbox(
                            'Parámetro 2 (Eje X):',
                            ['rot_speed_rpm', 'tool_wear_min', 'torque_nm', 'air_temp_k'],
                            key='param2_heat'
                        )
                    
                    if param1_heat != param2_heat:
                        pivot_heat = calculate_failure_rates_by_bins(hist, param1_heat, param2_heat, bins1=4, bins2=4)
                        
                        if not pivot_heat.empty:
                            fig_heat = create_failure_heatmap(pivot_heat, param1_heat, param2_heat)
                            if fig_heat:
                                st.plotly_chart(fig_heat, use_container_width=True)
                                st.caption('**Interpretación:** Las zonas rojas (más claras) indican mayor tasa de fallos en esa combinación de parámetros.')
                        else:
                            st.warning('No hay datos suficientes para este par de parámetros.')
                    else:
                        st.warning('Selecciona dos parámetros diferentes.')
                    
                    st.markdown('---')
                    st.markdown('### Análisis 3D Interactivo')
                    st.caption('Explora tres dimensiones simultáneamente. Los colores indican presencia de fallo.')
                    
                    col_3d1, col_3d2, col_3d3 = st.columns(3)
                    
                    with col_3d1:
                        x_3d = st.selectbox('Eje X:', ['air_temp_k', 'rot_speed_rpm', 'torque_nm', 'tool_wear_min'], 
                                           key='x_3d', index=0)
                    
                    with col_3d2:
                        y_3d = st.selectbox('Eje Y:', ['rot_speed_rpm', 'torque_nm', 'air_temp_k', 'tool_wear_min'],
                                           key='y_3d', index=1)
                    
                    with col_3d3:
                        z_3d = st.selectbox('Eje Z:', ['tool_wear_min', 'torque_nm', 'air_temp_k', 'rot_speed_rpm'],
                                           key='z_3d', index=2)
                    
                    if len({x_3d, y_3d, z_3d}) == 3:  # Verificar que sean diferentes
                        fig_3d = create_3d_scatter(hist, x_3d, y_3d, z_3d, 'Machine failure')
                        if fig_3d:
                            st.plotly_chart(fig_3d, use_container_width=True)
                            st.caption('**Interactivo:** Puedes rotar, hacer zoom y pasar el mouse para ver detalles. Los puntos rojos indican fallos.')
                    else:
                        st.warning('Selecciona tres ejes diferentes.')
                    
            except Exception as e:
                st.error(f'Error procesando análisis: {e}')
                import traceback
                st.code(traceback.format_exc())


    with tab_info:
        st.markdown('## Concepto de Automatas')
        st.markdown('''**Automatas** es un sistema inteligente de mantenimiento predictivo para centros CNC con diagnóstico multimodal, simulación de escenarios operativos y calibración de riesgos mediante historial inteligente. Anticipa fallos en herramientas y máquinas para permitir intervenciones proactivas que reducen paros no planificados y optimizan costos de operación.''')
        st.markdown('### Leyenda de Variables de Entrada')
        st.markdown('''<ul>
        <li><strong>Temperatura ambiente [K]</strong>: Temperatura del entorno donde opera la máquina.</li>
        <li><strong>Temperatura de proceso [K]</strong>: Temperatura alcanzada durante el mecanizado (mayor que ambiente).</li>
        <li><strong>Velocidad de rotación [rpm]</strong>: Revoluciones por minuto del husillo/herramienta.</li>
        <li><strong>Torque [Nm]</strong>: Fuerza de torsión aplicada durante el corte.</li>
        <li><strong>Desgaste herramienta [min]</strong>: Tiempo acumulado de uso de la herramienta (límite: 240 min).</li>
        </ul>''', unsafe_allow_html=True)
        st.markdown('### Indicadores de Riesgo y Umbrales Clave')
        st.markdown('''<ul>
        <li><strong>Delta térmico</strong>: Diferencia proceso - ambiente. &lt; 9 K + rotación baja (&lt;1400 rpm) implica riesgo de disipación (HDF).</li>
        <li><strong>Potencia</strong>: Producto torque * velocidad angular (W). Fuera de 3500–9000 W sugiere ineficiencia o riesgo PWF.</li>
        <li><strong>Desgaste herramienta</strong>: ≥ 200 min alcanza umbral crítico (TWF).</li>
        <li><strong>Sobrestrain (desgaste × torque)</strong>: Producto de desgaste de herramienta y torque aplicado. Superar el límite según tipo de pieza indica esfuerzo excesivo y riesgo de fallo.</li>
        </ul>''', unsafe_allow_html=True)
        st.markdown('### Leyenda de Tipos de Fallo')
        st.markdown('''<ul>
        <li><strong>TWF (Tool Wear Failure)</strong>: Fallo por Desgaste de Herramienta - La herramienta se ha desgastado demasiado y requiere cambio.</li>
        <li><strong>HDF (Heat Dissipation Failure)</strong>: Fallo por Disipación de Calor - Problemas con la refrigeración o disipación térmica insuficiente.</li>
        <li><strong>PWF (Power Failure)</strong>: Fallo por Potencia - La máquina opera fuera del rango de potencia óptimo (3.5–9 kW).</li>
        <li><strong>OSF (Overstrain Failure)</strong>: Fallo por Sobrestrain - Esfuerzo excesivo en la herramienta por combinación de desgaste y torque.</li>
        <li><strong>RNF (Random Network Failure)</strong>: Fallo Aleatorio - Fallos impredecibles causados por factores externos o errores aleatorios.</li>
        </ul>''', unsafe_allow_html=True)
        st.markdown('### Estado del Modelo')
        st.json(load_metrics_status())
        st.markdown('### Estado de modelo Multilabel')
        st.json(load_multilabel_status())
        st.markdown('### Buenas Prácticas Industriales')
        st.markdown('''<ol>
        <li>Registrar cada intervención y comparar predicción vs resultado real.</li>
        <li>Calibrar sensores de temperatura y torque de forma mensual.</li>
        <li>Programar reemplazo preventivo antes de superar 90% del desgaste máximo.</li>
        <li>Analizar tendencias de potencia para detectar deriva mecánica.</li>
        <li>Reentrenar modelo si F1 cae por debajo de objetivo o se incorporan nuevas condiciones de operación.</li>
        </ol>''', unsafe_allow_html=True)
        # Botón reentrenar
        

    with tab_hist:
        st.subheader('Histórico de Riesgo y Parámetros Operativos')
        log_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'predicciones.csv')
        
        if os.path.exists(log_path):
            try:
                hist = pd.read_csv(log_path, engine='python', on_bad_lines='warn')
                if not hist.empty:
                    # Parsear timestamps con tolerancia a formatos mixtos y zonas horarias
                    # Uso de dateutil para mayor robustez en el parseo
                    import dateutil.parser as dp
                    def _safe_parse(x):
                        try:
                            return pd.Timestamp(dp.parse(str(x)))
                        except Exception:
                            return pd.NaT
                    hist['timestamp'] = hist['timestamp'].apply(_safe_parse)
                    # Convertir timestamps con zona horaria a naive (sin tz)
                    def _drop_tz(x):
                        try:
                            if pd.notna(x) and hasattr(x, 'tz_localize'):
                                if x.tzinfo is not None:
                                    return x.tz_localize(None)
                            return x
                        except Exception:
                            return x
                    hist['timestamp'] = hist['timestamp'].apply(_drop_tz)
                    # Asegurarse de que 'prob' sea numérico
                    hist['prob'] = pd.to_numeric(hist.get('prob', pd.Series([])), errors='coerce')
                    hist = hist.sort_values('timestamp')
                    
                    # Opciones de filtrado temporal
                    st.markdown('### Filtros')
                    col_filter1, col_filter2 = st.columns(2)
                    
                    with col_filter1:
                        time_range = st.selectbox(
                            'Período:',
                            ['Última hora', 'Últimas 6 horas', 'Últimas 24 horas', 'Última semana', 'Último mes', 'Todos los registros'],
                            index=5
                        )
                    
                    with col_filter2:
                        # Mostrar total de predicciones con feedback
                        total_predictions = hist.shape[0]
                        st.metric('Total Predicciones', total_predictions)
                    
                    
                    
                    # Filtrar según rango temporal usando Timestamp sin zona horaria
                    now = pd.Timestamp.now()
                    if time_range == 'Última hora':
                        hist_filtered = hist[hist['timestamp'] >= now - pd.Timedelta(hours=1)]
                    elif time_range == 'Últimas 6 horas':
                        hist_filtered = hist[hist['timestamp'] >= now - pd.Timedelta(hours=6)]
                    elif time_range == 'Últimas 24 horas':
                        hist_filtered = hist[hist['timestamp'] >= now - pd.Timedelta(hours=24)]
                    elif time_range == 'Última semana':
                        hist_filtered = hist[hist['timestamp'] >= now - pd.Timedelta(days=7)]
                    elif time_range == 'Último mes':
                        hist_filtered = hist[hist['timestamp'] >= now - pd.Timedelta(days=30)]
                    else:
                        hist_filtered = hist
                    
                    hist_filtered = hist_filtered.tail(1000)
                    
                    # Estadísticas generales
                    st.markdown('### Estadísticas del Período')
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    col_stat1.metric('Total Predicciones', len(hist_filtered))
                    col_stat2.metric('Riesgo Promedio', f"{hist_filtered['prob'].mean():.2%}")
                    col_stat3.metric('Riesgo Máximo', f"{hist_filtered['prob'].max():.2%}")
                    col_stat4.metric('Eventos Críticos (>60%)', len(hist_filtered[hist_filtered['prob'] > 0.6]))
                    
                    st.markdown('---')
                    
                    # Gráfico principal de evolución de riesgo
                    st.markdown('### Evolución del Riesgo de Fallo')
                    fig_risk = go.Figure()
                    
                    # Línea principal
                    fig_risk.add_trace(go.Scatter(
                        x=hist_filtered['timestamp'],
                        y=hist_filtered['prob'],
                        mode='lines+markers',
                        name='Probabilidad de Fallo',
                        line=dict(color='#1976d2', width=2),
                        marker=dict(size=6),
                        hovertemplate='<b>Tiempo:</b> %{x}<br><b>Riesgo:</b> %{y:.2%}<extra></extra>'
                    ))
                    
                    # Zonas de riesgo
                    fig_risk.add_hrect(y0=0, y1=0.3, fillcolor='green', opacity=0.1, line_width=0, annotation_text='Bajo', annotation_position='left')
                    fig_risk.add_hrect(y0=0.3, y1=0.6, fillcolor='orange', opacity=0.1, line_width=0, annotation_text='Moderado', annotation_position='left')
                    fig_risk.add_hrect(y0=0.6, y1=1.0, fillcolor='red', opacity=0.1, line_width=0, annotation_text='Alto', annotation_position='left')
                    
                    fig_risk.update_layout(
                        title='Probabilidad de Fallo en el Tiempo',
                        xaxis_title='Fecha y Hora',
                        yaxis_title='Probabilidad de Fallo',
                        height=500,
                        hovermode='x unified',
                        yaxis=dict(tickformat='.0%', range=[0, 1])
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)
                    
                    # Gráficos de parámetros operativos
                    st.markdown('### Evolución de Parámetros Operativos')
                    
                    # Crear subplots para múltiples parámetros
                    fig_params = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=['Temperatura (K)', 'Velocidad Rotación (rpm)', 'Torque (Nm)', 'Desgaste (min)'],
                        vertical_spacing=0.15,
                        horizontal_spacing=0.12
                    )
                    
                    # Temperatura
                    fig_params.add_trace(go.Scatter(
                        x=hist_filtered['timestamp'], y=hist_filtered['air_temp_k'],
                        name='Aire', mode='lines', line=dict(color='#1976d2', width=2)
                    ), row=1, col=1)
                    fig_params.add_trace(go.Scatter(
                        x=hist_filtered['timestamp'], y=hist_filtered['process_temp_k'],
                        name='Proceso', mode='lines', line=dict(color='#d32f2f', width=2)
                    ), row=1, col=1)
                    
                    # Velocidad
                    fig_params.add_trace(go.Scatter(
                        x=hist_filtered['timestamp'], y=hist_filtered['rot_speed_rpm'],
                        name='RPM', mode='lines', line=dict(color='#6a1b9a', width=2), showlegend=False
                    ), row=1, col=2)
                    
                    # Torque
                    fig_params.add_trace(go.Scatter(
                        x=hist_filtered['timestamp'], y=hist_filtered['torque_nm'],
                        name='Torque', mode='lines', line=dict(color='#00838f', width=2), showlegend=False
                    ), row=2, col=1)
                    
                    # Desgaste
                    fig_params.add_trace(go.Scatter(
                        x=hist_filtered['timestamp'], y=hist_filtered['tool_wear_min'],
                        name='Desgaste', mode='lines', line=dict(color='#ff8f00', width=2), showlegend=False
                    ), row=2, col=2)
                    fig_params.add_hline(y=200, line_dash='dash', line_color='red', annotation_text='Crítico', row=2, col=2)
                    
                    # Actualizar layout para mejor visualización
                    fig_params.update_layout(
                        height=700,
                        showlegend=True,
                        font=dict(size=13),
                        title_text='Tendencias Temporales de Variables Operativas',
                        title_font_size=16,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    fig_params.update_xaxes(title_font_size=12, tickfont_size=11)
                    fig_params.update_yaxes(title_font_size=12, tickfont_size=11)
                    fig_params.update_annotations(font_size=14)
                    
                    st.plotly_chart(fig_params, use_container_width=True)
                    
                    st.caption(f'Mostrando {len(hist_filtered)} registros del período seleccionado. Las líneas muestran la tendencia temporal de cada parámetro.')
                    
                    # Tabla resumen con más registros
                    st.markdown('### Tabla Detallada de Predicciones')
                    num_show = st.slider('Cantidad de registros a mostrar:', 5, 100, 20)
                    display_cols = ['timestamp','air_temp_k','process_temp_k','rot_speed_rpm','torque_nm','tool_wear_min','type','prob']
                    st.dataframe(
                        hist_filtered.tail(num_show)[display_cols].style.format({'prob': '{:.2%}'}),
                        use_container_width=True,
                        height=400
                    )
                else:
                    st.info('Log vacío. Genere predicciones para ver el histórico.')
            except Exception as e:
                st.error(f'Error leyendo histórico: {e}')
        else:
            st.info('No existe archivo de historial todavía. Genere una predicción en la pestaña "Predicción" para crear el log.')

if __name__ == '__main__':
    print("Ejecute con: streamlit run app/streamlit_app.py")
    main()

# Estilos CSS para tarjetas
st.markdown("""
<style>
.reco-card{padding:8px 10px;margin-bottom:6px;border-radius:6px;font-size:14px;line-height:1.3;border:1px solid #e0e0e0;}
.reco-card{color:#111;background:#ffffffcc;}
.reco-high{background:#ffebee;border-left:6px solid #c62828;color:#111;}
.reco-medium{background:#fff3cd;border-left:6px solid #ff8f00;color:#111;}
.reco-low{background:#e8f5e9;border-left:6px solid #2e7d32;color:#111;}
</style>
""", unsafe_allow_html=True)
