import streamlit as st
import joblib
import pandas as pd
import os
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
import time
import altair as alt

# Configuración de página (debe ser lo primero)
st.set_page_config(
    page_title="Mantenimiento Predictivo AI",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/jlbjulio/mantenimiento-predictivo',
        'Report a bug': 'https://github.com/jlbjulio/mantenimiento-predictivo/issues',
        'About': '# Sistema Inteligente de Mantenimiento Predictivo\nVersion 1.0 - Nov 2025'
    }
)

# Asegura importación del paquete src
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data.data_loader import engineer_features, normalize_columns, load_dataset, find_dataset
from src.data.data_loader import augment_dataset
from src.ml.recommendation import generate_recommendations
from src.ml.shap_utils import shap_for_instance
from src.ml import train as train_module
try:
    from src.ml.combine_feedback import load_predictions, load_feedback, combine_predictions_with_feedback, save_labeled_data, upgrade_pred_log
except Exception:
    # Be tolerant if newer helper not available (e.g. on older deployments); fallback gracefully
    from src.ml.combine_feedback import load_predictions, load_feedback, combine_predictions_with_feedback, save_labeled_data
    upgrade_pred_log = None

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, 'failure_binary_model.joblib')
MULTI_PATH = os.path.join(MODELS_DIR, 'failure_multilabel_models.joblib')
METRICS_PATH = os.path.join(MODELS_DIR, 'failure_binary_metrics.joblib')

#########################
# Utilidades y helpers (antes de main para evitar NameError)
#########################
import csv
import shutil
from datetime import datetime

LOG_PATH = os.path.join(ROOT_DIR, 'logs')
os.makedirs(LOG_PATH, exist_ok=True)
PRED_LOG = os.path.join(LOG_PATH, 'predicciones.csv')
# Feedback handled inside pred CSV now; keep FEEDBACK_LOG for compatibility but we won't write to it
FEEDBACK_LOG = os.path.join(LOG_PATH, 'feedback.csv')

@st.cache_resource
def load_models():
    model = joblib.load(MODEL_PATH)
    multi = joblib.load(MULTI_PATH) if os.path.exists(MULTI_PATH) else None
    return model, multi

def prepare_feature_row(user_data: dict) -> pd.DataFrame:
    base_df = load_dataset()
    # Use canonical feature list to ensure exact same pipeline columns
    feature_cols = [
        'air_temp_k','process_temp_k','rot_speed_rpm','torque_nm','tool_wear_min',
        'type','uid','product_id','delta_temp_k','omega_rad_s','power_w','wear_pct'
    ]
    template = {c: None for c in feature_cols}
    template.update({
        'air_temp_k': user_data['air_temp_k'],
        'process_temp_k': user_data['process_temp_k'],
        'rot_speed_rpm': user_data['rot_speed_rpm'],
        'torque_nm': user_data['torque_nm'],
        'tool_wear_min': user_data['tool_wear_min'],
        'type': user_data['type'],
        # Technical identifiers are not used by the model (explicitly excluded in preprocessor)
        'uid': 0,
        'product_id': f"{user_data['type']}_SIM"
    })
    df = pd.DataFrame([template])
    df = engineer_features(df)
    # Reordenar
    # Ensure all canonical columns present
    for col in feature_cols:
        if col not in df.columns:
            df[col] = None
    df = df[feature_cols]
    return df


def _to_original_schema(df_row: pd.DataFrame, base_file: str):
    """Convert a canonical (normalized) df_row to the original ai4i2020.csv schema, if possible.
    This uses the header of the base CSV as the target column names and maps common normalized names back.
    """
    try:
        # read header of base file
        import pandas as pd
        head = pd.read_csv(base_file, nrows=0)
        orig_cols = list(head.columns)
        # Build mapping original -> normalized
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
        # build output dict matching orig_cols using scalar values
        out = {}
        # Handle df_row being either a DataFrame with one row or a Series
        if isinstance(df_row, pd.DataFrame):
            if df_row.shape[0] >= 1:
                row_series = df_row.iloc[0]
            else:
                row_series = pd.Series([])
        else:
            row_series = df_row
        for c in orig_cols:
            norm = mapping.get(c, c)  # map original to normalized
            # Prefer normalized name
            val = row_series.get(norm, None) if norm in row_series.index else row_series.get(c, None)
            if pd.notna(val):
                out[c] = val
            else:
                out[c] = ''
        return out
    except Exception:
        # fallback: return the canonical row as dict, unmatched columns left empty
        try:
            return df_row.to_dict(orient='records')[0]
        except Exception:
            return {}


def _align_input_with_pipeline(df_row: pd.DataFrame, pipeline):
    """Ensure df_row has columns expected by the pipeline's preprocessor (or pipeline.feature_names_in_)
    Adds missing columns with NA and reorders columns to match expected order.
    """
    import pandas as pd
    try:
        pre = pipeline.named_steps.get('pre', None)
        expected_cols = []
        if pre is not None and hasattr(pre, 'transformers_'):
            # Attempt to extract canonical input columns from the transformer's selectors
            try:
                for _name, _tr, _cols in pre.transformers_:
                    if _cols is None:
                        continue
                    # slice
                    if isinstance(_cols, slice):
                        expected_cols.extend(list(df_row.columns[_cols]))
                        continue
                    # array/list or tuple of indices/column names
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
        # Prefer pre.feature_names_in_ to preserve exact input ordering (sklearn checks this)
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
            # Default numeric-like columns to np.nan, others to empty string
            if c in CANONICAL_NUMERIC or 'prob' in c.lower():
                df_row[c] = np.nan
            else:
                df_row[c] = ''
    # Reorder, plus keep any extra cols at end
    extras = [c for c in df_row.columns if c not in expected_cols]
    ordered_cols = expected_cols + extras
    return df_row[ordered_cols]

def predict_instance(model, user_data: dict):
    df_row = prepare_feature_row(user_data)
    # Ensure df_row contains columns expected by model pipeline (old models may expect additional log fields)
    try:
        df_row_aligned = _align_input_with_pipeline(df_row, model)
    except Exception:
        df_row_aligned = df_row
    prob = model.predict_proba(df_row_aligned)[0][1]
    pred = model.predict(df_row_aligned)[0]
    return pred, prob

def best_model_name():
    if os.path.exists(METRICS_PATH):
        try:
            m = joblib.load(METRICS_PATH)
            return m.get('best', 'desconocido')
        except Exception:
            return 'desconocido'
    return 'no entrenado'

def load_metrics_status():
    status = {}
    if os.path.exists(METRICS_PATH):
        try:
            data = joblib.load(METRICS_PATH)
            status['best_model'] = data.get('best')
            status['aucs'] = data.get('aucs')
        except Exception as e:
            status['error'] = str(e)
    else:
        status['message'] = 'Ejecute entrenamiento: python -m src.ml.train'
    if os.path.exists(MODEL_PATH):
        status['model_file'] = MODEL_PATH
        status['last_modified'] = str(pd.to_datetime(os.path.getmtime(MODEL_PATH), unit='s'))
    return status

def log_prediction(data: dict, prob: float, pred: int, machine_failure: int = None, feedback_timestamp: pd.Timestamp = None):
    # Safety guard: allow suppressing writes during UI actions like "Borrar predicción"
    try:
        if st.session_state.get('suppress_logging', False):
            return
    except Exception:
        pass
    header = ['timestamp','air_temp_k','process_temp_k','rot_speed_rpm','torque_nm','tool_wear_min','type','pred','prob','Machine failure','feedback_timestamp']
    write_header = not os.path.exists(PRED_LOG)
    # Ensure CSV header includes new columns; if not, add missing columns without rewriting valid data
    full_header = ['timestamp','air_temp_k','process_temp_k','rot_speed_rpm','torque_nm','tool_wear_min','type','pred','prob','Machine failure','feedback_timestamp']
    if os.path.exists(PRED_LOG):
        try:
            with open(PRED_LOG, 'r', newline='') as f:
                first_line = f.readline().strip()
            existing_cols = [c.strip() for c in first_line.split(',') if c.strip()]
            missing = [c for c in full_header if c not in existing_cols]
            if missing:
                # Rewrite CSV with new header and append old rows with empty values for missing columns
                df_old = pd.read_csv(PRED_LOG, engine='python', on_bad_lines='warn')
                for c in missing:
                    df_old[c] = ''
                df_old.to_csv(PRED_LOG, index=False, columns=full_header)
                write_header = False
        except Exception:
            # best effort; ignore
            pass
    def _write(ts):
        # Normalize timestamps to ISO seconds to avoid parse issues
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
        with open(PRED_LOG, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow([ts, data['air_temp_k'], data['process_temp_k'], data['rot_speed_rpm'], data['torque_nm'], data['tool_wear_min'], data['type'], pred, f"{prob:.4f}", machine_failure if machine_failure is not None else '', fb_ts])
    # Accept timestamp override if provided
    # deduplicate/update: if a row with the same prediction timestamp (string exact) already exists, update it
    try:
        if os.path.exists(PRED_LOG):
            existing = pd.read_csv(PRED_LOG)
            if 'timestamp' in existing.columns and not existing.empty:
                try:
                    provided_raw = data.get('prediction_timestamp')
                    provided_dt = pd.to_datetime(provided_raw, errors='coerce')
                    if pd.isna(provided_dt):
                        raise ValueError('Invalid prediction timestamp')
                    provided_str = provided_dt.strftime('%Y-%m-%dT%H:%M:%S')
                    # First attempt: exact string match on timestamp
                    mask = (existing['timestamp'].astype(str) == provided_str)
                    if not mask.any():
                        # Fallback: match rows with empty Machine failure and identical feature values
                        feature_cols = ['air_temp_k','process_temp_k','rot_speed_rpm','torque_nm','tool_wear_min','type','pred','prob']
                        try:
                            for col in feature_cols:
                                if col == 'prob':
                                    # Prob stored as string formatted to 4 decimals; compare rounding
                                    prob_str = f"{prob:.4f}" if prob is not None else ''
                                    mask_feat = (existing[col].astype(str) == prob_str)
                                else:
                                    mask_feat = (existing[col].astype(str) == str(data.get(col if col!='pred' and col!='prob' else col)))
                                mask = mask & mask_feat if 'mask' in locals() and len(mask) == len(existing) else mask_feat
                            # Only keep candidates with empty Machine failure
                            if 'Machine failure' in existing.columns:
                                mask = mask & (existing['Machine failure'].astype(str).isin(['','nan']))
                        except Exception:
                            pass
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
    

# Simple version in use; advanced removal logic removed per user request
 

def remove_last_prediction_row() -> bool:
    """Simplest path: delete the last row in predicciones.csv (if any)."""
    try:
        if not os.path.exists(PRED_LOG):
            return False
        df = pd.read_csv(PRED_LOG)
        if df.empty:
            return False
        df = df.iloc[:-1].copy()
        df.to_csv(PRED_LOG, index=False)
        return True
    except Exception:
        return False


def ensure_pred_log_has_history(model, count: int = 500):
    """Seed predicciones.csv with entries from base dataset to create initial history.
    Only seeds when file doesn't exist or has few rows (< count).
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
        # write in bulk
        write_header = not os.path.exists(PRED_LOG)
        header = ['timestamp','air_temp_k','process_temp_k','rot_speed_rpm','torque_nm','tool_wear_min','type','pred','prob']
        with open(PRED_LOG, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            for r in rows:
                w.writerow(r)
    except Exception:
        # ignore seeding errors
        return

def log_feedback(timestamp: str, actual_failure: int):
    """Backward-compatible: write feedback directly in predicciones.csv for the given prediction timestamp.
    If a prediction entry doesn't exist, create it with missing prediction values.
    """
    try:
        # Attempt to find existing prediction row and update it
        if os.path.exists(PRED_LOG):
            df = pd.read_csv(PRED_LOG)
            ts = str(timestamp)
            if 'timestamp' in df.columns and ts in df['timestamp'].astype(str).tolist():
                # Ensure text columns are dtype object to avoid dtype deprecation warnings
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
        # If not found, create a new prediction entry with minimal fields
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
    # expects failure_prob column
    if 'failure_prob' not in df.columns:
        return
    write_header = not os.path.exists(PRED_LOG)
    header = ['timestamp','air_temp_k','process_temp_k','rot_speed_rpm','torque_nm','tool_wear_min','type','pred','prob','Machine failure','feedback_timestamp']
    with open(PRED_LOG, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        for _, row in df.iterrows():
            w.writerow([pd.Timestamp.utcnow(), row.get('air_temp_k'), row.get('process_temp_k'), row.get('rot_speed_rpm'), row.get('torque_nm'), row.get('tool_wear_min'), row.get('type'), 'NA', f"{row.get('failure_prob'):.4f}", '', ''])

def run_retrain():
    # Llama la función main del módulo de entrenamiento
    try:
        # Ejecutar en hilo para no bloquear la UI
        from threading import Thread
        t = Thread(target=train_module.main, daemon=True)
        t.start()
    except Exception as e:
        st.error(f"Error en reentrenamiento: {e}")


def save_row_to_additional(df_row: pd.DataFrame, prefix: str = 'augmented_manual') -> str:
    """Save a single row to data/additional/<prefix>_<timestamp>.csv and return path."""
    try:
        ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'additional')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{prefix}_{ts}.csv")
        import pandas as pd
        df_row.to_csv(out_path, index=False)
        return out_path
    except Exception as e:
        return None


def append_row_to_base(df_row: pd.DataFrame) -> str:
    """Append row to base ai4i2020.csv with a timestamped backup; return backup path.
    Returns backup path on success, None otherwise.
    """
    try:
        base_path = find_dataset()
        backup_path = f"{base_path}.bak_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(base_path, backup_path)
        # Convert df_row to original schema
        rec = _to_original_schema(df_row.iloc[0], base_path)
        # Append as a single-line CSV (keeping original header order)
        import pandas as pd
        base_df = pd.read_csv(base_path, dtype=str)
        # create a new row with same columns
        new_row = {c: rec.get(c, '') for c in base_df.columns}
        new_row_df = pd.DataFrame([new_row], columns=base_df.columns)
        base_df = pd.concat([base_df, new_row_df], ignore_index=True)
        # Write back (preserve original CSV formatting)
        base_df.to_csv(base_path, index=False)
        return backup_path
    except Exception as e:
        return None


def main():
    st.title('Sistema Inteligente de Mantenimiento Predictivo para maquinaria industrial')
    st.caption('Monitoreo avanzado de parámetros operativos, predicción de fallos y recomendaciones accionables.')
    model, multi = load_models()
    # Inicializar estado de sesión para almacenar última predicción
    try:
        if 'last_prediction_data' not in st.session_state:
            st.session_state.last_prediction_data = None
    except Exception:
        # st.session_state may not be available outside of Streamlit runtime
        pass
    # Automatic re-training is enforced (no user toggle) — default threshold 1
    try:
        st.session_state.auto_retrain_enabled = True
        st.session_state.retrain_threshold = 1
    except Exception:
        pass

    base_df = load_dataset()
    def dyn_range(col, pad=0.05):
        if col not in base_df.columns:
            return (0.0, 1.0)
        mn, mx = base_df[col].min(), base_df[col].max()
        span = mx - mn
        return float(mn - span*pad), float(mx + span*pad)

    # Keep dyn_range for reference but do not limit inputs by dataset bounds
    a_min, a_max = dyn_range('air_temp_k')
    p_min, p_max = dyn_range('process_temp_k')
    r_min, r_max = dyn_range('rot_speed_rpm')
    tq_min, tq_max = dyn_range('torque_nm')
    w_min, w_max = dyn_range('tool_wear_min')

    st.sidebar.header('Entrada de Parámetros Operativos')
    # Check if there's a pending prediction to disable inputs
    has_pending_prediction = st.session_state.get('last_prediction_data') is not None
    if has_pending_prediction:
        st.sidebar.warning('Confirma el feedback de la predicción actual para modificar parámetros.')
    # Use number_input instead of sliders to allow values beyond dataset bounds
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

**Leyenda de umbrales críticos:**
- Delta térmico crítico < 9 K con baja rotación (<1400 rpm)
- Potencia segura 3500–9000 W
- Reemplazo herramienta ≥ 200 min
""")

    tab_pred, tab_info, tab_explain, tab_hist = st.tabs(["Predicción", "Info / Ayuda", "Explicabilidad", "Histórico"])

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
        st.caption('Indicadores clave del instante operativo. El desgaste (%) se calcula sobre 240 min como límite superior.')
        st.markdown('<hr/>', unsafe_allow_html=True)
        
        # Disable prediction button if there's a pending prediction without feedback
        has_pending_prediction = st.session_state.get('last_prediction_data') is not None
        if has_pending_prediction:
            st.info('Debes confirmar el feedback de la predicción actual antes de calcular una nueva o modificar parámetros.')
        
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
            now_ts = pd.Timestamp.utcnow()
            st.session_state.last_prediction_data = {**data, 'pred': int(pred), 'prob': float(prob), 'prediction_timestamp': now_ts}
            log_prediction(
                data={**data, 'prediction_timestamp': now_ts},
                prob=float(prob),
                pred=int(pred),
                machine_failure=None,
                feedback_timestamp=None
            )
            st.rerun()

        # Render resultado + acciones si hay predicción activa en sesión
        if st.session_state.get('last_prediction_data') is not None:
            data_view = st.session_state.get('last_prediction_data')
            prob_v = float(data_view.get('prob', 0.0))
            pred_v = int(data_view.get('pred', 0))
            st.subheader('Resultado de Predicción')
            risk_label = 'ALTO' if prob_v>=0.6 else ('MODERADO' if prob_v>=0.3 else 'BAJO')
            st.markdown(f"### Riesgo de fallo: **{risk_label}** - **{prob_v:.2f}**")
            st.markdown(f"Modelo seleccionado: **{best_model_name()}**")

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_v*100,
                title={'text': 'Índice de Riesgo (%)'},
                gauge={
                    'axis': {'range': [0,100]},
                    'steps': [
                        {'range':[0,30],'color':'#2e7d32'},
                        {'range':[30,60],'color':'#ffb300'},
                        {'range':[60,100],'color':'#c62828'}
                    ],
                    'threshold': {'line': {'color': '#c62828', 'width': 4}, 'thickness': 0.75, 'value': prob_v*100}
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

            st.subheader('Recomendaciones Acción Inmediata')
            recs = generate_recommendations(data_view, prob_v, shap_contrib=None)
            for r in recs:
                sev_class = 'reco-high' if 'reemplazo' in r['accion'].lower() or 'inspección' in r['accion'].lower() else ('reco-medium' if 'ajustar' in r['accion'].lower() or 'reducir' in r['accion'].lower() else 'reco-low')
                st.markdown(
                    f"<div class='reco-card {sev_class}'><strong>{r['accion']}</strong><br><span style='font-size:13px;'>{r['justificacion']}</span></div>",
                    unsafe_allow_html=True)
            st.markdown("""
                        **Clasificación de severidad:**
                        - Alto (rojo): Acción inmediata / reemplazo / inspección.
                        - Medio (naranja): Ajuste operativo recomendado pronto.
                        - Bajo (verde): Condición estable, monitoreo continuo.
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
                        # Clear active prediction UI and avoid re-logging the same prediction
                        st.session_state.last_prediction_data = None
                        st.session_state.suppress_logging = True
                        try:
                            preds_df = load_predictions(PRED_LOG)
                            combined = combine_predictions_with_feedback(preds_df, None)
                            if combined is not None:
                                out = save_labeled_data(combined, os.path.join(os.path.dirname(__file__), '..', 'data', 'additional'))
                                if out is not None:
                                    out_path, new_rows = out
                                    if new_rows >= 1:
                                        run_retrain()
                        except Exception as e:
                            st.warning(f'No se pudo combinar/guardar feedback: {e}')
                        st.rerun()
                    else:
                        st.error('No hay predicción activa. Primero realiza una predicción.')
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
                        # Clear active prediction UI and avoid re-logging the same prediction
                        st.session_state.last_prediction_data = None
                        st.session_state.suppress_logging = True
                        try:
                            preds_df = load_predictions(PRED_LOG)
                            combined = combine_predictions_with_feedback(preds_df, None)
                            if combined is not None:
                                out = save_labeled_data(combined, os.path.join(os.path.dirname(__file__), '..', 'data', 'additional'))
                                if out is not None:
                                    out_path, new_rows = out
                                    if new_rows >= 1:
                                        run_retrain()
                        except Exception as e:
                            st.warning(f'No se pudo combinar/guardar feedback: {e}')
                        st.rerun()
                    else:
                        st.error('No hay predicción activa. Primero realiza una predicción.')

            if st.button('Borrar predicción actual', key='clear_pred'):
                try:
                    if not os.path.exists(PRED_LOG):
                        st.session_state.last_prediction_data = None
                        st.session_state.suppress_logging = True
                        st.warning('No hay predicciones registradas. Genera una predicción antes de borrar.')
                    else:
                        try:
                            _df_chk = pd.read_csv(PRED_LOG)
                            is_empty = _df_chk.empty
                        except Exception:
                            is_empty = True
                        if is_empty:
                            st.session_state.last_prediction_data = None
                            st.session_state.suppress_logging = True
                            st.warning('No hay predicciones registradas. Genera una predicción antes de borrar.')
                        else:
                            removed = remove_last_prediction_row()
                            st.session_state.last_prediction_data = None
                            st.session_state.suppress_logging = True
                            if removed:
                                st.success('🗑️ Última fila eliminada de predicciones. No se registrará feedback para esta instancia.')
                            else:
                                st.info('No se pudo eliminar. Sesión limpiada y sin registro adicional.')
                            st.rerun()
                except Exception:
                    st.info('No se pudo eliminar. Sesión limpiada y sin registro adicional.')
                    st.rerun()



    with tab_info:
        st.markdown('## Concepto del Sistema')
        st.markdown('''**Sistema Inteligente de Mantenimiento Predictivo** que integra datos históricos y en línea para anticipar modos de fallo y guiar intervenciones proactivas, reduciendo paros no planificados y costos de operación.''')
        st.markdown('### Leyenda y Umbrales Clave')
        st.markdown('''<ul>
        <li><strong>Delta térmico</strong>: Diferencia proceso - ambiente. &lt; 9 K + rotación baja (&lt;1400 rpm) implica riesgo de disipación (HDF).</li>
        <li><strong>Potencia</strong>: Producto torque * velocidad angular (W). Fuera de 3500–9000 W sugiere ineficiencia o riesgo PWF.</li>
        <li><strong>Desgaste herramienta</strong>: ≥ 200 min alcanza umbral crítico (TWF).</li>
        <li><strong>Sobrestrain</strong>: wear * torque supera límite según tipo L/M/H.</li>
        </ul>''', unsafe_allow_html=True)
        st.markdown('### Estado del Modelo')
        st.json(load_metrics_status())
        st.markdown('### Buenas Prácticas Industriales')
        st.markdown('''<ol>
        <li>Registrar cada intervención y comparar predicción vs resultado real.</li>
        <li>Calibrar sensores de temperatura y torque de forma mensual.</li>
        <li>Programar reemplazo preventivo antes de superar 90% del desgaste máximo.</li>
        <li>Analizar tendencias de potencia para detectar deriva mecánica.</li>
        <li>Reentrenar modelo si F1 cae por debajo de objetivo o se incorporan nuevas condiciones de operación.</li>
        </ol>''', unsafe_allow_html=True)
        # Botón reentrenar
        

    with tab_explain:
        st.subheader('Explicabilidad de la Predicción con SHAP')
        
        # Información sobre SHAP
        with st.expander('¿Qué es SHAP y cómo interpretarlo?', expanded=False):
            st.markdown("""
            ### SHAP (SHapley Additive exPlanations)
            
            **SHAP** es una técnica avanzada de **interpretabilidad de modelos de IA** que explica cómo cada parámetro 
            operativo contribuye a la predicción de riesgo de fallo.
            
            #### ¿Cómo funciona?
            - Basado en **teoría de juegos** (valores de Shapley)
            - Calcula la **contribución marginal** de cada característica
            - Proporciona explicaciones **locales** (para cada predicción individual)
            
            #### ¿Cómo interpretar los valores?
            
            | Color | Significado | Interpretación |
            |-------|-------------|----------------|
            | **Rojo** | Valor positivo | Este parámetro **aumenta** el riesgo de fallo |
            | **Verde** | Valor negativo | Este parámetro **reduce** el riesgo de fallo |
            | **Magnitud** | Tamaño del número | Mayor valor = Mayor impacto en la predicción |
            
            #### Ejemplo práctico:
            - `torque_nm: +0.2150` (Rojo) → El torque actual está aumentando significativamente el riesgo
            - `delta_temp: -0.0450` (Verde) → El delta térmico está ayudando a reducir el riesgo
            - `tool_wear_min: +0.1820` (Rojo) → El desgaste contribuye al riesgo moderadamente
            
            #### ¿Por qué es importante?
            - **Transparencia**: Entender qué factores influyen en cada decisión
            - **Confianza**: Validar que el modelo considera factores correctos
            - **Acción**: Identificar qué parámetros ajustar para reducir riesgo
            """)
        
        if st.session_state.get('last_prediction_data', None) is None:
            st.warning('Primero realiza una predicción en la pestaña "Predicción" para generar explicaciones SHAP.')
        else:
            # Mostrar parámetros de la última predicción
            st.markdown('#### Parámetros de la Predicción Actual')
            data = st.session_state.get('last_prediction_data')
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric('Temp. Ambiente', f"{data['air_temp_k']:.1f} K")
                st.metric('Temp. Proceso', f"{data['process_temp_k']:.1f} K")
            with col2:
                st.metric('Velocidad', f"{data['rot_speed_rpm']:.0f} rpm")
                st.metric('Torque', f"{data['torque_nm']:.1f} Nm")
            with col3:
                st.metric('Desgaste', f"{data['tool_wear_min']:.0f} min")
                st.metric('Tipo Producto', data['type'])
            
            st.markdown('---')
            
            if st.button('Generar Explicabilidad SHAP', type='primary'):
                with st.spinner('Calculando explicaciones SHAP... (esto puede tomar unos segundos)'):
                    try:
                        contrib = shap_for_instance(data)
                    except Exception as e:
                        import traceback
                        st.error(f"Error generando SHAP: {e}")
                        st.code(traceback.format_exc())
                        contrib = None
                    
                    if contrib is None:
                        st.warning('No fue posible generar explicaciones SHAP para la instancia solicitada.')
                    else:
                        st.markdown('### Análisis de Contribución de Factores')
                        st.caption('Los siguientes factores explican cómo cada parámetro influye en la predicción de riesgo:')
                        
                        # Crear visualización mejorada
                        import plotly.express as px
                        
                        # Preparar datos para gráfica
                        features = [feat for feat, _ in contrib]
                        values = [val for _, val in contrib]
                        colors = ['#d32f2f' if v > 0 else '#2e7d32' for v in values]
                    
                        # Gráfica de barras horizontal
                        fig_shap = go.Figure()
                        fig_shap.add_trace(go.Bar(
                            y=features[::-1],  # Invertir para mostrar el más importante arriba
                            x=values[::-1],
                            orientation='h',
                            marker_color=colors[::-1],
                            text=[f"{v:+.4f}" for v in values[::-1]],
                            textposition='outside'
                        ))
                        
                        fig_shap.update_layout(
                            title='Contribución SHAP de cada Factor al Riesgo de Fallo',
                            xaxis_title='Impacto en la Predicción (SHAP Value)',
                            yaxis_title='Parámetro Operativo',
                            height=600,
                            showlegend=False,
                            font=dict(size=12)
                        )
                        fig_shap.add_vline(x=0, line_width=2, line_color='black', line_dash='dash')
                        
                        st.plotly_chart(fig_shap, use_container_width=True)
                    
                        # Mostrar tabla detallada con TODAS las variables
                        st.markdown('#### Detalle de Contribuciones')
                        st.caption('Nota: Si una variable aparece con contribución 0 puede ser porque el modelo no usa esa característica o está codificada/agrupada en otras (p. ej. one-hot).')
                        for i, (feat, val) in enumerate(contrib, 1):
                            is_zero = abs(float(val)) < 1e-12
                            clazz = 'shap-risk' if (val > 0 and not is_zero) else ('shap-safe' if (val <= 0 and not is_zero) else '')
                            direction = ('AUMENTA' if val > 0 else 'REDUCE') if not is_zero else '—'
                            suffix = ' <span style="color:#616161;">(no usada por el modelo)</span>' if is_zero else ''
                            st.markdown(
                                f"<div class='shap-row {clazz}'>"
                                f"<strong>{i}. {feat}{suffix}</strong>: {val:+.4f} → {direction} el riesgo"
                                f"</div>", 
                                unsafe_allow_html=True
                            )
                    
                    st.markdown('---')
                    st.markdown("""
                    ### Interpretación y Acciones Recomendadas
                    
                    **Cómo usar esta información:**
                    1. **Identifique factores rojos (positivos)**: Son los que más contribuyen al riesgo
                    2. **Priorice ajustes**: Enfóquese en reducir/modificar los parámetros con mayor impacto positivo
                    3. **Mantenga factores verdes**: Los valores negativos están ayudando a reducir el riesgo
                    4. **Combine con recomendaciones**: Use las sugerencias de la pestaña "Predicción"
                    
                    **Nota técnica:** Los valores SHAP suman al valor base del modelo para obtener la predicción final.
                    """)

    with tab_hist:
        st.subheader('Histórico de Riesgo y Parámetros Operativos')
        log_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'predicciones.csv')
        
        if os.path.exists(log_path):
            try:
                hist = pd.read_csv(log_path, engine='python', on_bad_lines='warn')
                if not hist.empty:
                    # Parse timestamps with tolerance to mixed formats and timezones
                    # Parse timestamps robustly across mixed formats using dateutil for highest tolerance
                    import dateutil.parser as dp
                    def _safe_parse(x):
                        try:
                            return pd.Timestamp(dp.parse(str(x)))
                        except Exception:
                            return pd.NaT
                    hist['timestamp'] = hist['timestamp'].apply(_safe_parse)
                    # Convert tz-aware to naive (drop timezone info)
                    def _drop_tz(x):
                        try:
                            if pd.notna(x) and hasattr(x, 'tz_localize'):
                                if x.tzinfo is not None:
                                    return x.tz_localize(None)
                            return x
                        except Exception:
                            return x
                    hist['timestamp'] = hist['timestamp'].apply(_drop_tz)
                    # Ensure prob is numeric
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
                    
                    
                    
                    # Filtrar según rango temporal - usar Timestamp naive (sin timezone)
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
.shap-row{padding:4px 8px;margin:3px;border-radius:4px;font-size:13px;line-height:1.2;}
.shap-risk{background:#ffebee;border-left:6px solid #d32f2f;color:#222;}
.shap-safe{background:#e8f5e9;border-left:6px solid #2e7d32;color:#222;}
</style>
""", unsafe_allow_html=True)
