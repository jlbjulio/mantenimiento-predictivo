import os
import joblib
import shap
import numpy as np
import pandas as pd
from .train import MODELS_DIR
from ..data.data_loader import engineer_features, load_dataset

MODEL_PATH = os.path.join(MODELS_DIR, 'failure_binary_model.joblib')

def _prepare(model, row_df: pd.DataFrame):
    def _align_input_with_pipeline(model, row_df):
        import numpy as np
        df = row_df.copy()
        expected_cols = []
        # Prefer extracting expected input columns from the preprocessor's transformers_ (original features)
        if 'pre' in model.named_steps and hasattr(model.named_steps['pre'], 'transformers_'):
            try:
                for _name, _tr, _cols in model.named_steps['pre'].transformers_:
                    if _cols is None:
                        continue
                    # if it's a slice
                    if isinstance(_cols, slice):
                        expected_cols.extend(list(df.columns[_cols]))
                        continue
                    # if it's a list/tuple/ndarray
                    try:
                        iterable = list(_cols)
                        if len(iterable) > 0 and isinstance(iterable[0], int):
                            # indexes -> map to df columns
                            expected_cols.extend([df.columns[i] for i in iterable if isinstance(i, int) and i < len(df.columns)])
                        else:
                            expected_cols.extend([c for c in iterable if isinstance(c, str)])
                        continue
                    except Exception:
                        pass
                    # fallback if it's a single string
                    if isinstance(_cols, str):
                        expected_cols.append(_cols)
            except Exception:
                expected_cols = []
        # Fallbacks
        # Prefer using the preprocessor's feature_names_in_ if available (ensures exact order)
        if 'pre' in model.named_steps and hasattr(model.named_steps['pre'], 'feature_names_in_'):
            expected_cols = list(model.named_steps['pre'].feature_names_in_)
        else:
            if not expected_cols:
                if hasattr(model, 'feature_names_in_'):
                    expected_cols = list(model.feature_names_in_)
                else:
                    expected_cols = list(df.columns)
        # Fill missing expected cols
        for c in expected_cols:
            if c not in df.columns:
                # numeric-like columns default to NaN
                if any(k in c.lower() for k in ['prob', 'temp', 'rpm', 'torque', 'wear', 'rot_speed', 'prediction', 'timestamp']):
                    df[c] = np.nan
                else:
                    df[c] = ''
        # Deduplicate expected_cols preserving order
        seen = set()
        ordered_expected = []
        for c in expected_cols:
            if c not in seen:
                ordered_expected.append(c)
                seen.add(c)
        expected_cols = ordered_expected
        extras = [c for c in df.columns if c not in expected_cols]
        ordered_cols = expected_cols + extras
        return df[ordered_cols]

    if 'pre' in model.named_steps:
        # Align inputs to the pipeline's expectation
        aligned = _align_input_with_pipeline(model, row_df)
        X_trans = model.named_steps['pre'].transform(aligned)
    else:
        X_trans = row_df
    if hasattr(X_trans, 'toarray'):
        X_trans = X_trans.toarray()
    # Normalize to numeric ndarray; if object dtype (strings/arrays), convert safely
    try:
        X_trans = np.asarray(X_trans)
        if X_trans.dtype == object:
            # Convert each element to scalar using _to_scalar
            shp = X_trans.shape
            flat = [ _to_scalar(x) for x in X_trans.ravel() ]
            X_trans = np.array(flat, dtype=float).reshape(shp)
        else:
            X_trans = X_trans.astype(float)
    except Exception:
        try:
            X_trans = np.asarray(X_trans, dtype=float)
        except Exception:
            # Last resort: coerce using python float conversion per element
            flat = [ _to_scalar(x) for x in np.asarray(X_trans).ravel() ]
            X_trans = np.array(flat, dtype=float).reshape(np.asarray(X_trans).shape)
    return X_trans


def _to_scalar(val):
    """Return a float scalar from a numeric or array-like input.
    If val is array-like, returns the sum as scalar.
    """
    import numpy as _np
    try:
        arr = _np.asarray(val)
        if arr.size == 1:
            return float(arr.item())
        # If array-like with multiple elements, return sum
        return float(_np.sum(arr))
    except Exception:
        try:
            return float(val)
        except Exception:
            return 0.0


def _shap_to_array(sv, sample_idx=0, class_idx=1):
    """Normalize shap_values output to a 2D numpy array for sample/sample index.
    Returns 2D np.array of shape (n_samples, n_features) and selected class index.
    Handles shap Tree/KernalExplainer multi-class outputs and 1D/2D shapes.
    """
    import numpy as _np
    # If shap returns a list (per-class), pick class_idx if available else 0
    if isinstance(sv, list):
        if len(sv) > class_idx:
            arr = _np.asarray(sv[class_idx])
        else:
            arr = _np.asarray(sv[0])
    else:
        arr = _np.asarray(sv)
    # If arr has shape (n_samples, n_features) -> ok
    if arr.ndim == 2:
        return arr
    # If arr has shape (n_features,) -> treat as single sample
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    # If arr has shape (n_classes, n_samples, n_features)
    if arr.ndim == 3:
        # choose class dimension if class_idx in range
        if arr.shape[0] > class_idx:
            return arr[class_idx]
        else:
            return arr[0]
    # otherwise attempt to squeeze
    try:
        return _np.squeeze(arr)
    except Exception:
        return _np.asarray(arr).reshape(1, -1)

def shap_for_instance(data_dict: dict):
    model = joblib.load(MODEL_PATH)
    full_df = load_dataset()
    # Exclude technical identifiers like product_id and uid from SHAP computations
    feature_cols = [c for c in full_df.columns if c not in ['machine_failure','twf','hdf','pwf','osf','rnf','product_id','uid']]
    template = {c: None for c in feature_cols}
    template.update({
        'air_temp_k': data_dict['air_temp_k'],
        'process_temp_k': data_dict['process_temp_k'],
        'rot_speed_rpm': data_dict['rot_speed_rpm'],
        'torque_nm': data_dict['torque_nm'],
        'tool_wear_min': data_dict['tool_wear_min'],
        'type': data_dict['type'],
        'uid': 0,
        'product_id': f"{data_dict['type']}_SIM"
    })
    row_df = pd.DataFrame([template])
    row_df = engineer_features(row_df)
    for col in feature_cols:
        if col not in row_df.columns:
            row_df[col] = None
    row_df = row_df[feature_cols]
    X_trans = _prepare(model, row_df)
    clf = model.named_steps.get('clf', model)
    try:
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_trans)
        sv_used = _shap_to_array(sv)
    except Exception:
        explainer = shap.KernelExplainer(clf.predict_proba, X_trans)
        sv_all = explainer.shap_values(X_trans)
        sv_used = _shap_to_array(sv_all)
    # Map feature names post-transform if pipeline
    if 'pre' in model.named_steps:
        # OneHot feature names
        ohe = model.named_steps['pre'].transformers_[1][1].named_steps['onehot']
        num_cols = model.named_steps['pre'].transformers_[0][2]
        cat_cols = model.named_steps['pre'].transformers_[1][2]
        cat_names = list(ohe.get_feature_names_out(cat_cols))
        feature_names = num_cols + cat_names
        # Agrupar contribuciones por característica original para OHE (mejor interpretabilidad)
        agg = {}
        # sv_used normalized to 2D (n_samples, n_features)
        vals = np.asarray(sv_used)
        if vals.ndim == 2:
            vals = vals[0]
        else:
            vals = np.asarray(vals).reshape(-1)
        for fname, val in zip(feature_names, vals):
            grouped = False
            # Agrupar todas las columnas one-hot que pertenecen a la misma categoría original
            for cat in cat_cols:
                prefix = f"{cat}_"
                if fname.startswith(prefix):
                    # val might be array-like; sum to scalar if needed
                    agg_val = _to_scalar(val)
                    agg[cat] = agg.get(cat, 0.0) + agg_val
                    grouped = True
                    break
            if not grouped:
                scalar_val = _to_scalar(val)
                agg[fname] = agg.get(fname, 0.0) + scalar_val
        contrib = sorted(agg.items(), key=lambda x: abs(x[1]), reverse=True)
    else:
        feature_names = list(row_df.columns)
        # Ensure scalar values for each contribution
        vals = np.asarray(sv_used)
        if vals.ndim == 2:
            vals = vals[0]
        else:
            vals = np.asarray(vals).reshape(-1)
        safe_vals = []
        for v in vals:
            safe_vals.append(_to_scalar(v))
        contrib = sorted(zip(feature_names, safe_vals), key=lambda x: abs(x[1]), reverse=True)
    # Return all contributions by default (UI will decide how to render/limit)
    return contrib
