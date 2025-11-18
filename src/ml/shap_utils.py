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
        pre = model.named_steps['pre']
        # Get transformed feature names robustly
        feature_names_out = None
        num_cols = []
        cat_cols = []
        ohe = None
        try:
            feature_names_out = list(pre.get_feature_names_out())
        except Exception:
            feature_names_out = None
        # Also gather transformer pieces by name to compute counts if needed
        try:
            # Find 'num' and 'cat' slots irrespective of order
            for name, tr, cols in getattr(pre, 'transformers_', []):
                if name == 'num':
                    if isinstance(cols, slice):
                        num_cols = []
                    else:
                        try:
                            num_cols = list(cols)
                        except Exception:
                            num_cols = []
                if name == 'cat':
                    if isinstance(cols, slice):
                        cat_cols = []
                    else:
                        try:
                            cat_cols = list(cols)
                        except Exception:
                            cat_cols = []
                    try:
                        ohe = tr.named_steps['onehot']
                    except Exception:
                        ohe = None
        except Exception:
            pass
        # Build fallback names if necessary and ensure length alignment with SHAP values
        agg = {}
        vals = np.asarray(sv_used)
        if vals.ndim == 2:
            vals = vals[0]
        else:
            vals = np.asarray(vals).reshape(-1)
        # Preferred: use feature_names_out if it matches length
        if feature_names_out is not None and len(feature_names_out) == len(vals):
            for fname, val in zip(feature_names_out, vals):
                base = str(fname)
                if '__' in base:
                    _, base = base.split('__', 1)
                if base.startswith('type_'):
                    base = 'type'
                agg[base] = agg.get(base, 0.0) + _to_scalar(val)
        else:
            # Derive names by segmenting vals into numeric + onehot parts using counts
            n_num = len(num_cols)
            n_cat = 0
            cat_names = []
            try:
                if ohe is not None and cat_cols:
                    cat_names = list(ohe.get_feature_names_out(cat_cols))
                    n_cat = len(cat_names)
            except Exception:
                cat_names = []
                n_cat = 0
            if n_num + n_cat == len(vals) and (n_num > 0 or n_cat > 0):
                # Map first n_num to numeric names, rest to 'type'
                for i in range(n_num):
                    name = str(num_cols[i]) if i < len(num_cols) else f"num_{i}"
                    agg[name] = agg.get(name, 0.0) + _to_scalar(vals[i])
                for j in range(n_num, n_num + n_cat):
                    # group all one-hot to 'type'
                    agg['type'] = agg.get('type', 0.0) + _to_scalar(vals[j])
            else:
                # Last resort: map by index but avoid f0/f1 confusion by labeling as unknown_i
                for i, val in enumerate(vals):
                    agg[f"unknown_{i}"] = agg.get(f"unknown_{i}", 0.0) + _to_scalar(val)
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
    # Ensure we return also canonical features not used by the model (contrib 0)
    used_names = [k for k, _ in contrib]
    canonical = ['air_temp_k','process_temp_k','rot_speed_rpm','torque_nm','tool_wear_min','delta_temp_k','omega_rad_s','power_w','wear_pct','type']
    missing = [c for c in canonical if c not in used_names]
    for m in missing:
        contrib.append((m, 0.0))
    # Sort again preserving high-abs contributions first, but keep zeros grouped at the end
    contrib = sorted([c for c in contrib if c[1] != 0.0], key=lambda x: abs(x[1]), reverse=True) + [c for c in contrib if c[1] == 0.0]
    return contrib
