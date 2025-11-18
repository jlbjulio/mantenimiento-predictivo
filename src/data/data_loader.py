import os
import pandas as pd
from typing import Tuple

DEFAULT_PATHS = [
    "data/ai4i2020.csv",
    "ai4i2020.csv",
    os.path.join(os.path.dirname(__file__), "..", "..", "ai4i2020.csv")
]


def find_dataset(path: str = None) -> str:
    if path and os.path.exists(path):
        return path
    for p in DEFAULT_PATHS:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Dataset ai4i2020.csv no encontrado en rutas conocidas.")


def load_raw(path: str = None) -> pd.DataFrame:
    file_path = find_dataset(path)
    df = pd.read_csv(file_path)
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    df = df.rename(columns={c: mapping.get(c, c) for c in df.columns})
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    if 'air_temp_k' in df.columns and 'process_temp_k' in df.columns:
        df['delta_temp_k'] = df['process_temp_k'] - df['air_temp_k']
    if 'rot_speed_rpm' in df.columns and 'torque_nm' in df.columns:
        df['omega_rad_s'] = df['rot_speed_rpm'] * 2 * 3.141592653589793 / 60.0
        df['power_w'] = df['torque_nm'] * df['omega_rad_s']
    if 'type' in df.columns and 'tool_wear_min' in df.columns:
        # Margen relativo de umbral de desgaste segun rango 200-240
        df['wear_pct'] = df['tool_wear_min'] / 240.0
    return df


def augment_dataset(df: pd.DataFrame, n: int = 1000, seed: int = 42, extremes_frac: float = 0.1,
                   mode: str = 'preserve', target_failure_ratio: float = None,
                   targeted_feature: str = None, targeted_value: float = None,
                   targeted_frac: float = 0.0) -> pd.DataFrame:
    """Genera N filas sintéticas basadas en df.

    - Mantiene la proporción de fallos de Machine failure (si se requiere).
    - una fracción `extremes_frac` se genera con valores fuera de rango para algunas columnas.
    - Devuelve un DataFrame con n filas generadas que pueden ser concatenadas al dataset original.
    """
    import numpy as np
    rng = np.random.RandomState(seed)
    # Basic guards
    if df is None or df.empty:
        return pd.DataFrame()
    cols_needed = ['air_temp_k','process_temp_k','rot_speed_rpm','torque_nm','tool_wear_min','type','machine_failure']
    for c in cols_needed[:-1]:
        if c not in df.columns:
            raise ValueError(f"Columna requerida no encontrada: {c}")
    df_out = []
    # get ranges
    ranges = {c: (df[c].min(), df[c].max()) for c in ['air_temp_k','process_temp_k','rot_speed_rpm','torque_nm','tool_wear_min']}
    # failure ratio if present
    if 'machine_failure' in df.columns:
        frac_failure = float(df['machine_failure'].mean())
    else:
        frac_failure = 0.01
    types = df['type'].dropna().unique().tolist() if 'type' in df.columns else ['L','M','H']
    for i in range(n):
        base = df.sample(n=1, replace=True).iloc[0]
        # start from base values
        air = base['air_temp_k']
        proc = base['process_temp_k']
        rpm = base['rot_speed_rpm']
        trq = base['torque_nm']
        wear = base['tool_wear_min']
        t = base['type'] if pd.notna(base.get('type')) else rng.choice(types)
        # small noise around base
        air = float(air + rng.normal(0, 2.0))
        proc = float(proc + rng.normal(0, 2.0))
        rpm = float(max(0, rpm + rng.normal(0, 50.0)))
        trq = float(max(0, trq + rng.normal(0, 2.0)))
        wear = float(max(0, wear + rng.normal(0, 5.0)))
        # extremes injection
        if rng.rand() <= extremes_frac:
            # choose one feature and push to extreme
            cext = rng.choice(['air_temp_k','process_temp_k','rot_speed_rpm','torque_nm','tool_wear_min'])
            lo, hi = ranges[cext]
            if rng.rand() < 0.5:
                val = lo - rng.uniform(1.0, 200.0)
            else:
                val = hi + rng.uniform(1.0, 200.0)
            if cext == 'air_temp_k':
                air = float(val)
            elif cext == 'process_temp_k':
                proc = float(val)
            elif cext == 'rot_speed_rpm':
                rpm = float(max(0, val))
            elif cext == 'torque_nm':
                trq = float(max(0, val))
            elif cext == 'tool_wear_min':
                wear = float(max(0, val))
        # Decide failure label based on mode
        failure = 0
        if mode == 'preserve':
            failure = int(rng.rand() < frac_failure)
        elif mode == 'balance' and (target_failure_ratio is not None):
            # We'll compute required positives later in a simple pass — mark as placeholder for now
            failure = int(rng.rand() < frac_failure)
        elif mode == 'targeted':
            failure = int(rng.rand() < frac_failure)
        else:
            failure = int(rng.rand() < frac_failure)
        df_out.append({'air_temp_k': air, 'process_temp_k': proc, 'rot_speed_rpm': rpm, 'torque_nm': trq, 'tool_wear_min': wear, 'type': t, 'machine_failure': failure})
    out_df = pd.DataFrame(df_out)
    # engineer features and normalize columns to match original
    out_df = normalize_columns(out_df) if 'Product ID' in out_df.columns else out_df
    out_df = engineer_features(out_df)
    # Post-processing for special modes
    if mode == 'balance' and (target_failure_ratio is not None):
        # Calculate current counts and required positives to achieve target ratio
        existing_pos = int(df['machine_failure'].sum()) if 'machine_failure' in df.columns else 0
        existing_total = len(df)
        desired_total = existing_total + len(out_df)
        desired_pos = int(round(target_failure_ratio * desired_total))
        # Number of positives to add
        add_positives = max(0, desired_pos - existing_pos)
        add_positives = min(add_positives, len(out_df))
        # Force a subset of rows in the synthetic set to be positives to reach target
        if add_positives > 0:
            inds = rng.choice(out_df.index, size=add_positives, replace=False)
            out_df.loc[inds, 'machine_failure'] = 1
        # Ensure the rest are 0
        out_df['machine_failure'] = out_df['machine_failure'].fillna(0).astype(int)

    if mode == 'targeted' and targeted_feature and targeted_value is not None and targeted_frac > 0:
        count_targeted = int(round(len(out_df) * targeted_frac))
        if count_targeted > 0:
            inds = rng.choice(out_df.index, size=count_targeted, replace=False)
            out_df.loc[inds, targeted_feature] = float(targeted_value)
            # Recompute features that depend on targeted feature if needed
            if targeted_feature in ['air_temp_k', 'process_temp_k']:
                out_df['delta_temp_k'] = out_df['process_temp_k'] - out_df['air_temp_k']
            if targeted_feature in ['rot_speed_rpm', 'torque_nm']:
                out_df['omega_rad_s'] = out_df['rot_speed_rpm'] * 2 * 3.141592653589793 / 60.0
                out_df['power_w'] = out_df['torque_nm'] * out_df['omega_rad_s']
    return out_df


def load_dataset(path: str = None, include_additional: bool = False, additional_dir: str = None) -> pd.DataFrame:
    """Load base dataset optionally merging CSVs from data/additional.

    Args:
        path: optional path to base CSV
        include_additional: whether to include files in data/additional
        additional_dir: optional directory for extra CSVs (defaults to data/additional)
    """
    df = load_raw(path)
    df = normalize_columns(df)
    df = engineer_features(df)

    if include_additional:
        if additional_dir is None:
            additional_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'additional')
        if os.path.isdir(additional_dir):
            extras = []
            for fname in os.listdir(additional_dir):
                if not fname.lower().endswith('.csv'):
                    continue
                fpath = os.path.join(additional_dir, fname)
                try:
                    tmp = pd.read_csv(fpath)
                    tmp = normalize_columns(tmp)
                    if 'machine_failure' in tmp.columns:
                        tmp = engineer_features(tmp)
                        extras.append(tmp)
                except Exception:
                    continue
            if extras:
                df = pd.concat([df] + extras, ignore_index=True)
    return df

if __name__ == "__main__":
    data = load_dataset()
    print(data.head())
