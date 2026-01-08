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
    # Mapeo de nombres originales a normalizados
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
    # Calcular delta de temperatura
    if 'air_temp_k' in df.columns and 'process_temp_k' in df.columns:
        df['delta_temp_k'] = df['process_temp_k'] - df['air_temp_k']
    
    # Calcular velocidad angular y potencia
    if 'rot_speed_rpm' in df.columns and 'torque_nm' in df.columns:
        df['omega_rad_s'] = df['rot_speed_rpm'] * 2 * 3.141592653589793 / 60.0
        df['power_w'] = df['torque_nm'] * df['omega_rad_s']
    
    # Calcular porcentaje de desgaste
    if 'type' in df.columns and 'tool_wear_min' in df.columns:
        df['wear_pct'] = df['tool_wear_min'] / 240.0
    
    return df


def augment_dataset(df: pd.DataFrame, n: int = 1000, seed: int = 42, extremes_frac: float = 0.1,
                   mode: str = 'preserve', target_failure_ratio: float = None,
                   targeted_feature: str = None, targeted_value: float = None,
                   targeted_frac: float = 0.0) -> pd.DataFrame:
    """Genera N filas sintéticas basadas en df.

    Nota: utilidad de aumento sintético pensada para experimentos/tests; la UI actual no la usa.

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
    # obtener rangos
    ranges = {c: (df[c].min(), df[c].max()) for c in ['air_temp_k','process_temp_k','rot_speed_rpm','torque_nm','tool_wear_min']}
    # razón de fallos si existe
    if 'machine_failure' in df.columns:
        frac_failure = float(df['machine_failure'].mean())
    else:
        frac_failure = 0.01
    types = df['type'].dropna().unique().tolist() if 'type' in df.columns else ['L','M','H']
    for i in range(n):
        base = df.sample(n=1, replace=True).iloc[0]
        # Partir de valores base
        air = base['air_temp_k']
        proc = base['process_temp_k']
        rpm = base['rot_speed_rpm']
        trq = base['torque_nm']
        wear = base['tool_wear_min']
        t = base['type'] if pd.notna(base.get('type')) else rng.choice(types)
        # Añadir ruido pequeño alrededor del valor base
        air = float(air + rng.normal(0, 2.0))
        proc = float(proc + rng.normal(0, 2.0))
        rpm = float(max(0, rpm + rng.normal(0, 50.0)))
        trq = float(max(0, trq + rng.normal(0, 2.0)))
        wear = float(max(0, wear + rng.normal(0, 5.0)))
        # Inyectar valores extremos
        if rng.rand() <= extremes_frac:
            # Elegir una característica y llevarla a un extremo
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
        # Decidir la etiqueta de fallo según el modo
        failure = 0
        if mode == 'preserve':
            failure = int(rng.rand() < frac_failure)
        elif mode == 'balance' and (target_failure_ratio is not None):
            # Los positivos necesarios se calculan después; por ahora es un marcador
            failure = int(rng.rand() < frac_failure)
        elif mode == 'targeted':
            failure = int(rng.rand() < frac_failure)
        else:
            failure = int(rng.rand() < frac_failure)
        df_out.append({'air_temp_k': air, 'process_temp_k': proc, 'rot_speed_rpm': rpm, 'torque_nm': trq, 'tool_wear_min': wear, 'type': t, 'machine_failure': failure})
    out_df = pd.DataFrame(df_out)
    # Calcular features derivadas y normalizar columnas al esquema original
    out_df = normalize_columns(out_df) if 'Product ID' in out_df.columns else out_df
    out_df = engineer_features(out_df)
    # Posprocesamiento para modos especiales
    if mode == 'balance' and (target_failure_ratio is not None):
        # Calcular conteo actual y positivos requeridos para llegar a la razón objetivo
        existing_pos = int(df['machine_failure'].sum()) if 'machine_failure' in df.columns else 0
        existing_total = len(df)
        desired_total = existing_total + len(out_df)
        desired_pos = int(round(target_failure_ratio * desired_total))
        # Número de positivos a añadir
        add_positives = max(0, desired_pos - existing_pos)
        add_positives = min(add_positives, len(out_df))
        # Forzar que un subconjunto de filas sintéticas sea positivo para llegar al objetivo
        if add_positives > 0:
            inds = rng.choice(out_df.index, size=add_positives, replace=False)
            out_df.loc[inds, 'machine_failure'] = 1
        # Asegurar que el resto queden en 0
        out_df['machine_failure'] = out_df['machine_failure'].fillna(0).astype(int)

    if mode == 'targeted' and targeted_feature and targeted_value is not None and targeted_frac > 0:
        count_targeted = int(round(len(out_df) * targeted_frac))
        if count_targeted > 0:
            inds = rng.choice(out_df.index, size=count_targeted, replace=False)
            out_df.loc[inds, targeted_feature] = float(targeted_value)
            # Recalcular las features dependientes si aplica
            if targeted_feature in ['air_temp_k', 'process_temp_k']:
                out_df['delta_temp_k'] = out_df['process_temp_k'] - out_df['air_temp_k']
            if targeted_feature in ['rot_speed_rpm', 'torque_nm']:
                out_df['omega_rad_s'] = out_df['rot_speed_rpm'] * 2 * 3.141592653589793 / 60.0
                out_df['power_w'] = out_df['torque_nm'] * out_df['omega_rad_s']
    return out_df


def load_dataset(path: str = None, include_additional: bool = False, additional_dir: str = None) -> pd.DataFrame:
    """Cargar dataset base, opcionalmente combinando CSVs desde data/additional.

    Args:
        path: ruta opcional al CSV base
        include_additional: si incluir archivos en data/additional
        additional_dir: directorio opcional para CSVs extra (por defecto data/additional)
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
