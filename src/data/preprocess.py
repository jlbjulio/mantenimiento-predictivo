import pandas as pd
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

TARGET_COL = 'machine_failure'
MULTILABEL_COLS = ['twf','hdf','pwf','osf','rnf']


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    CANONICAL_NUMERIC = [
        'air_temp_k', 'process_temp_k', 'rot_speed_rpm', 'torque_nm', 'tool_wear_min',
        'delta_temp_k', 'omega_rad_s', 'power_w', 'wear_pct'
    ]
    CANONICAL_CAT = ['type']
    
    numeric_cols = [c for c in CANONICAL_NUMERIC if c in df.columns]
    cat_cols = [c for c in CANONICAL_CAT if c in df.columns]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, cat_cols)
        ])
    
    return preprocessor


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def get_multilabel_targets(df: pd.DataFrame) -> pd.DataFrame:
    return df[MULTILABEL_COLS]


def split_multilabel(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=MULTILABEL_COLS)
    Y = get_multilabel_targets(df)
    return train_test_split(X, Y, test_size=test_size, random_state=random_state, stratify=Y['twf'])

