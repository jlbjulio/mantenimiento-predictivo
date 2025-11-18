import os
import tempfile
import pandas as pd
import numpy as np
from src.data.data_loader import augment_dataset
from src.ml.combine_feedback import upgrade_pred_log


def build_base_df(n=1000, frac_failure=0.05, seed=42):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        'air_temp_k': rng.normal(300, 2, n),
        'process_temp_k': rng.normal(310, 2, n),
        'rot_speed_rpm': rng.normal(1500, 100, n),
        'torque_nm': rng.normal(40, 2, n),
        'tool_wear_min': np.abs(rng.normal(20, 5, n)),
        'type': ['M'] * n,
        'machine_failure': rng.binomial(1, frac_failure, n)
    })
    return df


def test_augment_balance_mode_matches_target_ratio(tmp_path=None):
    df = build_base_df(n=1000, frac_failure=0.05)
    # Generate 1000 synthetic rows and target overall ratio to 0.1
    out = augment_dataset(df, n=1000, seed=1, mode='balance', target_failure_ratio=0.1)
    combined = pd.concat([df, out], ignore_index=True)
    frac = combined['machine_failure'].mean()
    # Should be close to 0.1 (allow some rounding tolerance)
    assert abs(frac - 0.1) < 0.02


def test_augment_targeted_feature_injection():
    df = build_base_df(n=1000, frac_failure=0.05)
    # Inject a targeted value (air_temp_k = 500) in 10% of rows
    out = augment_dataset(df, n=500, seed=2, mode='targeted', targeted_feature='air_temp_k', targeted_value=500.0, targeted_frac=0.1)
    count_targeted = (out['air_temp_k'] == 500.0).sum()
    assert abs(count_targeted - int(round(500 * 0.1))) <= 2


def test_upgrade_pred_log_normalizes(tmp_path):
    # create a small pred log with missing columns and weird formats
    f = tmp_path / "predicciones_old.csv"
    data = [
        ["2025-11-17 12:00:00+00:00", '300.0', '310.0', '1500', '40', '20', 'M', '0', '0.1234'],
        ["11/17/2025 13:00:00", '301.0', '311.0', '1501', '41', '21', 'M', '1', '0.9876']
    ]
    # old header without machine_failure, notes, feedback_timestamp
    header = ['timestamp','air_temp_k','process_temp_k','rot_speed_rpm','torque_nm','tool_wear_min','type','pred','prob']
    df = pd.DataFrame(data, columns=header)
    df.to_csv(f, index=False)
    # Run upgrade
    ok = upgrade_pred_log(str(f))
    assert ok
    df2 = pd.read_csv(str(f))
    # Should have the full header
    full_header = ['timestamp','air_temp_k','process_temp_k','rot_speed_rpm','torque_nm','tool_wear_min','type','pred','prob','Machine failure','feedback_timestamp']
    assert list(df2.columns) == full_header
    # timestamp should be ISO format without TZ
    assert 'T' in df2.loc[0, 'timestamp']
    # prob should be numeric in string, but convertible to float
    assert float(df2.loc[0, 'prob']) == float(df.loc[0, 'prob'])


def test_load_dataset_includes_additional(tmp_path):
    # create a temporary base csv and an additional csv, ensure load_dataset can merge them
    from src.data.data_loader import load_dataset
    base = tmp_path / 'ai4i2020.csv'
    df_base = build_base_df(n=50, frac_failure=0.05)
    df_base.to_csv(base, index=False)
    add_dir = tmp_path / 'additional'
    add_dir.mkdir()
    df_add = build_base_df(n=10, frac_failure=0.2)
    df_add.to_csv(add_dir / 'extra.csv', index=False)
    df_no = load_dataset(path=str(base), include_additional=False)
    df_yes = load_dataset(path=str(base), include_additional=True, additional_dir=str(add_dir))
    assert len(df_yes) == len(df_no) + 10
