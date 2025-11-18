import os
import joblib
import shutil
import tempfile
import pandas as pd
import numpy as np
import json

from src.ml import train as train_module
from src.ml.train import train_multilabel
from src.ml.train import prune_old_versions


def backup_and_restore(path):
    # Helper to backup a file and restore after use
    if not os.path.exists(path):
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.close()
    shutil.copy2(path, tmp.name)
    return tmp.name


def restore_backup(original_path, backup_path):
    if backup_path is None:
        # No original existed
        if os.path.exists(original_path):
            os.remove(original_path)
    else:
        shutil.copy2(backup_path, original_path)
        os.remove(backup_path)


def test_binary_model_not_replaced_if_auc_higher():
    models_dir = os.path.join(os.path.dirname(train_module.__file__), '..', '..', 'models')
    models_dir = os.path.abspath(models_dir)
    os.makedirs(models_dir, exist_ok=True)

    prod_model_path = os.path.join(models_dir, 'failure_binary_model.joblib')
    prod_metrics_path = os.path.join(models_dir, 'failure_binary_metrics.joblib')

    # Backup existing production model & metrics
    model_bkp = backup_and_restore(prod_model_path)
    metrics_bkp = backup_and_restore(prod_metrics_path)

    try:
        # Write a fake high-AUC metrics so our new model won't beat it
        fake_metrics = {'best': 'random_forest', 'aucs': {'random_forest': 0.9999}}
        joblib.dump(fake_metrics, prod_metrics_path)
        joblib.dump({'dummy': 'model'}, prod_model_path)

        # Run training (uses data in repo) and ensure model file didn't change
        # Record original bytes
        with open(prod_model_path, 'rb') as f:
            orig_bytes = f.read()

        train_module.main()

        # Check model file still equals original
        with open(prod_model_path, 'rb') as f:
            new_bytes = f.read()
        assert orig_bytes == new_bytes, "Production model was replaced despite higher existing AUC"
    finally:
        restore_backup(prod_model_path, model_bkp)
        restore_backup(prod_metrics_path, metrics_bkp)


def test_multilabel_skips_labels_with_nan_or_single_class():
    # Build small demo dataset
    n = 100
    df = pd.DataFrame({
        'air_temp_k': np.random.normal(300, 2, n),
        'process_temp_k': np.random.normal(310, 2, n),
        'rot_speed_rpm': np.random.normal(1500, 100, n),
        'torque_nm': np.random.normal(40, 2, n),
        'tool_wear_min': np.abs(np.random.normal(20, 5, n)),
        'type': ['M'] * n,
        'machine_failure': np.random.binomial(1, 0.05, n),
        'twf': [np.nan] * n,  # all NaN -> should be skipped
        'hdf': [0] * n,  # single class -> should be skipped
        'pwf': np.random.binomial(1, 0.1, n),  # valid label
        'osf': np.random.binomial(1, 0.02, n),
        'rnf': np.random.binomial(1, 0.01, n),
    })

    models, metrics = train_multilabel(df)

    # twf and hdf should not be in models (skipped)
    assert 'twf' not in models
    assert 'hdf' not in models
    # pwf should be trained normally
    assert 'pwf' in models
    assert 'pwf' in metrics


def test_prune_old_versions_keeps_latest(tmp_path):
    versions_dir = os.path.join(os.path.dirname(train_module.__file__), '..', '..', 'models', 'versions')
    versions_dir = os.path.abspath(versions_dir)
    os.makedirs(versions_dir, exist_ok=True)
    # Create 8 fake version dirs with metadata
    for i in range(8):
        dname = os.path.join(versions_dir, f"binary_20250101_00000{i}")
        os.makedirs(dname, exist_ok=True)
        md = {'version': f"20250101_00000{i}", 'created_at': f"2025-01-01T00:00:0{i}Z", 'model_type': 'binary'}
        with open(os.path.join(dname, 'metadata.json'), 'w') as f:
            json.dump(md, f)
    # Run prune keeping last 3
    prune_old_versions(keep_last_n=3)
    # Count remaining
    remaining = [d for d in os.listdir(versions_dir) if os.path.isdir(os.path.join(versions_dir, d))]
    assert len(remaining) == 3
