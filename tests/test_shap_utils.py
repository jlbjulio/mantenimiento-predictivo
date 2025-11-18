import pytest
from src.ml.shap_utils import shap_for_instance
from src.data.data_loader import load_dataset


def test_shap_for_instance_runs():
    df = load_dataset()
    row = df.sample(1).iloc[0].to_dict()
    contrib = shap_for_instance({
        'air_temp_k': row['air_temp_k'],
        'process_temp_k': row['process_temp_k'],
        'rot_speed_rpm': row['rot_speed_rpm'],
        'torque_nm': row['torque_nm'],
        'tool_wear_min': row['tool_wear_min'],
        'type': row['type']
    })
    # Should return a non-empty list of (feature, contribution)
    assert isinstance(contrib, list)
    assert len(contrib) > 0
    assert isinstance(contrib[0], tuple)
    assert isinstance(contrib[0][0], str)
    assert isinstance(contrib[0][1], float)
