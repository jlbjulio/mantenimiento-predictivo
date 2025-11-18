import pytest
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ml.recommendation import generate_recommendations

def test_wear_rule():
    inst = {'tool_wear_min': 220, 'rot_speed_rpm':1500, 'air_temp_k':300, 'process_temp_k':309, 'torque_nm':40, 'type':'L'}
    recs = generate_recommendations(inst, failure_prob=0.2)
    assert any('reemplazo' in r['accion'].lower() for r in recs)

def test_power_rule_low():
    inst = {'tool_wear_min': 50, 'rot_speed_rpm':1000, 'air_temp_k':300, 'process_temp_k':309, 'torque_nm':5, 'type':'L'}
    recs = generate_recommendations(inst, failure_prob=0.1)
    assert any('potencia' in r['justificacion'].lower() for r in recs)

def test_operate_normally():
    inst = {'tool_wear_min': 10, 'rot_speed_rpm':2000, 'air_temp_k':300, 'process_temp_k':310, 'torque_nm':40, 'type':'L'}
    recs = generate_recommendations(inst, failure_prob=0.1)
    assert any('operar normalmente' in r['accion'].lower() for r in recs)
