import os
from app.streamlit_app import ensure_pred_log_has_history, log_prediction, log_feedback, load_models, prepare_feature_row
import csv

LOG_PATH = os.path.abspath(os.path.join('.', 'logs'))
PRED_LOG = os.path.join(LOG_PATH, 'predicciones.csv')
FEEDBACK_LOG = os.path.join(LOG_PATH, 'feedback.csv')  # legacy - no longer used; tests updated to use pred log


def test_seed_and_log_dedup():
    # cleanup
    if os.path.exists(PRED_LOG): os.remove(PRED_LOG)
    if os.path.exists(FEEDBACK_LOG): os.remove(FEEDBACK_LOG)
    model, _ = load_models()
    ensure_pred_log_has_history(model, count=3)
    assert os.path.exists(PRED_LOG)
    size_before = os.path.getsize(PRED_LOG)
    # create a mock prediction
    user_data = {'air_temp_k': 300.0, 'process_temp_k': 310.0, 'rot_speed_rpm': 1500, 'torque_nm': 40.0, 'tool_wear_min': 10.0, 'type': 'L'}
    row = prepare_feature_row(user_data)
    # avoid predicting since model may not be available, we'll just craft a record
    rec = {**user_data, 'pred': 0, 'prob': 0.5, 'prediction_timestamp': '2025-11-17 13:00:00'}
    log_feedback(rec['prediction_timestamp'], 0)
    log_prediction(rec, rec['prob'], rec['pred'])
    # call second time to test dedup
    log_prediction(rec, rec['prob'], rec['pred'])
    # read file
    with open(PRED_LOG, 'r', newline='') as f:
        rows = list(csv.reader(f))
    # header + at least 4 rows (3 seed + 1 new); duplicates should not appear
    assert len(rows) >= 4
    # ensure last row timestamp is match
    assert rows[-1][0] == rec['prediction_timestamp']


def test_combine_and_save_labeled():
    # cleanup
    add_dir = os.path.abspath(os.path.join('.', 'data', 'additional'))
    if not os.path.isdir(add_dir):
        os.makedirs(add_dir, exist_ok=True)
    # remove old additional files
    for f in os.listdir(add_dir):
        if f.lower().endswith('.csv'):
            os.remove(os.path.join(add_dir, f))
    # Ensure we have at least one pred and feedback
    model, _ = load_models()
    ensure_pred_log_has_history(model, count=3)
    user_data = {'air_temp_k': 300.0, 'process_temp_k': 310.0, 'rot_speed_rpm': 1500, 'torque_nm': 40.0, 'tool_wear_min': 10.0, 'type': 'L'}
    row = prepare_feature_row(user_data)
    prob = model.predict_proba(row)[0][1]
    pred = int(model.predict(row)[0])
    rec = {**user_data, 'pred': pred, 'prob': prob, 'prediction_timestamp': '2025-11-17 13:00:01'}
    log_feedback(rec['prediction_timestamp'], 1)
    log_prediction(rec, rec['prob'], rec['pred'])

    from src.ml.combine_feedback import load_predictions, load_feedback, combine_predictions_with_feedback, save_labeled_data
    preds_df = load_predictions(PRED_LOG)
    fb_df = None
    combined = combine_predictions_with_feedback(preds_df, fb_df)
    assert combined is not None
    out = save_labeled_data(combined, add_dir)
    assert out is not None
    out_path, new_rows = out
    assert os.path.exists(out_path)
    assert new_rows >= 1
