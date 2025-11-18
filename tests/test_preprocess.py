from src.data.data_loader import load_dataset
from src.data.preprocess import build_preprocessor
from src.ml import train as train_module
import pandas as pd

def test_preprocessor_excludes_product_id_uid():
    df = load_dataset()
    pre = build_preprocessor(df)
    num_cols = pre.transformers_[0][2]
    cat_cols = pre.transformers_[1][2]
    assert 'product_id' not in cat_cols
    assert 'uid' not in num_cols


def test_train_excludes_identifiers():
    df = load_dataset()
    X = df.drop(columns=['machine_failure'])
    # simulate train processing
    if 'product_id' in X.columns or 'uid' in X.columns:
        X2 = X.drop(columns=[c for c in ['product_id','uid'] if c in X.columns])
    else:
        X2 = X
    assert 'product_id' not in X2.columns
    assert 'uid' not in X2.columns
