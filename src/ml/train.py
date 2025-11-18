import os
import joblib
import pandas as pd
import json
import shutil
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from ..data.data_loader import load_dataset, normalize_columns, engineer_features
from ..data.preprocess import build_preprocessor, TARGET_COL, MULTILABEL_COLS

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
VERSIONS_DIR = os.path.join(MODELS_DIR, "versions")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VERSIONS_DIR, exist_ok=True)


def save_model_version(model, metrics, version_stamp, model_type='binary'):
    """
    Guarda una versión del modelo con timestamp y metadata.
    
    Args:
        model: Modelo entrenado
        metrics: Métricas del modelo
        version_stamp: Timestamp de la versión
        model_type: Tipo de modelo ('binary' o 'multilabel')
    """
    version_dir = os.path.join(VERSIONS_DIR, f"{model_type}_{version_stamp}")
    os.makedirs(version_dir, exist_ok=True)
    
    # Guardar modelo
    model_path = os.path.join(version_dir, f'{model_type}_model.joblib')
    joblib.dump(model, model_path)
    
    # Guardar métricas
    metrics_path = os.path.join(version_dir, f'{model_type}_metrics.joblib')
    joblib.dump(metrics, metrics_path)
    
    # Crear metadata
    metadata = {
        'version': version_stamp,
        'model_type': model_type,
        'created_at': datetime.utcnow().isoformat(),
        'metrics_summary': {}
    }
    
    if model_type == 'binary':
        metadata['best_model'] = metrics.get('best', 'unknown')
        metadata['metrics_summary']['aucs'] = metrics.get('aucs', {})
        metadata['trained_samples'] = metrics.get('trained_samples', 0)
    else:
        metadata['trained_samples'] = metrics.get('trained_samples', 0)
        # Extraer AUCs de multilabel
        auc_summary = {}
        for label, data in metrics.items():
            if isinstance(data, dict) and 'auc' in data:
                auc_summary[label] = data['auc']
        metadata['metrics_summary']['label_aucs'] = auc_summary
    
    # Guardar metadata como JSON
    metadata_path = os.path.join(version_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Versión guardada en: {version_dir}")
    return version_dir


def get_version_history():
    """Obtiene el historial de versiones de modelos"""
    if not os.path.exists(VERSIONS_DIR):
        return []
    
    versions = []
    for item in os.listdir(VERSIONS_DIR):
        version_path = os.path.join(VERSIONS_DIR, item)
        if os.path.isdir(version_path):
            metadata_path = os.path.join(version_path, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    versions.append(metadata)
    
    # Ordenar por fecha de creación
    versions.sort(key=lambda x: x['created_at'], reverse=True)
    return versions


def prune_old_versions(keep_last_n: int = 5):
    """Remove older model version directories keeping only the latest `keep_last_n` based on metadata created_at."""
    try:
        versions = []
        for item in os.listdir(VERSIONS_DIR):
            version_path = os.path.join(VERSIONS_DIR, item)
            if os.path.isdir(version_path):
                metadata_path = os.path.join(version_path, 'metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        md = json.load(f)
                        versions.append((version_path, md.get('created_at', '')))
        # Sort by created_at (string ISO) descending
        versions.sort(key=lambda x: x[1], reverse=True)
        # Remove older beyond keep_last_n
        to_remove = versions[keep_last_n:]
        for path, _ in to_remove:
            try:
                shutil.rmtree(path)
                print(f"   Pruned version: {path}")
            except Exception:
                pass
    except Exception as e:
        print(f"   Error pruning versions: {e}")


def train_binary(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    # Excluir identificadores técnicos que no deben afectar al modelo
    if 'product_id' in X.columns or 'uid' in X.columns:
        X = X.drop(columns=[c for c in ['product_id', 'uid'] if c in X.columns])
    y = df[TARGET_COL]
    preprocessor = build_preprocessor(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    models = {
        'random_forest': RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, class_weight='balanced'),
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
        'logistic_regression': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
    }

    fitted = {}
    reports = {}
    aucs = {}
    for name, clf in models.items():
        from sklearn.pipeline import Pipeline
        pipe = Pipeline(steps=[('pre', preprocessor), ('clf', clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        reports[name] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        try:
            aucs[name] = roc_auc_score(y_test, y_prob)
        except Exception:
            aucs[name] = None
        fitted[name] = pipe

    # Selección por mayor AUC
    best_name = sorted([(k, v) for k, v in aucs.items() if v is not None], key=lambda x: x[1], reverse=True)[0][0]
    best_model = fitted[best_name]
    
    # Guardar modelo actual
    # Do not overwrite production model here; leaving to caller to conditionally replace
    # Save metrics for the version
    joblib.dump({'reports': reports, 'aucs': aucs, 'best': best_name}, os.path.join(MODELS_DIR, 'failure_binary_metrics.joblib'))
    
    return best_model, reports, aucs, best_name


def train_multilabel(df: pd.DataFrame):
    # Entrena un RandomForest por cada modo de fallo
    models = {}
    metrics = {}
    for label in MULTILABEL_COLS:
        # Work on a copy so we don't mutate original
        df_label = df.copy()
        # Make sure the label column exists
        if label not in df_label.columns:
            print(f"   Ignorando {label}: no existe en el dataset")
            continue
        # Drop rows where label is NaN
        y = df_label[label].copy()
        if y.isna().all():
            print(f"   Ignorando {label}: todos los valores son NaN")
            continue
        # If label has less than 2 classes after dropping NaN, skip
        non_na_mask = ~y.isna()
        y_non_na = y[non_na_mask]
        if y_non_na.nunique() < 2:
            print(f"   Ignorando {label}: solo {y_non_na.nunique()} clase(s) después de quitar NaNs")
            continue
        # Ensure we have at least 2 instances per class to stratify
        vc = y_non_na.value_counts()
        if vc.min() < 2:
            print(f"   Ignorando {label}: la clase menos poblada tiene {vc.min()} muestras (<2) para stratify")
            continue
            continue
        X = df_label.drop(columns=MULTILABEL_COLS + [TARGET_COL])
        # Excluir identificadores técnicos que no deben afectar al modelo
        if 'product_id' in X.columns or 'uid' in X.columns:
            X = X.drop(columns=[c for c in ['product_id', 'uid'] if c in X.columns])
        y = df[label]
        pre = build_preprocessor(df.drop(columns=[label]))
        # Ensure we use only non-NaN rows for this label
        X = X.loc[non_na_mask]
        y = y.loc[non_na_mask]
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        from sklearn.pipeline import Pipeline
        clf = RandomForestClassifier(n_estimators=250, random_state=42, class_weight='balanced')
        pipe = Pipeline(steps=[('pre', pre), ('clf', clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        try:
            y_prob = pipe.predict_proba(X_test)[:, 1]
        except Exception:
            y_prob = None
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
        models[label] = pipe
        metrics[label] = {'report': report, 'auc': auc}

    # Guardar modelos actuales
    joblib.dump(models, os.path.join(MODELS_DIR, 'failure_multilabel_models.joblib'))
    joblib.dump(metrics, os.path.join(MODELS_DIR, 'failure_multilabel_metrics.joblib'))
    
    return models, metrics


def main():
    print("=" * 60)
    print("ENTRENAMIENTO DE MODELOS DE PREDICCIÓN DE FALLOS")
    print("=" * 60)
    
    # Cargar dataset base
    print("\n1. Cargando dataset base...")
    df = load_dataset()
    print(f"   Dataset base: {len(df)} registros")
    
    # Integrar datos adicionales etiquetados si existen en data/additional/*.csv
    additional_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'additional')
    if os.path.isdir(additional_dir):
        extra_frames = []
        for fname in os.listdir(additional_dir):
            if not fname.lower().endswith('.csv'):
                continue
            fpath = os.path.join(additional_dir, fname)
            try:
                tmp = pd.read_csv(fpath)
                tmp = normalize_columns(tmp)
                # Requiere columna machine_failure para entrenamiento binario
                if 'machine_failure' in tmp.columns:
                    tmp = engineer_features(tmp)
                    extra_frames.append(tmp)
                    print(f"   Integrado: {fname} ({len(tmp)} registros)")
                else:
                    print(f"   Ignorando {fname}: no contiene etiqueta 'Machine failure'.")
            except Exception as e:
                print(f"   Error leyendo adicional {fname}: {e}")
        if extra_frames:
            df = pd.concat([df] + extra_frames, ignore_index=True)
            print(f"\n   Total después de integración: {len(df)} registros")
    
    # Generar timestamp de versión
    version_stamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    print(f"\n2. Versión del modelo: {version_stamp}")
    
    # Configuration: delta and retention
    MODEL_REPLACEMENT_DELTA = float(os.environ.get('AUC_REPLACEMENT_DELTA', 0.001))
    MODEL_RETENTION_KEEP = int(os.environ.get('MODEL_RETENTION_KEEP', 1))

    # Before training, read current production metrics to decide on replacement later
    current_production_metrics = None
    current_production_model_path = os.path.join(MODELS_DIR, 'failure_binary_model.joblib')
    current_production_metrics_path = os.path.join(MODELS_DIR, 'failure_binary_metrics.joblib')
    if os.path.exists(current_production_metrics_path):
        try:
            current_production_metrics = joblib.load(current_production_metrics_path)
        except Exception:
            current_production_metrics = None

    def get_current_production_auc(metrics_dict):
        if not metrics_dict:
            return None
        best = metrics_dict.get('best')
        au_dict = metrics_dict.get('aucs', {})
        if not best or best not in au_dict:
            return None
        return au_dict.get(best)

    current_auc = get_current_production_auc(current_production_metrics)

    # Entrenar modelo binario
    print("\n3. Entrenando modelos binarios...")
    best_model, reports, aucs, best_name = train_binary(df)
    
    # Guardar metadata en métricas
    metrics_path = os.path.join(MODELS_DIR, 'failure_binary_metrics.joblib')
    metrics_data = joblib.load(metrics_path)
    metrics_data['version'] = version_stamp
    metrics_data['trained_samples'] = len(df)
    joblib.dump(metrics_data, metrics_path)
    
    print(f'   Mejor modelo: {best_name}')
    print(f'   AUC: {aucs[best_name]:.4f}')
    
    # Guardar versión del modelo binario
    print("\n4. Guardando versión del modelo binario (siempre se guarda la versión)...")
    save_model_version(best_model, metrics_data, version_stamp, 'binary')

    # Replace production model only if AUC improves by at least delta (or first time)
    new_auc = aucs.get(best_name)
    replace = False
    if current_auc is None:
        # If there is no current model or AUC known, replace
        replace = True
    elif new_auc is None:
        replace = False
    else:
        # Replace only if improves by at least the configured delta
        replace = new_auc >= (current_auc + MODEL_REPLACEMENT_DELTA)

    if replace:
        print(f"\n5. Reemplazando modelo de producción: AUC actual {current_auc} -> nueva AUC {new_auc}")
        joblib.dump(best_model, os.path.join(MODELS_DIR, 'failure_binary_model.joblib'))
        joblib.dump({'reports': reports, 'aucs': aucs, 'best': best_name}, os.path.join(MODELS_DIR, 'failure_binary_metrics.joblib'))
    else:
        print(f"\n5. No se reemplaza el modelo de producción: AUC actual {current_auc} >= nueva AUC {new_auc}")
    
    # Entrenar modelos multilabel
    print("\n5. Entrenando modelos multilabel...")
    multi_models, multi_metrics = train_multilabel(df)
    
    # Añadir versionado multilabel
    multi_metrics_path = os.path.join(MODELS_DIR, 'failure_multilabel_metrics.joblib')
    mm_data = joblib.load(multi_metrics_path)
    mm_data['version'] = version_stamp
    mm_data['trained_samples'] = len(df)
    joblib.dump(mm_data, multi_metrics_path)
    
    print('   Modelos multilabel entrenados')
    
    # Guardar versión de modelos multilabel
    print("\n6. Guardando versión de modelos multilabel...")
    save_model_version(multi_models, mm_data, version_stamp, 'multilabel')
    # Prune older versions to conserve space if configured
    try:
        prune_old_versions(keep_last_n=MODEL_RETENTION_KEEP)
    except Exception:
        pass
    # Mostrar historial de versiones
    print("\n7. Historial de versiones:")
    versions = get_version_history()
    if versions:
        print(f"   Total de versiones: {len(versions)}")
        print("\n   Últimas 5 versiones:")
        for i, v in enumerate(versions[:5], 1):
            print(f"   {i}. {v['version']} - {v['model_type']} - {v.get('trained_samples', 'N/A')} muestras")
    
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"Versión: {version_stamp}")
    print(f"Muestras totales: {len(df)}")
    print(f"Modelo binario: {best_name} (AUC: {aucs[best_name]:.4f})")
    print("=" * 60)
if __name__ == '__main__':
    main()
