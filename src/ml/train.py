import os
import joblib
import pandas as pd
import json
import shutil
import numpy as np
import warnings
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from ..data.data_loader import load_dataset, normalize_columns, engineer_features
from ..data.preprocess import build_preprocessor, TARGET_COL, MULTILABEL_COLS

warnings.filterwarnings('ignore')

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
VERSIONS_DIR = os.path.join(MODELS_DIR, "versions")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VERSIONS_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def validate_training_data(df: pd.DataFrame, target_col: str = TARGET_COL) -> dict:
    """Valida datos de entrenamiento."""
    warnings_list = []
    info = {}
    
    if df is None or df.empty:
        return {'valid': False, 'warnings': ['Dataset vacío'], 'info': {}}
    
    # Valores faltantes
    missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()
    high_missing = {k: v for k, v in missing_pct.items() if v > 10}
    if high_missing:
        warnings_list.append(f"Columnas con >10% valores faltantes: {high_missing}")
    
    if target_col not in df.columns:
        return {'valid': False, 'warnings': [f"Columna objetivo '{target_col}' no existe"], 'info': {}}
    
    y = df[target_col].dropna()
    class_distribution = y.value_counts().to_dict()
    info['class_distribution'] = class_distribution
    
    if len(class_distribution) < 2:
        warnings_list.append("Menos de 2 clases en la columna objetivo")
        return {'valid': False, 'warnings': warnings_list, 'info': info}
    
    min_class = min(class_distribution.values())
    max_class = max(class_distribution.values())
    imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
    info['imbalance_ratio'] = imbalance_ratio
    
    if imbalance_ratio > 5:
        warnings_list.append(f"Alto desbalance de clases (ratio: {imbalance_ratio:.2f})")
    
    if len(df) < 100:
        warnings_list.append(f"Dataset muy pequeño ({len(df)} muestras, recomendado >100)")
    
    return {
        'valid': True,
        'warnings': warnings_list,
        'info': {
            'n_samples': len(df),
            'n_features': df.shape[1],
            'class_distribution': class_distribution,
            'imbalance_ratio': imbalance_ratio,
            'missing_values': high_missing
        }
    }


def get_class_weights_analysis(y: pd.Series) -> dict:
    """Analiza pesos de clase para mejor equilibrio."""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    weight_dict = {cls: w for cls, w in zip(classes, weights)}
    
    return {
        'weights': weight_dict,
        'imbalance_ratio': max(weights) / min(weights) if min(weights) > 0 else 1.0
    }


def save_model_version(model, metrics, version_stamp, model_type='binary'):
    """Guarda una versión del modelo con timestamp y metadata."""
    version_dir = os.path.join(VERSIONS_DIR, f"{model_type}_{version_stamp}")
    os.makedirs(version_dir, exist_ok=True)
    
    model_path = os.path.join(version_dir, f'{model_type}_model.joblib')
    joblib.dump(model, model_path)
    
    metrics_path = os.path.join(version_dir, f'{model_type}_metrics.joblib')
    joblib.dump(metrics, metrics_path)
    
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
    """Elimina versiones antiguas de modelos."""
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
        versions.sort(key=lambda x: x[1], reverse=True)
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
    """Entrena modelos binarios con validación robusta."""
    
    validation = validate_training_data(df, TARGET_COL)
    if not validation['valid']:
        raise ValueError(f"Datos inválidos: {validation['warnings']}")
    
    print("\n   [Validación de datos]")
    for warning in validation['info']:
        print(f"   - {warning}: {validation['info'].get(warning, 'N/A')}")
    
    X = df.drop(columns=[TARGET_COL])
    
    if 'product_id' in X.columns or 'uid' in X.columns:
        X = X.drop(columns=[c for c in ['product_id', 'uid'] if c in X.columns])
    
    y = df[TARGET_COL]
    preprocessor = build_preprocessor(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    class_analysis = get_class_weights_analysis(y_train)
    print(f"\n   [Análisis de desbalance]")
    print(f"   - Ratio de desbalance: {class_analysis['imbalance_ratio']:.2f}")
    print(f"   - Pesos de clase: {class_analysis['weights']}")

    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=300, 
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE, 
            class_weight='balanced',
            n_jobs=1,
            oob_score=True
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=RANDOM_STATE,
            validation_fraction=0.1,
            n_iter_no_change=20
        ),
        'logistic_regression': LogisticRegression(
            max_iter=2000, 
            class_weight='balanced', 
            random_state=RANDOM_STATE,
            solver='lbfgs'
        )
    }

    fitted = {}
    reports = {}
    aucs = {}
    cv_scores = {}
    
    print(f"\n   [Entrenamiento de modelos]")
    
    for name, clf in models.items():
        print(f"\n   - Entrenando {name}...")
        
        pipe = Pipeline(steps=[('pre', preprocessor), ('clf', clf)])
        pipe.fit(X_train, y_train)
        
        y_pred = pipe.predict(X_test)
        y_pred_proba = pipe.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        cv_scores[name] = {
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print(f"     - Test AUC: {cv_scores[name]['auc']:.4f}")
        print(f"     - Test Precision: {cv_scores[name]['precision']:.4f}")
        print(f"     - Test Recall: {cv_scores[name]['recall']:.4f}")
        print(f"     - Test F1: {cv_scores[name]['f1']:.4f}")
        
        reports[name] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        aucs[name] = auc
        reports[name]['test_metrics'] = {
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        fitted[name] = pipe

    # Selección por mayor AUC
    candidates = [(k, v) for k, v in aucs.items() if v is not None]
    if not candidates:
        raise ValueError("Ningún modelo logró AUC válido")
    
    best_name = sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]
    best_model = fitted[best_name]
    
    print(f"\n   [Selección de modelo]")
    print(f"   - Mejor modelo: {best_name}")
    print(f"   - AUC: {aucs[best_name]:.4f}")
    
    # Guardar métricas completas
    metrics_data = {
        'reports': reports,
        'aucs': aucs,
        'cv_scores': cv_scores,
        'best': best_name,
        'class_analysis': class_analysis
    }
    
    joblib.dump(metrics_data, os.path.join(MODELS_DIR, 'failure_binary_metrics.joblib'))
    
    return best_model, reports, aucs, best_name


def train_multilabel(df: pd.DataFrame):
    """Entrena modelos multilabel independientes por cada etiqueta.
    
    Con validaciones robustas y métnicas detalladas.
    """
    models = {}
    metrics = {}
    
    print(f"\n   [Entrenamiento multilabel]")
    
    for label in MULTILABEL_COLS:
        df_label = df.copy()
        
        # Validar que la columna exista
        if label not in df_label.columns:
            print(f"   - Ignorando {label}: no existe en el dataset")
            continue
        
        y = df_label[label].copy()
        if y.isna().all():
            print(f"   - Ignorando {label}: todos los valores son NaN")
            continue
        
        # Verificar suficientes clases y muestras
        non_na_mask = ~y.isna()
        y_non_na = y[non_na_mask]
        
        if y_non_na.nunique() < 2:
            print(f"   - Ignorando {label}: solo {y_non_na.nunique()} clase(s)")
            continue
        
        vc = y_non_na.value_counts()
        if vc.min() < 2:
            print(f"   - Ignorando {label}: clase menos poblada con {vc.min()} muestras")
            continue
        
        print(f"\n   - Entrenando {label}...")
        
        # Preparar features
        X = df_label.drop(columns=MULTILABEL_COLS + [TARGET_COL])
        
        # Excluir identificadores técnicos
        if 'product_id' in X.columns or 'uid' in X.columns:
            X = X.drop(columns=[c for c in ['product_id', 'uid'] if c in X.columns])
        
        y = df[label]
        pre = build_preprocessor(df.drop(columns=[label]))
        
        # Trabajar solo con filas no nulas
        X = X.loc[non_na_mask]
        y = y.loc[non_na_mask]
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=RANDOM_STATE
        )
        
        # Análisis de clase
        class_dist = y_train.value_counts().to_dict()
        class_weights = get_class_weights_analysis(y_train)
        print(f"     - Distribución: {class_dist}")
        
        # Entrenar modelo
        clf = RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        )
        
        pipe = Pipeline(steps=[('pre', pre), ('clf', clf)])
        pipe.fit(X_train, y_train)
        
        # Evaluar en test
        y_pred = pipe.predict(X_test)
        try:
            y_prob = pipe.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, y_prob)
        except Exception:
            test_auc = None
        
        test_f1 = f1_score(y_test, y_pred, zero_division=0)
        
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        print(f"     - Test AUC: {test_auc:.4f}" if test_auc else "     - Test AUC: N/A")
        print(f"     - Test F1: {test_f1:.4f}")
        
        models[label] = pipe
        metrics[label] = {
            'report': report,
            'auc': test_auc,
            'f1': test_f1,
            'class_distribution': class_dist,
            'class_weights': class_weights['weights']
        }

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
    
    # NOTA: NO se integran datos de feedback en el entrenamiento
    # El proyecto usa calibración post-predicción, NO reentrenamiento
    # Ver: streamlit_app.py - calibrated_probability() para el flujo correcto
    
    # Generar timestamp de versión
    version_stamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    print(f"\n2. Versión del modelo: {version_stamp}")
    
    # Configuración: umbral de reemplazo y retención de versiones
    MODEL_REPLACEMENT_DELTA = float(os.environ.get('AUC_REPLACEMENT_DELTA', 0.001))
    MODEL_RETENTION_KEEP = int(os.environ.get('MODEL_RETENTION_KEEP', 1))

    # Antes de entrenar, leer métricas actuales de producción para decidir reemplazo
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
    
    # Cargar métricas completas (incluye cv_scores)
    metrics_path = os.path.join(MODELS_DIR, 'failure_binary_metrics.joblib')
    metrics_data = joblib.load(metrics_path)
    metrics_data['version'] = version_stamp
    metrics_data['trained_samples'] = len(df)
    metrics_data['training_date'] = datetime.utcnow().isoformat()
    joblib.dump(metrics_data, metrics_path)
    
    # Reemplazar el modelo de producción solo si mejora
    new_auc = aucs.get(best_name)
    replace = False
    if current_auc is None:
        replace = True
    elif new_auc is None:
        replace = False
    else:
        replace = new_auc >= (current_auc + MODEL_REPLACEMENT_DELTA)

    if replace:
        print(f"\n4. Actualizando modelo de producción...")
        print(f"   - AUC anterior: {current_auc:.4f}" if current_auc else "   - AUC anterior: No existe")
        print(f"   - AUC nuevo: {new_auc:.4f}")
        print(f"   - Mejora: {(new_auc - current_auc):.4f}" if current_auc else "   - Es el primer modelo")
        joblib.dump(best_model, os.path.join(MODELS_DIR, 'failure_binary_model.joblib'))
    else:
        print(f"\n4. Modelo de producción mantenido (mejora insuficiente)")
        print(f"   - AUC anterior: {current_auc:.4f}")
        print(f"   - AUC nuevo: {new_auc:.4f}")
        print(f"   - Delta requerido: {MODEL_REPLACEMENT_DELTA:.4f}")    
    # Entrenar modelos multilabel
    print("\n5. Entrenando modelos multilabel...")
    multi_models, multi_metrics = train_multilabel(df)
    
    # Guardar versionado multilabel
    multi_metrics_path = os.path.join(MODELS_DIR, 'failure_multilabel_metrics.joblib')
    mm_data = joblib.load(multi_metrics_path)
    mm_data['version'] = version_stamp
    mm_data['trained_samples'] = len(df)
    mm_data['training_date'] = datetime.utcnow().isoformat()
    joblib.dump(mm_data, multi_metrics_path)
    
    print(f"\n   - {len(multi_models)} modelos multilabel entrenados exitosamente")
    
    # Guardar versión
    print("\n6. Guardando versiones...")
    save_model_version(best_model, metrics_data, version_stamp, 'binary')
    save_model_version(multi_models, mm_data, version_stamp, 'multilabel')
    
    # Limpiar versiones antiguas
    try:
        prune_old_versions(keep_last_n=MODEL_RETENTION_KEEP)
    except Exception as e:
        print(f"   Advertencia al limpiar versiones: {e}")
    
    # Mostrar resumen final
    print("\n7. Resumen final:")
    print(f"   - Versión: {version_stamp}")
    print(f"   - Muestras totales: {len(df)}")
    print(f"   - Modelo binario: {best_name} (AUC: {aucs[best_name]:.4f})")
    
    versions = get_version_history()
    if versions:
        print(f"   - Versiones guardadas: {len(versions)}")
    
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 60)
if __name__ == '__main__':
    main()
