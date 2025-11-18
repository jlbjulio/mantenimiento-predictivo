"""
Script para combinar predicciones.csv con feedback.csv y generar dataset etiquetado.
Este dataset puede ser usado para reentrenar el modelo con datos reales del usuario.
"""

import os
import pandas as pd
from datetime import datetime


def load_predictions(pred_path):
    """Carga el archivo de predicciones"""
    if not os.path.exists(pred_path):
        print(f"No existe el archivo de predicciones: {pred_path}")
        return None
    
    # Normalize file first
    try:
        upgrade_pred_log(pred_path)
    except Exception:
        pass
    try:
        df = pd.read_csv(pred_path, engine='python', on_bad_lines='warn')
    except TypeError:
        # fallback for older pandas versions
        df = pd.read_csv(pred_path, engine='python')
    # tolerate mixed timestamp formats and remove timezone info if present
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    try:
        df['timestamp'] = df['timestamp'].dt.tz_convert(None)
    except Exception:
        try:
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        except Exception:
            pass
    # Support both 'machine_failure' and 'Machine failure' column names in logs
    if 'Machine failure' in df.columns and 'machine_failure' not in df.columns:
        df['machine_failure'] = df['Machine failure']
    if 'machine_failure' in df.columns:
        df['machine_failure'] = pd.to_numeric(df['machine_failure'], errors='coerce')
    # Ensure at least expected columns exist
    expected = ['timestamp','air_temp_k','process_temp_k','rot_speed_rpm','torque_nm','tool_wear_min','type','pred','prob','Machine failure','feedback_timestamp']
    for c in expected:
        if c not in df.columns:
            df[c] = None
    return df


def upgrade_pred_log(pred_path):
    """Normalize and upgrade an existing predicciones.csv file: ensure headers, types and timestamps are consistent.
    This function rewrites the CSV in place with a canonical set of columns and formats.
    """
    if not os.path.exists(pred_path):
        return False
    full_header = ['timestamp','air_temp_k','process_temp_k','rot_speed_rpm','torque_nm','tool_wear_min','type','pred','prob','Machine failure','feedback_timestamp']
    try:
        df = pd.read_csv(pred_path, engine='python', on_bad_lines='warn')
    except Exception:
        df = pd.read_csv(pred_path, engine='python')
    # Ensure columns exist
    for c in full_header:
        if c not in df.columns:
            df[c] = None
    # Support both log column names; map 'Machine failure' to 'machine_failure' if present
    if 'Machine failure' in df.columns and 'machine_failure' not in df.columns:
        df['machine_failure'] = df['Machine failure']
    # Normalize timestamp columns
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['timestamp'] = df['timestamp'].dt.tz_convert(None)
    except Exception:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        except Exception:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    # Normalize numeric columns
    df['prob'] = pd.to_numeric(df['prob'], errors='coerce')
    try:
        df['pred'] = pd.to_numeric(df['pred'], errors='coerce')
    except Exception:
        df['pred'] = df['pred']
    df['machine_failure'] = pd.to_numeric(df['machine_failure'], errors='coerce')
    # Normalize feedback timestamp
    if 'feedback_timestamp' in df.columns:
        df['feedback_timestamp'] = pd.to_datetime(df['feedback_timestamp'], errors='coerce')
    # Write back with canonical header, casting to strings where needed
    # Convert timestamp columns back to ISO strings
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    df['feedback_timestamp'] = df['feedback_timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    # Fill NaNs with empty strings for consistent CSV
    df = df.fillna('')
    df.to_csv(pred_path, index=False, columns=full_header)
    return True


def load_feedback(feedback_path):
    """Backward-compatible: keep loader but it's no longer used; return None."""
    if not os.path.exists(feedback_path):
        return None
    try:
        df = pd.read_csv(feedback_path)
        return df
    except Exception:
        return None


def combine_predictions_with_feedback(predictions_df, feedback_df, time_window_minutes=5):
    """
    Combina predicciones con feedback basándose en timestamps cercanos.
    
    Args:
        predictions_df: DataFrame con predicciones
        feedback_df: DataFrame con feedback del usuario
        time_window_minutes: Ventana de tiempo en minutos para asociar feedback con predicciones
    
    Returns:
        DataFrame combinado con etiquetas reales
    """
    if predictions_df is None:
        print("Error: No hay datos de predicciones para combinar")
        return None
    
    # Support two modes:
    # - If feedback_df provided, combine predictions with feedback by timestamp
    # - Else if predictions_df contains 'machine_failure', extract those rows
    if feedback_df is None:
        if 'machine_failure' not in predictions_df.columns:
            print('No hay feedback embebido en predicciones (busque machine_failure en predicciones)')
            return None
        # Map predictions columns to training columns and return rows with machine_failure in [0,1]
        labeled = predictions_df[predictions_df['machine_failure'].isin([0, 1])].copy()
        if labeled.empty:
            print('No se encontraron predicciones etiquetadas con machine_failure')
            return None
        combined = []
        for idx, row in labeled.iterrows():
            combined_row = {
                'UDI': idx + 1,  # Sequential ID for new rows
                'Product ID': f"{row['type']}99999",  # Placeholder product ID
                'Type': row['type'],
                'Air temperature [K]': row['air_temp_k'],
                'Process temperature [K]': row['process_temp_k'],
                'Rotational speed [rpm]': row['rot_speed_rpm'],
                'Torque [Nm]': row['torque_nm'],
                'Tool wear [min]': row['tool_wear_min'],
                'Machine failure': int(row['machine_failure']),
                'TWF': 0,  # Default: no tool wear failure
                'HDF': 0,  # Default: no heat dissipation failure
                'PWF': 0,  # Default: no power failure
                'OSF': 0,  # Default: no overstrain failure
                'RNF': 0,  # Default: no random failure
                '_prediction_prob': row.get('prob') if 'prob' in row else None,
                '_prediction_timestamp': row['timestamp'] if 'timestamp' in row else None,
                '_feedback_timestamp': row.get('feedback_timestamp', '')
            }
            combined.append(combined_row)
        combined_df = pd.DataFrame(combined)
        print(f"Se combinaron {len(combined_df)} registros de predicciones con machine_failure embebido")
        return combined_df
    
    # Filtrar feedback válido (actual_failure = 0 o 1, -1 son solo notas)
    feedback_valid = feedback_df[feedback_df['actual_failure'].isin([0, 1])].copy()
    if len(feedback_valid) == 0:
        print("No hay feedback válido con etiquetas de fallo")
        return None
    # Combinar basándose en el timestamp más cercano
    combined = []
    for _, fb_row in feedback_valid.iterrows():
        fb_time = fb_row['timestamp']
        # Buscar predicciones dentro de la ventana de tiempo
        time_diff = abs(predictions_df['timestamp'] - fb_time)
        within_window = time_diff <= pd.Timedelta(minutes=time_window_minutes)
        if within_window.any():
            # Tomar la predicción más cercana
            closest_idx = time_diff.idxmin()
            pred_row = predictions_df.loc[closest_idx]
            # Crear registro combinado
            combined_row = {
                'UDI': closest_idx + 1,  # Sequential ID
                'Product ID': f"{pred_row['type']}99999",  # Placeholder product ID
                'Type': pred_row['type'],
                'Air temperature [K]': pred_row['air_temp_k'],
                'Process temperature [K]': pred_row['process_temp_k'],
                'Rotational speed [rpm]': pred_row['rot_speed_rpm'],
                'Torque [Nm]': pred_row['torque_nm'],
                'Tool wear [min]': pred_row['tool_wear_min'],
                'Machine failure': fb_row['actual_failure'],
                'TWF': 0,  # Default: no tool wear failure
                'HDF': 0,  # Default: no heat dissipation failure
                'PWF': 0,  # Default: no power failure
                'OSF': 0,  # Default: no overstrain failure
                'RNF': 0,  # Default: no random failure
                '_prediction_prob': pred_row['prob'] if 'prob' in pred_row else None,
                '_prediction_timestamp': pred_row['timestamp'],
                '_feedback_timestamp': fb_row['feedback_timestamp']
            }
            combined.append(combined_row)
    
    if len(combined) == 0:
        print("No se encontraron coincidencias entre predicciones y feedback")
        return None
    
    combined_df = pd.DataFrame(combined)
    print(f"Se combinaron {len(combined_df)} registros de predicciones con feedback")
    print(f"Distribución de fallos: {combined_df['Machine failure'].value_counts().to_dict()}")
    
    return combined_df


def save_labeled_data(combined_df, output_dir):
    """Guarda el dataset etiquetado para reentrenamiento"""
    if combined_df is None or len(combined_df) == 0:
        print("No hay datos para guardar")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Archivo con timestamp
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'feedback_labeled_{timestamp}.csv')

    # Guardar solo las columnas relevantes para entrenamiento (columnas del dataset original)
    training_cols = [
        'UDI', 'Product ID', 'Type',
        'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
        'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'
    ]

    # Dedupe: avoid writing rows that already exist in previously saved files
    # Use internal _prediction_timestamp for deduplication but don't save it in final CSV
    existing_timestamps = set()
    for fname in os.listdir(output_dir):
        if not fname.lower().endswith('.csv') or not fname.startswith('feedback_full_'):
            continue
        fpath = os.path.join(output_dir, fname)
        try:
            tmp = pd.read_csv(fpath)
            if '_prediction_timestamp' in tmp.columns:
                existing_timestamps.update(tmp['_prediction_timestamp'].astype(str).tolist())
        except Exception:
            pass

    # Only keep rows with _prediction_timestamp not in existing_timestamps
    if '_prediction_timestamp' in combined_df.columns:
        combined_df['_prediction_timestamp'] = combined_df['_prediction_timestamp'].astype(str)
        new_rows = combined_df[~combined_df['_prediction_timestamp'].isin(existing_timestamps)].copy()
    else:
        new_rows = combined_df.copy()

    if len(new_rows) == 0:
        print('No hay nuevos registros etiquetados para guardar en output_dir')
        return None

    new_rows_to_save = new_rows[training_cols].copy()
    new_rows_to_save.to_csv(output_path, index=False)
    print(f"Dataset etiquetado guardado en: {output_path} ({len(new_rows_to_save)} nuevos)")

    # También guardar versión completa con metadata
    full_output_path = os.path.join(output_dir, f'feedback_full_{timestamp}.csv')
    new_rows.to_csv(full_output_path, index=False)
    print(f"Dataset completo guardado en: {full_output_path}")

    return output_path, len(new_rows_to_save)


def main():
    """Función principal"""
    # Rutas de archivos
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    logs_dir = os.path.join(base_dir, 'logs')
    pred_path = os.path.join(logs_dir, 'predicciones.csv')
    feedback_path = os.path.join(logs_dir, 'feedback.csv')
    output_dir = os.path.join(base_dir, 'data', 'additional')
    
    print("=" * 60)
    print("COMBINACIÓN DE PREDICCIONES CON FEEDBACK")
    print("=" * 60)
    
    # Cargar datos
    print("\n1. Cargando predicciones...")
    predictions_df = load_predictions(pred_path)
    
    print("\n2. Intentando detectar feedback embebido en predicciones o archivo legacy 'feedback.csv' ...")
    feedback_df = load_feedback(feedback_path)
    # If predictions exists, proceed; if not, abort
    if predictions_df is None:
        print("\nNo hay predicciones para combinar. Genere predicciones en la UI antes de ejecutar este script.")
        return
    
    print(f"\nPredicciones disponibles: {len(predictions_df)}")
    print(f"Registros de feedback legacy: {len(feedback_df) if feedback_df is not None else 0}")
    
    # Combinar
    print("\n3. Combinando predicciones con feedback...")
    combined_df = combine_predictions_with_feedback(predictions_df, feedback_df)
    
    if combined_df is not None:
        # Guardar
        print("\n4. Guardando dataset etiquetado...")
        output_path = save_labeled_data(combined_df, output_dir)
        
        if output_path:
            print("\n" + "=" * 60)
            print("✅ PROCESO COMPLETADO EXITOSAMENTE")
            print("=" * 60)
            print(f"\nEl dataset etiquetado está listo para reentrenamiento.")
            print(f"Ejecute: python -m src.ml.train")
            print("=" * 60)
    else:
        print("\n❌ No se pudo generar el dataset combinado")


if __name__ == '__main__':
    main()
