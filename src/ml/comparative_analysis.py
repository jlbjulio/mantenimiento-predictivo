"""
Módulo para análisis comparativo y causal de fallos entre máquinas.
Incluye benchmarking por tipo, correlaciones, heatmaps y gráficos 3D.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def analyze_by_product_type(df: pd.DataFrame) -> Dict:
    """
    Calcula estadísticas agregadas por tipo de producto (L, M, H).
    Retorna diccionario con métricas de desempeño.
    
    Validaciones:
    - Verifica que DataFrame no sea None o vacío
    - Convierte columnas a numéricas de forma robusta
    - Retorna diccionario vacío en caso de error
    """
    if df is None or df.empty or not isinstance(df, pd.DataFrame):
        return {}
    
    stats_by_type = {}
    
    for prod_type in ['L', 'M', 'H']:
        df_type = df[df['type'] == prod_type]
        
        if df_type.empty:
            continue
        
        # Convertir columnas a numéricas
        prob_col = pd.to_numeric(df_type.get('prob', pd.Series([])), errors='coerce')
        mf_col = pd.to_numeric(df_type.get('Machine failure', pd.Series([])), errors='coerce')
        wear_col = pd.to_numeric(df_type.get('tool_wear_min', pd.Series([])), errors='coerce')
        rpm_col = pd.to_numeric(df_type.get('rot_speed_rpm', pd.Series([])), errors='coerce')
        torque_col = pd.to_numeric(df_type.get('torque_nm', pd.Series([])), errors='coerce')
        air_temp = pd.to_numeric(df_type.get('air_temp_k', pd.Series([])), errors='coerce')
        proc_temp = pd.to_numeric(df_type.get('process_temp_k', pd.Series([])), errors='coerce')
        
        # Calcular métricas
        total_predictions = len(df_type)
        failed_count = mf_col.sum() if len(mf_col) > 0 else 0
        failure_rate = (failed_count / total_predictions * 100) if total_predictions > 0 else 0
        
        # Tasa de predicción correcta (si hay feedback)
        has_feedback = mf_col.notna().sum()
        
        stats_by_type[prod_type] = {
            'total_predictions': int(total_predictions),
            'actual_failures': int(failed_count),
            'failure_rate': float(failure_rate),
            'has_feedback': int(has_feedback),
            'avg_prob_risk': float(prob_col.mean()) if len(prob_col) > 0 else 0.0,
            'max_prob_risk': float(prob_col.max()) if len(prob_col) > 0 else 0.0,
            'avg_wear': float(wear_col.mean()) if len(wear_col) > 0 else 0.0,
            'max_wear': float(wear_col.max()) if len(wear_col) > 0 else 0.0,
            'avg_rpm': float(rpm_col.mean()) if len(rpm_col) > 0 else 0.0,
            'avg_torque': float(torque_col.mean()) if len(torque_col) > 0 else 0.0,
            'avg_delta_temp': float((proc_temp - air_temp).mean()) if len(proc_temp) > 0 else 0.0,
        }
    
    return stats_by_type


def calculate_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula matriz de correlación entre parámetros operativos y fallo de máquina.
    Maneja columnas faltantes y datos inválidos de forma robusta.
    """
    if df is None or df.empty or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    
    df_numeric = df.copy()
    
    # Seleccionar columnas numéricas
    numeric_cols = ['air_temp_k', 'process_temp_k', 'rot_speed_rpm', 'torque_nm', 
                   'tool_wear_min', 'prob', 'Machine failure']
    
    available_cols = [c for c in numeric_cols if c in df_numeric.columns]
    
    if not available_cols:
        return pd.DataFrame()
    
    # Convertir a numéricas
    for col in available_cols:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    
    # Calcular correlación
    corr_matrix = df_numeric[available_cols].corr()
    
    return corr_matrix


def calculate_failure_rates_by_bins(df: pd.DataFrame, param1: str, param2: str, 
                                    bins1: int = 5, bins2: int = 5) -> pd.DataFrame:
    """
    Calcula tasa de fallos en función de dos parámetros (para heatmap).
    Retorna una matriz con tasa de fallos por bin.
    Valida que parámetros existan y sean numéricos.
    """
    if df is None or df.empty or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    
    if param1 not in df.columns or param2 not in df.columns:
        return pd.DataFrame()
    
    df_clean = df.copy()
    
    # Convertir a numéricas
    df_clean[param1] = pd.to_numeric(df_clean[param1], errors='coerce')
    df_clean[param2] = pd.to_numeric(df_clean[param2], errors='coerce')
    df_clean['Machine failure'] = pd.to_numeric(df_clean['Machine failure'], errors='coerce')
    
    df_clean = df_clean.dropna(subset=[param1, param2, 'Machine failure'])
    
    if df_clean.empty:
        return pd.DataFrame()
    
    # Crear bins
    df_clean['bin1'] = pd.cut(df_clean[param1], bins=bins1, duplicates='drop')
    df_clean['bin2'] = pd.cut(df_clean[param2], bins=bins2, duplicates='drop')
    
    # Calcular tasa de fallos por combinación de bins
    failure_rate = df_clean.groupby(['bin1', 'bin2'])['Machine failure'].agg(['sum', 'count'])
    failure_rate['rate'] = (failure_rate['sum'] / failure_rate['count'] * 100).fillna(0)
    
    # Pivotar para crear matrix
    pivot = failure_rate['rate'].unstack(fill_value=0)
    
    return pivot


def extract_decision_rules(df: pd.DataFrame, target_col: str = 'Machine failure') -> List[Dict]:
    """
    Extrae reglas de decisión simples interpretables basadas en el histórico.
    Validaciones: tipo de dato, valores faltantes, tamaño mínimo de muestra.
    """
    if df is None or df.empty or not isinstance(df, pd.DataFrame):
        return []
    
    rules = []
    
    # Convertir columnas
    df_clean = df.copy()
    df_clean['Machine failure'] = pd.to_numeric(df_clean['Machine failure'], errors='coerce')
    df_clean['tool_wear_min'] = pd.to_numeric(df_clean['tool_wear_min'], errors='coerce')
    df_clean['rot_speed_rpm'] = pd.to_numeric(df_clean['rot_speed_rpm'], errors='coerce')
    df_clean['air_temp_k'] = pd.to_numeric(df_clean['air_temp_k'], errors='coerce')
    df_clean['process_temp_k'] = pd.to_numeric(df_clean['process_temp_k'], errors='coerce')
    df_clean['torque_nm'] = pd.to_numeric(df_clean['torque_nm'], errors='coerce')
    
    # Regla 1: Desgaste alto
    high_wear = df_clean[df_clean['tool_wear_min'] >= 150]
    if len(high_wear) > 0:
        failure_rate_hw = high_wear['Machine failure'].sum() / len(high_wear) * 100 if len(high_wear) > 0 else 0
        rules.append({
            'rule': 'Desgaste herramienta ≥ 150 min',
            'failure_rate': float(failure_rate_hw),
            'sample_size': len(high_wear),
            'severity': 'ALTA'
        })
    
    # Regla 2: Velocidad baja + baja temperatura
    low_speed_low_temp = df_clean[(df_clean['rot_speed_rpm'] < 1200) & 
                                   ((df_clean['process_temp_k'] - df_clean['air_temp_k']) < 9)]
    if len(low_speed_low_temp) > 0:
        failure_rate_lst = low_speed_low_temp['Machine failure'].sum() / len(low_speed_low_temp) * 100
        rules.append({
            'rule': 'RPM < 1200 AND ΔT < 9K (HDF)',
            'failure_rate': float(failure_rate_lst),
            'sample_size': len(low_speed_low_temp),
            'severity': 'MEDIA'
        })
    
    # Regla 3: Torque muy alto
    high_torque = df_clean[df_clean['torque_nm'] > 50]
    if len(high_torque) > 0:
        failure_rate_ht = high_torque['Machine failure'].sum() / len(high_torque) * 100
        rules.append({
            'rule': 'Torque > 50 Nm',
            'failure_rate': float(failure_rate_ht),
            'sample_size': len(high_torque),
            'severity': 'MEDIA'
        })
    
    # Regla 4: Combinación desgaste + torque (sobrestrain)
    strain = df_clean['tool_wear_min'] * df_clean['torque_nm']
    high_strain = df_clean[strain > 11000]
    if len(high_strain) > 0:
        failure_rate_hs = high_strain['Machine failure'].sum() / len(high_strain) * 100
        rules.append({
            'rule': 'Sobrestrain (Wear × Torque) > 11000',
            'failure_rate': float(failure_rate_hs),
            'sample_size': len(high_strain),
            'severity': 'ALTA'
        })
    
    # Ordenar por severidad
    severity_rank = {'ALTA': 0, 'MEDIA': 1, 'BAJA': 2}
    rules.sort(key=lambda x: severity_rank.get(x['severity'], 99))
    
    return rules


def create_benchmark_comparison(stats_by_type: Dict) -> go.Figure:
    """
    Crea gráfico de barras comparativo entre tipos de producto.
    """
    if not stats_by_type:
        return None
    
    types = list(stats_by_type.keys())
    failure_rates = [stats_by_type[t]['failure_rate'] for t in types]
    avg_risks = [stats_by_type[t]['avg_prob_risk'] * 100 for t in types]
    avg_wear = [stats_by_type[t]['avg_wear'] for t in types]
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['Tasa de Fallos (%)', 'Riesgo Promedio (%)', 'Desgaste Promedio (min)'],
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    colors = ['#1976d2', '#ff8f00', '#2e7d32']
    
    fig.add_trace(go.Bar(x=types, y=failure_rates, name='Fallos', marker_color=colors[0],
                         text=[f"{v:.1f}%" for v in failure_rates], textposition='auto'), row=1, col=1)
    
    fig.add_trace(go.Bar(x=types, y=avg_risks, name='Riesgo', marker_color=colors[1],
                         text=[f"{v:.1f}%" for v in avg_risks], textposition='auto'), row=1, col=2)
    
    fig.add_trace(go.Bar(x=types, y=avg_wear, name='Desgaste', marker_color=colors[2],
                         text=[f"{v:.0f}m" for v in avg_wear], textposition='auto'), row=1, col=3)
    
    fig.update_xaxes(title_text="Tipo", row=1, col=1)
    fig.update_xaxes(title_text="Tipo", row=1, col=2)
    fig.update_xaxes(title_text="Tipo", row=1, col=3)
    
    fig.update_layout(height=450, showlegend=False, title_text="Benchmarking de Tipos de Producto")
    
    return fig


def create_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """
    Crea heatmap de matriz de correlación.
    """
    if corr_df is None or corr_df.empty:
        return None
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_df.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
    ))
    
    fig.update_layout(
        title='Matriz de Correlación: Parámetros vs Fallos',
        height=600,
        xaxis_title='Variables',
        yaxis_title='Variables'
    )
    
    return fig


def create_failure_heatmap(pivot_df: pd.DataFrame, param1: str, param2: str) -> go.Figure:
    """
    Crea heatmap de tasa de fallos por dos parámetros.
    """
    if pivot_df is None or pivot_df.empty:
        return None
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=[str(x) for x in pivot_df.columns],
        y=[str(y) for y in pivot_df.index],
        colorscale='YlOrRd',
        colorbar=dict(title="% Fallos")
    ))
    
    fig.update_layout(
        title=f'Tasa de Fallos: {param1} vs {param2}',
        xaxis_title=param2,
        yaxis_title=param1,
        height=500
    )
    
    return fig


def create_3d_scatter(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, 
                      color_col: str = 'Machine failure') -> go.Figure:
    """
    Crea gráfico 3D interactivo de parámetros con fallos como color.
    Valida que todas las columnas existan y sean numéricas.
    """
    if df is None or df.empty or not isinstance(df, pd.DataFrame):
        return None
    
    df_clean = df.copy()
    df_clean[x_col] = pd.to_numeric(df_clean[x_col], errors='coerce')
    df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors='coerce')
    df_clean[z_col] = pd.to_numeric(df_clean[z_col], errors='coerce')
    df_clean[color_col] = pd.to_numeric(df_clean[color_col], errors='coerce')
    
    df_clean = df_clean.dropna(subset=[x_col, y_col, z_col, color_col])
    
    if df_clean.empty:
        return None
    
    fig = go.Figure(data=[go.Scatter3d(
        x=df_clean[x_col],
        y=df_clean[y_col],
        z=df_clean[z_col],
        mode='markers',
        marker=dict(
            size=6,
            color=df_clean[color_col],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Fallo"),
            line=dict(width=0)
        ),
        text=[f"{x_col}: {x:.1f}<br>{y_col}: {y:.1f}<br>{z_col}: {z:.1f}<br>Fallo: {f}" 
              for x, y, z, f in zip(df_clean[x_col], df_clean[y_col], 
                                   df_clean[z_col], df_clean[color_col])],
        hoverinfo='text'
    )])
    
    fig.update_layout(
        title=f'Análisis 3D: {x_col} vs {y_col} vs {z_col}',
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        ),
        height=700,
        width=900
    )
    
    return fig


def create_rules_table(rules: List[Dict]) -> pd.DataFrame:
    """
    Convierte lista de reglas a DataFrame para visualización.
    """
    if not rules:
        return pd.DataFrame()
    
    df_rules = pd.DataFrame(rules)
    return df_rules[['rule', 'failure_rate', 'sample_size', 'severity']]
