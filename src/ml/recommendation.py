from typing import List, Dict
import math


def generate_recommendations(instance: Dict, failure_prob: float, shap_contrib: Dict = None) -> List[Dict]:
    recs = []
    # Regla desgaste herramienta
    wear = instance.get('tool_wear_min')
    if wear is not None and wear >= 200:
        recs.append({
            'accion': 'Programar reemplazo inmediato de herramienta',
            'justificacion': f'Desgaste {wear} min >= 200 min umbral crítico (riesgo TWF).'
        })
    # Regla delta temperatura y velocidad
    delta_temp = instance.get('process_temp_k') - instance.get('air_temp_k') if instance.get('process_temp_k') and instance.get('air_temp_k') else None
    rot = instance.get('rot_speed_rpm')
    if delta_temp is not None and rot is not None:
        if delta_temp < 9 and rot < 1400:
            recs.append({
                'accion': 'Incrementar diferencia térmica o velocidad de rotación',
                'justificacion': f'Delta temp {delta_temp:.1f}K y rotación {rot}rpm en zona HDF.'
            })
    # Regla potencia fuera de rango
    torque = instance.get('torque_nm')
    if torque is not None and rot is not None:
        omega = rot * 2 * math.pi / 60
        power = torque * omega
        if power < 3500:
            recs.append({'accion': 'Ajustar torque/velocidad para alcanzar potencia mínima', 'justificacion': f'Potencia estimada {power:.0f}W < 3500W (riesgo PWF).'})
        elif power > 9000:
            recs.append({'accion': 'Reducir carga o velocidad para evitar sobrepotencia', 'justificacion': f'Potencia estimada {power:.0f}W > 9000W (riesgo PWF).'})
    # Regla sobrestrain variante tipo
    prod_type = instance.get('type') or instance.get('Type')
    if prod_type and torque is not None and wear is not None:
        thresholds = {'L': 11000, 'M': 12000, 'H': 13000}
        limit = thresholds.get(prod_type, 12000)
        strain = wear * torque
        if strain > limit:
            recs.append({'accion': 'Reducir torque o reemplazar herramienta (sobrestrain)', 'justificacion': f'Strain {strain:.0f} min*Nm > {limit} límite {prod_type} (OSF).'})
    # Regla probabilidad global
    if failure_prob >= 0.5:
        top = None
        if shap_contrib:
            # tomar feature con mayor contribución positiva
            top = sorted(shap_contrib.items(), key=lambda x: x[1], reverse=True)[0][0]
        recs.append({'accion': 'Activar inspección preventiva', 'justificacion': f'Probabilidad fallo {failure_prob:.2f} >= 0.50' + (f'. Feature crítica: {top}' if top else '')})
    if not recs:
        recs.append({'accion': 'Operar normalmente', 'justificacion': 'Parámetros dentro de rangos seguros.'})
    return recs
