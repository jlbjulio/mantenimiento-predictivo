# Intelligent Predictive Maintenance System with Advanced Multilabel Diagnosis

## Overview

**Automatas** is an intelligent predictive maintenance system that uses machine learning to anticipate failures in CNC tools and machines before they occur. Through real-time analysis of operating parameters (temperature, speed, torque, wear), the system predicts failure probabilities with 5 specific failure types and generates actionable recommendations for proactive interventions.

The system integrates an advanced multilabel diagnostic engine, intelligent probability calibration, and a causal recommendation engine based on real operating parameters to reduce unplanned downtime, optimize operational costs, and increase machine availability.

https://newautomatas.streamlit.app/

## Features

- Advanced multilabel diagnosis (5 failure types simultaneously)
- Intelligent probability calibration (70% model + 30% empirical)
- Causal recommendation engine based on real parameters
- 6 interactive tabs with complete analysis
- Automatic prediction logging with feedback system
- Robust data protection and CSV integrity handling
- Real-time parameter analysis with critical thresholds
- Scenario simulator for operational planning

## Technologies Used

**Backend / Machine Learning:**

- Python 3.10+
- scikit-learn (Random Forest, Gradient Boosting, Logistic Regression)
- pandas
- NumPy

**Frontend / UI:**

- Streamlit
- Plotly
- Altair

**Data & Storage:**

- CSV (pandas)
- ai4i2020 Dataset (10,000 industrial machine records)

**DevOps / Deployment:**

- Git / GitHub
- Streamlit Cloud

## Project Structure

```
ProyectoAutomatas/
├── app/
│   ├── streamlit_app.py           # Main UI with 6 tabs
│   └── api.py                     # REST API endpoints
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   ├── preprocess.py
│   │   └── __init__.py
│   └── ml/
│       ├── train.py
│       ├── comparative_analysis.py
│       └── __init__.py
├── models/
│   ├── failure_binary_model.joblib
│   ├── failure_multilabel_models.joblib
│   └── versions/
├── logs/
│   └── predicciones.csv
├── data/
│   └── ai4i2020.csv
├── requirements.txt
└── README.md
```

## Main Features

- **Prediction Tab**: Real-time risk analysis with operating parameter inputs (temperature, RPM, torque, wear, product type)
- **Failure Type Tab**: Multilabel diagnosis with 5 specific failure modes (TWF, HDF, PWF, OSF, RNF)
- **Simulator Tab**: Compare two operational scenarios with risk and failure probability visualization
- **Comparative Analysis Tab**: Benchmarking, correlation matrix, causal heatmaps, and 3D interactive exploration
- **Info/Help Tab**: Variable documentation, risk indicators, failure type explanations, and model status
- **History Tab**: Temporal filtering, statistical summaries, risk evolution graphs, and detailed tables

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

### 1. Train Model

```bash
python -m src.ml.train
```

### 2. Run Application

```bash
streamlit run app/streamlit_app.py
```

Open browser at `http://localhost:8501`

## System Usage

### Making Predictions

1. Adjust parameters in the sidebar (temperature, RPM, torque, wear, product type)
2. Click **"Calculate Prediction and Recommendations"**
3. Review risk level and actionable recommendations
4. View detailed analysis of critical parameters vs thresholds
5. Provide feedback after actual operation

### Feedback System

After each prediction and real operation:

1. In **"📝 Post-Operation Feedback"** section:
   - Click **"✅ No failure occurred"** if operation was successful
   - Click **"🚨 Failure did occur"** if a failure happened
2. System automatically records result in `logs/predicciones.csv`

### Key Features

**Calibrated Prediction:**

```
Input: Operating parameters
├─ Base model probability
├─ Historical neighbor search: k=25 similar records
├─ Empirical failure rate in neighbors
└─ Calibrated probability: 70% model + 30% empirical
Output: Calibrated probability [0-1]
```

**Root Cause Analysis:**

```
Identifies which parameters trigger risk:
- Tool wear ≥ 200 min → Replace tool
- Thermal delta < 9K → Improve heat dissipation
- Power < 3500W or > 9000W → Adjust load
- Strain > type_limit → Reduce torque
```

**Automatic Logging:**

```
Every prediction recorded with:
- Exact timestamp (ISO format)
- Operating parameters
- Predicted probability
- Product type
- Feedback result (when provided)
```

## Failure Modes

| Failure Type     | Code | Trigger              | Recommended Action          |
| ---------------- | ---- | -------------------- | --------------------------- |
| Tool Wear        | TWF  | Wear ≥ 200 min       | Replace tool                |
| Heat Dissipation | HDF  | ΔT < 9K, RPM < 1400  | Increase thermal difference |
| Power Failure    | PWF  | P < 3500W or > 9000W | Adjust torque/RPM           |
| Overstrain       | OSF  | Strain > type_limit  | Reduce load                 |
| Random Network   | RNF  | Probability ≥ 30%    | Execute diagnostic          |

## Model Performance

```
Model: Gradient Boosting Classifier
Dataset: 10,000 records (80-20 split)
Metric          | Score
─────────────────┼────────
AUC-ROC          | 0.95+
Precision        | 0.92
Recall           | 0.89
F1-Score         | 0.90
```

## Deployment

### Streamlit Cloud (Recommended)

```bash
git push origin main
# App deploys automatically at:
# https://[username]-automatas-app.streamlit.app
```

### Local Deployment

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
# Access at: http://localhost:8501
```

## Key Metrics and Thresholds

| Variable          | Safe Range             | Critical Threshold | Failure Mode |
| ----------------- | ---------------------- | ------------------ | ------------ |
| Wear (min)        | 0-150                  | ≥ 200              | TWF          |
| Thermal Delta (K) | ≥ 9 + RPM ≥ 1400       | < 9K               | HDF          |
| Power (W)         | 3500-9000              | < 3500 or > 9000   | PWF          |
| Strain            | L:<11K, M:<12K, H:<13K | Exceeds limit      | OSF          |

## Impact and Sustainability

- **Operational Resilience**: Fewer unplanned downtime events, greater continuity and efficient shift management
- **Energy Efficiency**: Maintaining power and torque in safe zone reduces unnecessary consumption and peaks
- **Less Waste**: Reduced scrap from tool failures and less rework, lowering associated carbon footprint
- **Extended Lifespan**: Proactive adjustments prevent overstrain, extending tool and component life
- **Continuous Improvement**: Labeled feedback feeds model retraining, keeping it aligned with new conditions

## Considerations

- Units are relative to the system defined by the user
- Calibration uses historical data (k=25 nearest neighbors) for probability adjustment
- Recommendations are triggered only when risk probability is significant (≥ 30%)
- CSV integrity is protected against data corruption through newline handling
- All predictions are timestamped for full auditability

## Possible Improvements

- Add additional failure modes beyond the current 5 types
- Integrate with IoT sensors for automated parameter reading
- Export results to PDF or detailed reports
- Enhance UI with advanced visualization libraries
- Implement distributed training for larger datasets
- Add predictive maintenance scheduling optimization
- Create mobile app for remote monitoring

## Dataset Citation

"Explainable Artificial Intelligence for Predictive Maintenance Applications"  
S. Matzka, Third International Conference on Artificial Intelligence for Industries (AI4I 2020)

---

> **Note**: This system was developed as a comprehensive solution for intelligent predictive maintenance in CNC manufacturing environments, combining machine learning models with domain knowledge for actionable insights.
