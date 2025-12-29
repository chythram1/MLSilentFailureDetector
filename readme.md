# 🔍 ML Silent Failure Detection System

An early warning system that detects machine learning model degradation **before** accuracy drops, giving teams time to investigate and fix issues.
Some Pictures from Streamlit dashboard:

<img width="935" height="317" alt="image" src="https://github.com/user-attachments/assets/850b146c-0d5d-48c5-a70e-21e915d8de02" />
<img width="652" height="353" alt="image" src="https://github.com/user-attachments/assets/957cd510-c465-4478-870e-f4b1086e4b75" />
<img width="658" height="374" alt="image" src="https://github.com/user-attachments/assets/c0f7f2dd-42a3-4ba1-9ada-042cc65386c9" />
<img width="667" height="337" alt="image" src="https://github.com/user-attachments/assets/7c32849d-ade8-4df2-ae5c-7bb23835ec36" />
<img width="593" height="341" alt="image" src="https://github.com/user-attachments/assets/7aacdc95-d648-4d46-97af-cf562218be8f" />

---

## 🎯 The Problem

Machine learning models in production **degrade silently**. By the time you notice accuracy dropped, you've already made thousands of bad predictions.

```
Traditional Monitoring:

Week 1-10:  Model seems fine (no labels yet to verify)
Week 11:    Labels arrive → Accuracy dropped to 60%
Week 12:    Panic. Thousands of wrong predictions already made.
```

**Why does this happen?**

- Customer behavior changes over time
- Economic conditions shift
- Marketing brings in different demographics
- Seasonal patterns emerge

The model was trained on historical data, but production data keeps evolving.

---

## ✅ The Solution

Monitor **input data distributions** instead of waiting for labels. If today's data looks different from training data, the model's predictions are suspect—even before you can prove it.

```
This System:

Week 1-3:   PSI low, model stable
Week 4:     ⚠️ PSI spikes! Data distribution shifting.
Week 5-10:  Team investigates, retrains model
Week 11:    Crisis averted. Model still accurate.
```

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Drift detected at | Batch 4 |
| Accuracy dropped at | Batch 15 |
| **Early warning** | **11 batches** |
| Accuracy degradation | 80% → 60% |

**The system detected problems 11 time windows before users would have noticed.**

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      TRAINING PHASE                         │
├─────────────────────────────────────────────────────────────┤
│  Historical Data → Train Model → Save Reference Baseline    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     PRODUCTION PHASE                        │
├─────────────────────────────────────────────────────────────┤
│  New Data Batch                                             │
│       │                                                     │
│       ▼                                                     │
│  Calculate PSI (compare to reference)                       │
│       │                                                     │
│       ▼                                                     │
│  PSI > 0.25? ───YES───→ 🚨 Alert! Investigate drift        │
│       │                                                     │
│       NO                                                    │
│       ▼                                                     │
│  Continue monitoring                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
ml-silent-failure/
├── data/
│   ├── raw/
│   │   └── credit_default.csv
│   └── processed/
│       ├── reference_data.csv
│       ├── reference_with_labels.csv
│       ├── simulated_batches.csv
│       ├── psi_results.csv
│       └── drift_accuracy_results.csv
├── models/
│   ├── baseline_model.pkl
│   └── scaler.pkl
├── src/
│   ├── drift_simulator.py
│   ├── psi_calculator.py
│   ├── batch_evaluator.py
│   └── dashboard.py
├── notebooks/
│   └── test.ipynb
├── README.md
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ml-silent-failure.git
cd ml-silent-failure
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download the [UCI Credit Card Default dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) and place it in `data/raw/credit_default.csv`.

### 4. Run the pipeline

```bash
# Train baseline model
run the third cell in test.ipynb

# Generate drifted batches
python src/drift_simulator.py

# Calculate PSI scores
python src/psi_calculator.py

# Evaluate accuracy per batch
python src/batch_evaluator.py

# Launch dashboard
python -m streamlit run src/dashboard.py
```

---

## 📈 What is PSI?

**Population Stability Index (PSI)** measures how different two distributions are.

```
Reference (training):   |  ▂▄▆█▆▄▂  |    ← What model learned
Current (production):   |▂▄▆█▆▄▂    |    ← What model sees now
                              ↑
                         PSI detects this shift
```

### Interpretation

| PSI Value | Meaning |
|-----------|---------|
| < 0.1 | No significant drift |
| 0.1 - 0.25 | Moderate drift, investigate |
| > 0.25 | Significant drift, model unreliable |

### Why PSI?

- **Label-free**: Works without ground truth
- **Interpretable**: Clear thresholds
- **Industry standard**: Used at banks, insurance, and tech companies

---

## 🔬 Drift Simulation

We simulate real-world drift by gradually shifting key features:

| Feature | What It Is | Drift Scenario |
|---------|------------|----------------|
| `PAY_0` | Payment status last month (0=on time, 1+=late) | Economic recession → more late payments |
| `BILL_AMT1` | Bill amount last month | Inflation → higher balances |
| `PAY_AMT2` | Payment amount 2 months ago | Customer mix changes |

These features were chosen because they have the **highest model coefficients**—drifting them impacts predictions most.

---

## 🖥️ Dashboard Features

The Streamlit dashboard provides:

- **Accuracy timeline**: Watch model performance over batches
- **PSI monitoring**: Color-coded drift detection (green=stable, red=alert)
- **Feature breakdown**: Which features are drifting most
- **Event timeline**: Clear narrative of when drift was detected vs. when accuracy dropped
- **Recommended actions**: What to do when drift is detected

---

## 🧠 Key Concepts Demonstrated

1. **Silent failure in ML**: Models degrade without obvious errors
2. **Distribution monitoring**: Detecting problems without labels
3. **PSI calculation**: Industry-standard drift metric
4. **Feature importance**: Knowing which features matter most
5. **Proactive monitoring**: Early warning systems for ML

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **scikit-learn**: Baseline model training
- **pandas/numpy**: Data manipulation
- **Streamlit**: Interactive dashboard
- **Plotly**: Visualizations
- **joblib**: Model serialization

---

## 📚 References

- [UCI Credit Card Default Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- [Understanding UCI Credit Default Dataset](https://medium.com/@manish.kumar_61520/the-default-of-credit-card-clients-dataset-81908562a67eh)
- [Population Stability Index (PSI) Explained](https://www.listendata.com/2015/05/population-stability-index.html)
- [Monitoring ML Models in Production](https://christophergs.com/machine-learning/2020/03/14/how-to-monitor-machine-learning-models/)

---

## 🔮 Future Improvements

- [ ] Add more drift metrics (KL Divergence, KS Test)
- [ ] Implement automated retraining pipeline
- [ ] Add email/Slack alerts
- [ ] Support real-time streaming data
- [ ] Add concept drift detection (feature-label relationship changes)
