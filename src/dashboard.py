# dashboard.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

# ===== Setup =====
st.set_page_config(page_title="ML Drift Monitor", layout="wide")
st.title("🔍 ML Silent Failure Detection Dashboard")

# ===== Context Section =====
st.markdown("""
### What is this?
This dashboard monitors a **credit default prediction model** in production. 
Machine learning models degrade silently over time as real-world data shifts away from training data.
By the time accuracy drops, you've already made thousands of bad predictions.

**Our solution:** Detect drift in input data *before* accuracy drops, giving you time to fix the model.
""")

st.divider()

# ===== Load Data =====
@st.cache_data
def load_data():
    # Try multiple paths
    possible_paths = [
        'data/processed/drift_accuracy_results.csv',  # From repo root
        '../data/processed/drift_accuracy_results.csv',  # From src folder
        os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'drift_accuracy_results.csv')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    
    st.error("Could not find drift_accuracy_results.csv")
    return None

df = load_data()

if df is None:
    st.stop()

# Calculate aggregate PSI
psi_features = ['PAY_0', 'BILL_AMT1', 'PAY_AMT2']
df['max_psi'] = df[psi_features].max(axis=1)

# ===== Sidebar =====
st.sidebar.header("Settings")
psi_threshold = st.sidebar.slider("PSI Alert Threshold", 0.1, 1.0, 0.25)

st.sidebar.divider()
st.sidebar.markdown("""
### PSI Thresholds
- **< 0.1** — No drift
- **0.1 - 0.25** — Investigate
- **> 0.25** — Model unreliable
""")

# ===== Key Finding Banner =====
first_drift_batch = df[df['max_psi'] > psi_threshold]['batch_id'].min()
accuracy_drop_batch = df[df['accuracy'] < 0.75]['batch_id'].min()

if pd.notna(first_drift_batch) and pd.notna(accuracy_drop_batch):
    lead_time = int(accuracy_drop_batch - first_drift_batch)
    st.success(f"""
    ### ✅ Key Finding
    **PSI detected drift at batch {int(first_drift_batch)}**, but **accuracy didn't drop until batch {int(accuracy_drop_batch)}**.
    
    That's **{lead_time} batches of early warning** — time to investigate and retrain before users are affected.
    """)

st.divider()

# ===== Chart 1: Accuracy Over Time =====
st.header("📉 Model Accuracy Over Time")

fig1 = go.Figure()

fig1.add_trace(
    go.Scatter(
        x=df['batch_id'],
        y=df['accuracy'],
        name="Accuracy",
        line=dict(color='#2ECC71', width=3),
        fill='tozeroy',
        fillcolor='rgba(46, 204, 113, 0.2)'
    )
)

fig1.add_hline(y=0.75, line_dash="dash", line_color="red", 
               annotation_text="Alert: 75%")

# Add annotation for accuracy drop
if pd.notna(accuracy_drop_batch):
    drop_accuracy = df[df['batch_id'] == accuracy_drop_batch]['accuracy'].values[0]
    fig1.add_annotation(
        x=accuracy_drop_batch, y=drop_accuracy,
        text=f"Accuracy drops below 75%",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
        ax=40, ay=-40
    )

fig1.update_layout(
    height=350,
    yaxis=dict(range=[0.5, 1.0], title="Accuracy"),
    xaxis=dict(title="Batch ID (simulated weeks)")
)

st.plotly_chart(fig1, use_container_width=True)

st.caption("Each batch represents a week of new customer applications. Accuracy measures how well the model predicts defaults.")

st.divider()

# ===== Chart 2: Aggregate PSI =====
st.header("🚨 Drift Detection (PSI Score)")

fig2 = go.Figure()

fig2.add_trace(
    go.Bar(
        x=df['batch_id'],
        y=df['max_psi'],
        name="Max PSI",
        marker_color=['#2ECC71' if x < psi_threshold else '#E74C3C' for x in df['max_psi']]
    )
)

fig2.add_hline(y=psi_threshold, line_dash="dash", line_color="red",
               annotation_text=f"Threshold ({psi_threshold})")

# Add annotation for first drift
if pd.notna(first_drift_batch):
    drift_psi = df[df['batch_id'] == first_drift_batch]['max_psi'].values[0]
    fig2.add_annotation(
        x=first_drift_batch, y=drift_psi,
        text=f"First drift detected!",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
        ax=-40, ay=-40
    )

fig2.update_layout(
    height=350,
    yaxis=dict(title="Max PSI Score"),
    xaxis=dict(title="Batch ID (simulated weeks)")
)

st.plotly_chart(fig2, use_container_width=True)

st.caption("""
**PSI (Population Stability Index)** measures how different current data is from training data.
Green bars = stable. Red bars = data has shifted significantly, model predictions may be unreliable.
""")

st.divider()

# ===== What's Causing the Drift? =====
st.header("🔬 What's Causing the Drift?")

st.markdown("""
We're monitoring three key features that the model relies on heavily for predictions.
Here's what each feature means and what drift would look like in the real world:
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### PAY_0
    **What it is:** Payment status last month
    - -1 = Paid early
    - 0 = Paid on time  
    - 1+ = Months overdue
    
    **Drift scenario:** Economic recession causes more customers to miss payments.
    
    **Impact:** Model sees more late payers than it was trained on.
    """)

with col2:
    st.markdown("""
    #### BILL_AMT1
    **What it is:** Bill statement amount last month (in dollars)
    
    **Drift scenario:** Inflation increases average spending, or marketing attracts higher-spending customers.
    
    **Impact:** Model sees larger balances than during training.
    """)

with col3:
    st.markdown("""
    #### PAY_AMT2  
    **What it is:** Payment amount two months ago (in dollars)
    
    **Drift scenario:** Mix of customer base changes — some paying more, some less.
    
    **Impact:** Model sees payment patterns it hasn't learned.
    """)

st.divider()

# ===== Chart 3: PSI Breakdown by Feature =====
st.header("📊 PSI Breakdown by Feature")

fig3 = go.Figure()

colors = {'PAY_0': '#E74C3C', 'BILL_AMT1': '#3498DB', 'PAY_AMT2': '#9B59B6'}

for feature in psi_features:
    fig3.add_trace(
        go.Scatter(
            x=df['batch_id'],
            y=df[feature],
            name=feature,
            line=dict(color=colors[feature], width=2)
        )
    )

fig3.add_hline(y=psi_threshold, line_dash="dash", line_color="gray",
               annotation_text="Threshold")

fig3.update_layout(
    height=350,
    yaxis=dict(title="PSI Score"),
    xaxis=dict(title="Batch ID (simulated weeks)")
)

st.plotly_chart(fig3, use_container_width=True)

# Identify which feature drifted first/most
max_psi_feature = df[psi_features].iloc[-1].idxmax()
st.info(f"**{max_psi_feature}** shows the strongest drift by the final batch, suggesting this feature changed most dramatically in the customer population.")

st.divider()

# ===== Timeline Summary =====
st.header("📅 Timeline of Events")

timeline_data = []

# Stable period
timeline_data.append({
    'Batch': '0-3',
    'Event': 'Stable Period',
    'Description': 'PSI low, accuracy ~80%. Model performing as expected.'
})

# First drift
if pd.notna(first_drift_batch):
    timeline_data.append({
        'Batch': str(int(first_drift_batch)),
        'Event': '⚠️ Drift Detected',
        'Description': f'PSI exceeds {psi_threshold} threshold. Data distribution shifting.'
    })

# Accuracy drop
if pd.notna(accuracy_drop_batch):
    timeline_data.append({
        'Batch': str(int(accuracy_drop_batch)),
        'Event': '🔴 Accuracy Drops',
        'Description': 'Model accuracy falls below 75%. Performance degraded.'
    })

# Final state
timeline_data.append({
    'Batch': str(len(df) - 1),
    'Event': 'Final State',
    'Description': f"Accuracy: {df['accuracy'].iloc[-1]:.1%}. Model needs retraining."
})

timeline_df = pd.DataFrame(timeline_data)
st.table(timeline_df)

st.divider()

# ===== Summary Stats =====
st.header("📈 Summary Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Initial Accuracy", f"{df['accuracy'].iloc[0]:.1%}")

with col2:
    st.metric("Final Accuracy", f"{df['accuracy'].iloc[-1]:.1%}", 
              delta=f"{df['accuracy'].iloc[-1] - df['accuracy'].iloc[0]:.1%}")

with col3:
    if pd.notna(first_drift_batch) and pd.notna(accuracy_drop_batch):
        st.metric("Early Warning", f"{int(accuracy_drop_batch - first_drift_batch)} batches")
    else:
        st.metric("Early Warning", "N/A")

with col4:
    st.metric("Batches Monitored", len(df))

st.divider()

# ===== What Would You Do? =====
st.header("🛠️ Recommended Actions")

st.markdown("""
In a real production environment, when PSI exceeds threshold, the team should:

1. **Investigate the drift** — Which features shifted? Is this expected (seasonal) or unexpected?
2. **Check upstream data** — Did a data pipeline break? Did a vendor change formats?
3. **Evaluate business context** — New marketing campaign? Economic changes? New customer segment?
4. **Retrain if needed** — Include recent data that reflects the new distribution
5. **Update monitoring** — Adjust thresholds if drift was benign
""")

st.divider()

# ===== Raw Data =====
with st.expander("View Raw Data"):
    st.dataframe(df.round(3), use_container_width=True)