import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Load models
scaler = pickle.load(open("model config/robust_scaler.pkl", "rb"))
pca_cluster_df = pd.read_csv("model config/pca_cluster_points.csv")
pca = pickle.load(open("model config/pca_model.pkl", "rb"))
kmeans = pickle.load(open("model config/kmeans_model.pkl", "rb"))

# Cluster label dictionary
cluster_labels = {
    0: "High-Value Multichannel Power Users (Highly Engaged)",
    1: "Low-Engagement Premium Spenders",
    2: "Mid-Value Swipe-Only Traditionalists",
    3: "High-Value In-Person Spenders"
}

# Cluster descriptions for display
cluster_descriptions = {
    0: """
• Very recent and frequent activity  
• Highest total spend but small average transactions  
• Heavy chip + online usage, moderate swipe  
• Broad merchant variety  
""",
    
    1: """
• Infrequent activity and longer recency  
• Low overall spend but high average transaction value  
• Almost no online usage, low merchant diversity  
• Low engagement across all channels  
""",
    
    2: """
• Moderately recent activity  
• Mid-level frequency and spend  
• Swipe-focused behavior, zero chip use  
• Occasional online usage (some users)  
""",
    
    3: """
• Active recently and frequently  
• Strong total spend with moderate transaction sizes  
• Heavy chip usage, almost no online  
• Broad merchant variety and stable in-person behavior  
"""
}

# Page setup
st.set_page_config(page_title="Customer Transactional Behavior Segmentation", page_icon="💳", layout="centered")
st.title("Customer Transactional Behavior Segmentation Web App")
st.markdown("Enter customer transaction behavior details to assign into a segment:")

# Form Input
with st.form("input_form"):
    days_since_last_txn = st.number_input("Days Since Last Transaction", min_value=0, step=1)
    active_days = st.number_input("Active Days", min_value=0, step=1)
    total_amount = st.number_input("Total Amount Spent", min_value=0.0, step=10.0)
    num_unique_merchants = st.number_input("Number of Unique Merchants", min_value=0, step=1)
    num_chip_txn = st.number_input("Number of Chip Transactions", min_value=0, step=10)
    num_online_txn = st.number_input("Number of Online Transactions", min_value=0, step=10)
    num_swipe_txn = st.number_input("Number of Swipe Transactions", min_value=0, step=10)

    submitted = st.form_submit_button("Assign Segment")

if submitted:
    num_txn = num_chip_txn + num_online_txn + num_swipe_txn
    avg_txn_amount = total_amount / num_txn if num_txn > 0 else 0.0
    # 1. Create input DataFrame
    input_data = pd.DataFrame([{
        "days_since_last_txn": days_since_last_txn,
        "active_days": active_days,
        "num_txn": num_txn,
        "total_amount": total_amount,
        "num_unique_merchants": num_unique_merchants,
        "num_chip_txn": num_chip_txn,
        "num_online_txn": num_online_txn,
        "num_swipe_txn": num_swipe_txn,
        "avg_txn_amount": avg_txn_amount
    }])

    # 2. Apply log1p transformation
    log_transformed = np.log1p(input_data)

    # 3. Apply robust scaler
    scaled = scaler.transform(log_transformed)

    # 5. Predict cluster
    cluster = kmeans.predict(scaled)[0]
    label = cluster_labels[cluster]

    # 6. Project the new input into PCA space for visualization
    pca_input = pca.transform(scaled)[0]  # shape: (3,)
    # Step 1: Map cluster number to label using your dictionary
    pca_cluster_df["label"] = pca_cluster_df["cluster"].map(cluster_labels)

    # Step 2: Convert to ordered categorical using your dict values
    ordered_labels = [cluster_labels[i] for i in sorted(cluster_labels.keys())]
    pca_cluster_df["label"] = pd.Categorical(
        pca_cluster_df["label"],
        categories=ordered_labels,
        ordered=True
    )

    # 7. Distance logic
    assigned_center = kmeans.cluster_centers_[cluster]
    distance = np.linalg.norm(scaled[0] - assigned_center)


    # Output
    st.success(f"🧩 Assigned to: **Cluster {cluster} ({label})**")
    st.subheader("💬 Cluster Description")
    st.markdown(cluster_descriptions[cluster])


    st.markdown(f"📏 **Distance to Cluster Center**: `{distance:.3f}`")
    if distance < 1.5:
        st.success("✅ This input closely matches the assigned cluster.")
    elif distance < 3.0:
        st.warning("⚠️ This input somewhat matches the assigned cluster.")
    else:
        st.error("🚨 This input is far from any cluster center. The assignment may be unreliable.")
    
    
    tab1, tab2 = st.tabs(["🧾 Input Summary", "📈 Cluster Visualization"])
    with tab1:
        # Input Summary
        st.subheader("🧾 Input Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Days Since Last Transaction", days_since_last_txn)
        col2.metric("Active Days", active_days)
        col3.metric("Numbers of Transaction", num_txn)

        col1.metric("Total Amount", f"${total_amount:,.2f}")
        col2.metric("Avg Transaction Amount", f"${avg_txn_amount:,.2f}")
        col3.metric("Unique Merchants", num_unique_merchants)
    
    with tab2:
        # Cluster Visualization
        st.subheader("📈 Cluster Visualization")
        fig = go.Figure()
        for clust_num, clust_label in cluster_labels.items():
            cluster_data = pca_cluster_df[pca_cluster_df["cluster"] == clust_num]
            fig.add_trace(go.Scatter3d(
                x=cluster_data["x"],
                y=cluster_data["y"],
                z=cluster_data["z"],
                mode='markers',
                name=clust_label,
                marker=dict(
                    color=px.colors.qualitative.Plotly[clust_num % len(px.colors.qualitative.Plotly)],
                    opacity=0.1 if clust_num != cluster else 0.9
                ),
            ))

        # Add the new input point
        fig.add_trace(go.Scatter3d(
            x=[pca_input[0]],
            y=[pca_input[1]],
            z=[pca_input[2]],
            mode='markers+text',
            marker=dict(size=6, symbol='x', color='black'),
            name="New Input",
            text=["New Input"],
            textposition='top center'
        ))

        # Layout
        fig.update_layout(
            title="PCA Cluster Visualization",
            legend_title="Cluster",
            scene=dict(
                xaxis_title="PCA 1",
                yaxis_title="PCA 2",
                zaxis_title="PCA 3"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
                font=dict(size=10)
            ),
            margin=dict(t=40, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)