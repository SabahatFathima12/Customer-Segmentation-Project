import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Page Config

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide"
)

# Load Dataset

@st.cache_data
def load_data():
    df = pd.read_csv("Mall_Customers.csv")
    return df

df = load_data()

# Title

st.title("🛍️ Customer Segmentation Dashboard")

st.markdown("""
This dashboard segments customers based on:
- Annual Income
- Spending Score
""")

# Sidebar

st.sidebar.header("🔍 Filters")

gender_filter = st.sidebar.multiselect(
    "Select Gender",
    options=df["Gender"].unique(),
    default=df["Gender"].unique()
)

filtered_df = df[df["Gender"].isin(gender_filter)]

# KPI Metrics

total_customers = len(filtered_df)
avg_income = filtered_df["Annual Income (k$)"].mean()
avg_spending = filtered_df["Spending Score (1-100)"].mean()

col1, col2, col3 = st.columns(3)

col1.metric("👥 Total Customers", total_customers)
col2.metric("💰 Avg Income", f"${avg_income:.2f}k")
col3.metric("🛒 Avg Spending Score", f"{avg_spending:.2f}")

st.markdown("---")

# Elbow Method

st.subheader("📈 Elbow Method")

X = filtered_df[["Annual Income (k$)", "Spending Score (1-100)"]]

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

fig_elbow = px.line(
    x=range(1, 11),
    y=wcss,
    markers=True,
    labels={"x": "Number of Clusters", "y": "WCSS"},
    title="Elbow Method for Optimal Clusters"
)

st.plotly_chart(fig_elbow, use_container_width=True)

# Apply KMeans Clustering

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)

filtered_df["Cluster"] = kmeans.fit_predict(X)

# Cluster Visualization

st.subheader("🎯 Customer Segments")

fig_clusters = px.scatter(
    filtered_df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color="Cluster",
    hover_data=["CustomerID", "Age", "Gender"],
    title="Customer Segmentation using K-Means",
    size="Age"
)

st.plotly_chart(fig_clusters, use_container_width=True)

# Cluster Distribution

st.subheader("📊 Cluster Distribution")

cluster_counts = filtered_df["Cluster"].value_counts().reset_index()
cluster_counts.columns = ["Cluster", "Count"]

fig_bar = px.bar(
    cluster_counts,
    x="Cluster",
    y="Count",
    color="Cluster",
    title="Number of Customers in Each Cluster"
)

st.plotly_chart(fig_bar, use_container_width=True)

# Dataset View

st.subheader("📋 Customer Data")

st.dataframe(filtered_df, use_container_width=True)

# Insights

st.markdown("---")

st.subheader("📌 Key Insights")

highest_cluster = cluster_counts.sort_values(
    by="Count",
    ascending=False
).iloc[0]["Cluster"]

st.write(f"• Cluster {highest_cluster} has the highest number of customers")
st.write("• High income + high spending customers are premium customers")
st.write("• Low spending customers may need targeted marketing")
st.write("• Customer segmentation helps businesses improve marketing strategies")
