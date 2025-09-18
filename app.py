# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import io

# Title
st.title("🌍 Global Development Clustering App")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Feature selection
    st.subheader("Select Features for Clustering")
    features = st.multiselect("Choose numeric features", df.select_dtypes(include=['float64','int64']).columns.tolist())

    if features:
        X = df[features]

        # Handle missing values (replace NaN with column mean)
        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(X)

        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        # Elbow method (SSE)
        sse = []
        for k in range(2, 11):
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(X_scaled)
            sse.append(km.inertia_)

        st.subheader("Elbow Method (SSE vs K)")
        fig, ax = plt.subplots()
        ax.plot(range(2, 11), sse, marker="o")
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel("SSE (Inertia)")
        ax.set_title("Elbow Method")
        st.pyplot(fig)

        # Choose k
        n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)

        # KMeans model
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X_scaled)

        st.subheader("Clustered Data")
        st.dataframe(df.head())

        # Plot clusters if 2 features are selected
        if len(features) == 2:
            fig2, ax2 = plt.subplots()
            scatter = ax2.scatter(df[features[0]], df[features[1]], c=df['Cluster'], cmap='viridis')
            ax2.set_xlabel(features[0])
            ax2.set_ylabel(features[1])
            ax2.set_title("KMeans Clusters")
            st.pyplot(fig2)
        else:
            st.info("Select exactly 2 features to visualize clusters.")

        # Download clustered data
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Clustered Data",
            data=csv,
            file_name="clustered_data.csv",
            mime="text/csv"
        )
