# streamlit_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Title
st.title("Global Development Clustering App")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Select features for clustering
    st.subheader("Select Features for Clustering")
    features = st.multiselect("Choose numeric features", df.select_dtypes(include=['float64','int64']).columns.tolist())

    if features:
        X = df[features]

        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Number of clusters
        n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)

        # KMeans model
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X_scaled)

        st.subheader("Clustered Data")
        st.dataframe(df.head())

        # Plot clusters if 2 features selected
        if len(features) == 2:
            plt.figure(figsize=(8,6))
            plt.scatter(df[features[0]], df[features[1]], c=df['Cluster'], cmap='viridis')
            plt.xlabel(features[0])
            plt.ylabel(features[1])
            plt.title("KMeans Clusters")
            st.pyplot(plt)
        else:
            st.write("Select exactly 2 features to visualize clusters.")

        # Download clustered data
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Clustered Data as CSV",
            data=csv,
            file_name='clustered_data.csv',
            mime='text/csv',
        )
