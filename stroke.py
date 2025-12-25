import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="Stroke Dataset EDA + PCA", page_icon="🧠", layout="wide")

# ------------------------------- Custom CSS
st.markdown("""
<style>
.card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 22px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.5);
    margin-bottom: 25px;
}
.stButton > button {
    background: linear-gradient(135deg, #4ade80, #16a34a);
    color: #020617;
    border-radius: 14px;
    padding: 12px 28px;
    font-weight: 700;
    border: none;
    font-size: 1rem;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: scale(1.08);
    box-shadow: 0 12px 35px rgba(74,222,128,0.6);
}
h2 {
    background: linear-gradient(90deg, #4ade80, #16a34a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 900;
}
[data-testid="stDataFrameContainer"] {
    overflow-x: auto;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------- Header
st.markdown("<div class='card'><h2>🧠 Stroke Dataset EDA + PCA</h2></div>", unsafe_allow_html=True)
st.markdown("Upload your Stroke CSV to explore **missing values, distributions, and PCA** interactively.")

# ------------------------------- File Upload
uploaded_file = st.file_uploader("Upload Stroke Dataset CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.markdown("<div class='card'><h3>📄 Dataset Overview</h3></div>", unsafe_allow_html=True)
    st.write(f"Shape: {df.shape}")
    st.write(f"Columns: {df.columns.tolist()}")
    
    if st.button("👀 Show Raw Dataset"):
        st.dataframe(df.head())
    
    # ------------------------------- Missing Values
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing_count, 'Missing %': missing_percent})
    
    st.markdown("<div class='card'><h3>🛠 Missing Values Summary</h3></div>", unsafe_allow_html=True)
    st.dataframe(missing_df)
    
    # Missing % barplot
    if st.button("📊 Show Missing % Barplot"):
        fig, ax = plt.subplots(figsize=(12,6))
        sns.barplot(x=missing_df.index, y="Missing %", data=missing_df, palette="viridis", ax=ax)
        ax.set_title("Percentage of Missing Values per Column")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
    
    # Missing heatmap
    if st.button("🗺 Show Missing Values Heatmap"):
        fig, ax = plt.subplots(figsize=(12,6))
        sns.heatmap(df.isnull(), cbar=False, cmap="coolwarm", ax=ax)
        ax.set_title("Missing Values Heatmap")
        st.pyplot(fig)
    
    # ------------------------------- Numeric Columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if st.button(f"📈 Show Distribution of {col}"):
            fig, ax = plt.subplots(figsize=(10,5))
            sns.histplot(df[col], bins=40, kde=True, color="dodgerblue", ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)
    
    # ------------------------------- Categorical Columns
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if st.button(f"📊 Show Frequency of {col}"):
            fig, ax = plt.subplots(figsize=(12,5))
            df[col].value_counts(dropna=False).plot(kind="bar", color="teal", ax=ax)
            ax.set_title(f"Frequency of Categories in {col}")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)
    
    # ------------------------------- PCA
    if st.button("📐 Perform PCA"):
        st.markdown("<div class='card'><h3>📊 PCA Analysis</h3></div>", unsafe_allow_html=True)
        
        # Fill missing numeric values
        df_numeric = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_numeric)
        
        # PCA 3 components
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(X_pca, columns=["PC1","PC2","PC3"])
        st.dataframe(pca_df.head())
        
        # 2D PCA Plot
        if st.button("📊 Show PCA 2D Plot"):
            fig, ax = plt.subplots(figsize=(8,6))
            scatter = ax.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.7, c='blue')
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("PCA 2D Projection")
            st.pyplot(fig)
        
        # 3D PCA Plot
        if st.button("🧊 Show PCA 3D Plot"):
            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(pca_df["PC1"], pca_df["PC2"], pca_df["PC3"], alpha=0.7, c='blue')
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
            ax.set_title("PCA 3D Projection")
            st.pyplot(fig)
