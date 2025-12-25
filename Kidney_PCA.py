import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Kidney PCA",
    page_icon="🩸",
    layout="wide"
)

# --------------------------------------------------
# Premium Dark UI CSS
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #020617, #0f172a);
    color: #e5e7eb;
}
.card {
    background: rgba(255,255,255,0.06);
    padding: 26px;
    border-radius: 22px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.45);
    margin-bottom: 26px;
    border: 1px solid rgba(255,255,255,0.08);
}
h2 {
    font-size: 2.2rem;
    font-weight: 900;
    background: linear-gradient(90deg, #4ade80, #16a34a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stButton > button {
    background: linear-gradient(135deg, #4ade80, #16a34a);
    color: #020617;
    border-radius: 16px;
    padding: 12px 28px;
    font-weight: 800;
    font-size: 1rem;
    border: none;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.05);
    box-shadow: 0 14px 35px rgba(74,222,128,0.6);
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("""
<div class="card">
<h2>🩸 Kidney Dataset — PCA Analysis</h2>
<p>
This page performs full preprocessing and <b>Manual Principal Component Analysis</b>
on the Kidney dataset with professional 2D & 3D visualizations.
</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Upload CSV
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Kidney Dataset CSV",
    type=["csv"]
)

# --------------------------------------------------
# Manual PCA Function
# --------------------------------------------------
def manual_pca(X, n_components=3):
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, idx][:, :n_components]
    return np.dot(X_centered, components)

# --------------------------------------------------
# Main Processing
# --------------------------------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ------------------ Raw Preview
    if st.button("👀 Show Raw Dataset"):
        st.dataframe(df.head(10))

    # ------------------ Cleaning
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    for col in df.select_dtypes(include="object"):
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace(["", " ", "?", "nan", "none", "-", "--"], np.nan)
        )

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.replace(r"[^0-9\.-]", "", regex=True)
            df[col] = pd.to_numeric(df[col], errors="ignore")

    # ------------------ Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Separate continuous vs binary numeric columns
    continuous_cols = [c for c in numeric_cols if df[c].nunique() > 2]
    binary_cols = [c for c in numeric_cols if df[c].dropna().nunique() == 2]

    # ------------------ Remove outliers only from continuous columns
    for col in continuous_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].mask(
            (df[col] < Q1 - 1.5 * IQR) |
            (df[col] > Q3 + 1.5 * IQR)
        )

    # ------------------ Fill missing values
    for col in continuous_cols:
        df[col] = df[col].fillna(df[col].mean())
    for col in binary_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    df = df.drop_duplicates()

    # Encode any remaining categorical columns
    for col in df.select_dtypes(include="object"):
        df[col] = LabelEncoder().fit_transform(df[col])

    # Drop Medication column if present
    if "Medication" in df.columns:
        df = df.drop(columns=["Medication"])

    # ------------------ Cleaned Preview
    if st.button("🧼 Show Cleaned Dataset"):
        st.dataframe(df.head(10))

    # ------------------ PCA Preparation
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[continuous_cols])

    # ------------------ PCA (PC1, PC2, PC3)
    pca_data = manual_pca(scaled_data, 3)
    pca_df = pd.DataFrame(pca_data, columns=["PC1", "PC2", "PC3"])

    if st.button("📐 Show PCA Dataset"):
        st.dataframe(pca_df.head(10))

    # ------------------ PCA 2D Plot
    if st.button("📊 Show PCA 2D Plot"):
        fig, ax = plt.subplots(figsize=(8,6))
        c = df["ckd_status"] if "ckd_status" in df.columns else "blue"
        ax.scatter(pca_df["PC1"], pca_df["PC2"], c=c, cmap="viridis", alpha=0.7)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA 2D Visualization")
        st.pyplot(fig)

    # ------------------ PCA 3D Plot
    if st.button("🧊 Show PCA 3D Plot"):
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection="3d")
        c = df["ckd_status"] if "ckd_status" in df.columns else "blue"
        ax.scatter(
            pca_df["PC1"],
            pca_df["PC2"],
            pca_df["PC3"],
            c=c,
            cmap="viridis",
            alpha=0.7
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title("PCA 3D Visualization")
        st.pyplot(fig)

    # ------------------ PCA Boxplot
    if st.button("📦 Show PCA Boxplots"):
        fig, ax = plt.subplots(figsize=(8,5))
        ax.boxplot(
            [pca_df["PC1"], pca_df["PC2"], pca_df["PC3"]],
            labels=["PC1", "PC2", "PC3"]
        )
        ax.set_title("Distribution of PCA Components")
        st.pyplot(fig)
