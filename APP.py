import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import pickle

st.set_page_config(layout="wide", page_title="World Development Explorer")

st.title("World Development — Explorer & Clustering App")
st.markdown("Upload your `World_development_mesurement.xlsx` file or use the default filename. The app will run preprocessing, PCA and provide clustering visualizations and evaluation metrics.")

# --- File upload / load
uploaded_file = st.file_uploader("Upload Excel file (or leave empty to load from disk)", type=["xlsx","xls","csv"])
use_default = False
if uploaded_file is None:
    st.info("No file uploaded — the app will try to load `World_development_mesurement.xlsx` from the working directory.")
    try:
        df = pd.read_excel("World_development_mesurement.xlsx")
        use_default = True
    except Exception as e:
        st.error("No uploaded file and default file not found. Please upload your file to continue.")
        st.stop()
else:
    # support csv as well
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

st.write("**Initial dataframe shape:**", df.shape)

if st.checkbox("Show raw data (first 5 rows)"):
    st.dataframe(df.head())

# --- Preprocessing (based on user's script)
st.header("Preprocessing")

# copy to avoid changing original
df_clean = df.copy()

# Drop 'Ease of Business' if present (user dropped it)
if 'Ease of Business' in df_clean.columns:
    df_clean.drop(columns=['Ease of Business'], inplace=True)

# columns that may contain formatted numbers
format_cols = ['Business Tax Rate', 'GDP', 'Health Exp/Capita', 'Tourism Inbound', 'Tourism Outbound']
for col in format_cols:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].astype(str).str.replace(',','')
        df_clean[col] = df_clean[col].str.replace('%','')
        df_clean[col] = df_clean[col].str.replace('$','')
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Fill missing values: mode for objects, median for numerics
for col in df_clean.columns:
    if df_clean[col].dtype == 'O':
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0])
    else:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

st.write("After filling missing values — nulls per column:")
st.dataframe(df_clean.isnull().sum())

# Show numeric / categorical
numerical_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = df_clean.select_dtypes(include=['object']).columns.tolist()
st.write(f"Numeric columns ({len(numerical_columns)}): {numerical_columns}")
st.write(f"Categorical columns ({len(categorical_columns)}): {categorical_columns}")

# Label encode Country if present
if 'Country' in df_clean.columns:
    le = LabelEncoder()
    df_clean['country_encoded'] = le.fit_transform(df_clean['Country'])
    countries = df_clean['Country'].copy()
    df_no_country = df_clean.drop(columns=['Country'])
else:
    df_no_country = df_clean.copy()

# final numeric columns after drop
numeric_cols_final = df_no_country.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols_final) == 0:
    st.error("No numeric columns available for scaling / PCA. Please check your file.")
    st.stop()

# Scaling
st.header("Scaling & PCA")
scale = st.checkbox("Apply RobustScaler (recommended)", value=True)
if scale:
    scaler = RobustScaler()
    df_scaled = df_no_country.copy()
    df_scaled[numeric_cols_final] = scaler.fit_transform(df_no_country[numeric_cols_final])
else:
    df_scaled = df_no_country.copy()

st.write("Scaled data preview")
st.dataframe(df_scaled[numeric_cols_final].head())

# PCA interactive
max_components = min(df_scaled.shape[0], df_scaled.shape[1])
n_components = st.slider("Number of PCA components (for dimensionality reduction)", min_value=2, max_value=max_components, value=min(10, max_components))

pca = PCA(n_components=n_components)
pca_data = pca.fit_transform(df_scaled[numeric_cols_final])
explained = np.cumsum(pca.explained_variance_ratio_)

fig, ax = plt.subplots()
ax.plot(range(1, len(explained)+1), explained, marker='o', linestyle='--')
ax.set_xlabel("Number of Components")
ax.set_ylabel("Cumulative Explained Variance")
ax.set_title("PCA Explained Variance")
ax.grid(True)
st.pyplot(fig)

st.write(f"Explained variance by selected components: {explained[-1]:.4f}")

# Components table
components_df = pd.DataFrame(pca.components_, columns=numeric_cols_final, index=[f'PC{i+1}' for i in range(pca.n_components_)])
if st.checkbox("Show PCA components (feature loadings)"):
    st.dataframe(components_df)

# Top features per component
if st.checkbox("Show top 3 features per principal component"):
    top_feats = {}
    for pc in components_df.index:
        top_feats[pc] = components_df.loc[pc].abs().sort_values(ascending=False).head(3)
    df_top = pd.DataFrame(top_feats)
    st.dataframe(df_top)

# Prepare PCA dataframe for plotting
pca_df = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(pca_data.shape[1])])

# --- Clustering
st.header("Clustering — KMeans / Agglomerative / DBSCAN")
cluster_method = st.selectbox("Choose clustering method", options=["KMeans","Agglomerative","DBSCAN"], index=0)

# Controls
if cluster_method == 'KMeans':
    k = st.slider("Number of clusters (K)", min_value=2, max_value=10, value=3)
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(df_scaled[numeric_cols_final])
elif cluster_method == 'Agglomerative':
    k = st.slider("Number of clusters (n_clusters)", min_value=2, max_value=10, value=3)
    model = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = model.fit_predict(df_scaled[numeric_cols_final])
else:
    eps = st.slider("DBSCAN eps (radius)", min_value=0.1, max_value=50.0, value=5.0)
    min_samples = st.slider("DBSCAN min_samples", min_value=3, max_value=50, value=10)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(df_scaled[numeric_cols_final])

st.write(f"Cluster label counts:\n{pd.Series(labels).value_counts().sort_index().to_dict()}")

# Silhouette (only if more than 1 cluster and not all noise)
if len(set(labels)) > 1 and (not (cluster_method=='DBSCAN' and set(labels)=={-1})):
    try:
        sil = silhouette_score(df_scaled[numeric_cols_final], labels)
        st.write(f"Silhouette Score: {sil:.4f}")
    except Exception as e:
        st.warning(f"Could not compute silhouette: {e}")

# Visualize clusters on first two PCs
fig2, ax2 = plt.subplots(figsize=(8,5))
scatter = ax2.scatter(pca_df.iloc[:,0], pca_df.iloc[:,1], c=labels, cmap='tab10', s=50)
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.set_title(f"{cluster_method} Clusters visualized on PC1 vs PC2")
plt.colorbar(scatter, ax=ax2, label='Cluster')
st.pyplot(fig2)

# Show cluster summary table
result_df = df_clean.copy()
result_df['cluster'] = labels
if 'country_encoded' in result_df.columns:
    # recover country if we dropped earlier
    pass

if st.checkbox("Show cluster assignment table (first 200 rows)"):
    st.dataframe(result_df.head(200))

# Top countries per cluster (if Country exists)
if 'Country' in df_clean.columns:
    st.subheader("Top countries by cluster (based on first numeric column)")
    topcol = numeric_cols_final[0]
    sel = st.selectbox("Select numeric column for ranking (top countries)", options=numeric_cols_final, index=0)
    cluster_summary = result_df.groupby('cluster').apply(lambda d: d.nlargest(5, sel)[['Country', sel]])
    st.write(cluster_summary)

# Save models and scaler
st.header("Export trained objects")
if st.button("Save model + scaler to disk (agglo/kmeans/scaler)"):
    try:
        # Save only if model exists and scaler applied
        with open('trained_model.pkl','wb') as f:
            pickle.dump(model, f)
        if scale:
            with open('scaler.pkl','wb') as f:
                pickle.dump(scaler, f)
        st.success("Saved trained_model.pkl and scaler.pkl to working directory.")
    except Exception as e:
        st.error(f"Error while saving: {e}")

# Download cluster assignments CSV
if st.button("Download cluster assignments (CSV)"):
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("Click to download", data=csv, file_name='clustered_data.csv', mime='text/csv')

# Quick EDA visuals
st.header("Quick EDA")
if st.checkbox("Show correlation heatmap"):
    fig3, ax3 = plt.subplots(figsize=(10,8))
    sns.heatmap(df_clean.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)

if st.checkbox("Show histograms for numeric columns (first 10)"):
    cols_hist = numerical_columns[:10]
    for col in cols_hist:
        fig4, ax4 = plt.subplots()
        sns.histplot(df_clean[col], kde=True, ax=ax4)
        ax4.set_title(f"Histogram: {col}")
        st.pyplot(fig4)

st.markdown("---")
st.write("**Notes & next steps:**\n- You can change clustering method and parameters in the sidebar controls.\n- Use the export buttons to save models and download results.\n- If you'd like, I can convert this to a Dockerfile or a Streamlit Cloud deployment configuration.")

st.caption("App generated from user's analysis script. If any columns have unexpected formats, re-upload a cleaned file or ask for additional cleaning rules.")
