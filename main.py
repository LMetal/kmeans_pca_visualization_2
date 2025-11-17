import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go

class Centroid:
    def __init__(self, nf):
        self.coords = [None]*nf
        self.nf = nf
        
    def move_to(self, new_coords):
        tmp = self.coords
        self.coords = new_coords
        if tmp[0] != None:
            return compute_distance(tmp, new_coords, NF,2)
    
    def get_coords(self):
        return self.coords

st.set_page_config(page_title="PCA & Clustering", layout="wide")
st.title("📊 Visualizzazione PCA multipla")

# Sidebar options
st.sidebar.title("Opzioni")
cluster_choice = st.sidebar.radio(
    "Seleziona numero di cluster:",
    ("2 Clusters", "3 Clusters", "4 Clusters")
)
show_centroids = st.sidebar.checkbox("Mostra centroidi", value=True)
point_size = st.sidebar.slider("Dimensione punti", 3, 12, 6)
centroid_size = st.sidebar.slider("Dimensione centroidi", 8, 20, 14)
legend_font_size = st.sidebar.slider("Dimensione legenda", 12, 30, 18)

# Map selection to pickle files (adjust paths/names as needed)
pickle_map = {
    "2 Clusters": ("kmeans_results_countries_2cl.pkl", "kmeans_results_countries_2cl_sklearn.pkl"),
    "3 Clusters": ("kmeans_results_countries_3cl.pkl", "kmeans_results_countries_3cl_sklearn.pkl"),
    "4 Clusters": ("kmeans_results_countries_4cl.pkl", "kmeans_results_countries_4cl_sklearn.pkl")
}

file_custom, file_sklearn = pickle_map[cluster_choice]

def load_data(pickle_file):
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    return data

def plot_pca_3d(data, title, custom):
    components = data["X"]
    centroids = data["best_optimized_centroids"]
    cluster_labels = data["cluster_labels"]
    countries = data["countries"]

    # PCA transform
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    components_pca = pca.fit_transform(components)

    if custom:
        centroid_coords = np.array([pca.transform(np.array([c.get_coords()]))[0] for c in centroids])
    else:
        centroid_coords = np.array([pca.transform(c.reshape(1, -1))[0] for c in centroids])

    if custom:
        centroid_array = np.array([c.get_coords() for c in data["best_optimized_centroids"]])  # nello spazio originale
    else:
        centroid_array = data["best_optimized_centroids"]

    # Labels and assignments
    X = components
    dists = np.linalg.norm(X[:, None, :] - centroid_array[None, :, :], axis=2)
    labels_assign = np.argmin(dists, axis=1)
    
    n_clusters = centroid_coords_pca.shape[0]

    fig = go.Figure()

    for cluster_id in range(n_clusters):
        class_name = cluster_labels[cluster_id]
        legend_name = f"{class_name} (cluster {cluster_id})"
        points_idx = np.where(labels_assign == cluster_id)[0]

        fig.add_trace(go.Scatter3d(
            x=components_pca[points_idx, 0],
            y=components_pca[points_idx, 1],
            z=components_pca[points_idx, 2],
            mode="markers",
            marker=dict(size=point_size, opacity=0.8),
            name=legend_name,
            text=[countries[i] for i in points_idx]
        ))

    if show_centroids:
        fig.add_trace(go.Scatter3d(
            x=centroid_coords_pca[:, 0],
            y=centroid_coords_pca[:, 1],
            z=centroid_coords_pca[:, 2],
            mode="markers",
            marker=dict(size=centroid_size, color="red", symbol="x"),
            name="Centroidi"
        ))

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        height=600,
        legend=dict(font=dict(size=legend_font_size), orientation="h", xanchor="center", x=0.5, yanchor="bottom", y=-0.2)
    )
    return fig

# Load data
data_custom = load_data(file_custom)
data_sklearn = load_data(file_sklearn)

# Display two plots: custom on top, sklearn on bottom
st.subheader(f"PCA 3D - Clustering Custom ({cluster_choice})")
fig_custom = plot_pca_3d(data_custom, f"PCA 3D - Custom Clustering ({cluster_choice})", True)
st.plotly_chart(fig_custom, use_container_width=True)

st.subheader(f"PCA 3D - Clustering Sklearn ({cluster_choice})")
fig_sklearn = plot_pca_3d(data_sklearn, f"PCA 3D - Sklearn Clustering ({cluster_choice})", False)
st.plotly_chart(fig_sklearn, use_container_width=True)
