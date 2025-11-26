import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# -----------------------
# LOAD IRIS DATA
# -----------------------
iris = load_iris()
X = iris.data[:, :2]    # use first 2 features so plot is visible

# Standardize for better clustering
sc = StandardScaler()
X_std = sc.fit_transform(X)

# -----------------------
# CREATE 3-PLOT FIGURE
# -----------------------
fig, axs = plt.subplots(3, 1, figsize=(8, 18))
fig.subplots_adjust(hspace=0.4)

# -----------------------
# PLOT 1: Original Iris Data
# -----------------------
axs[0].scatter(X_std[:, 0], X_std[:, 1], s=50)
axs[0].set_title("Iris Dataset (First 2 Features)")
axs[0].set_xlabel("Sepal Length (standardized)")
axs[0].set_ylabel("Sepal Width (standardized)")

# -----------------------
# APPLY K-MEANS
# -----------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_std)
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# -----------------------
# PLOT 2: K-Means Clustering
# -----------------------
axs[1].scatter(X_std[:, 0], X_std[:, 1], c=labels, cmap='viridis', s=50, alpha=0.8)
axs[1].scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centers')
axs[1].set_title("K-Means Clustering Results (k=3)")
axs[1].set_xlabel("Sepal Length (standardized)")
axs[1].set_ylabel("Sepal Width (standardized)")
axs[1].legend()

# -----------------------
# PLOT 3: Elbow Method
# -----------------------
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans_elbow = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_elbow.fit(X_std)
    inertia.append(kmeans_elbow.inertia_)

axs[2].plot(k_range, inertia, marker='o')
axs[2].set_title("Elbow Method for Optimal k")
axs[2].set_xlabel("Number of Clusters (k)")
axs[2].set_ylabel("Inertia")
axs[2].set_xticks(k_range)
axs[2].grid(True)

# -----------------------
# SAVE THE FIGURE
# -----------------------
plt.savefig('iris_kmeans_plots.png', bbox_inches='tight', dpi=150)

# Show all 3 plots
plt.show()
