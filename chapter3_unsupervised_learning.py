"""
Chapter 3: Unsupervised Learning and Preprocessing
Examples and algorithms from "Introduction to Machine Learning with Python" by Andreas C. Müller & Sarah Guido.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn

from sklearn.datasets import load_breast_cancer, load_digits, make_blobs, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
from sklearn.svm import SVC
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics.cluster import adjusted_rand_score, silhouette_score
from scipy.cluster.hierarchy import dendrogram, ward

# Set non-blocking plot display mode for headless/script execution
plt.ion()

print("="*60)
print("CHAPTER 3: UNSUPERVISED LEARNING AND PREPROCESSING")
print("="*60)

# ---------------------------------------------------------
# 1. Preprocessing and Scaling
# ---------------------------------------------------------
print("\n--- 1. Preprocessing and Scaling ---")

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)

print(f"Original training data shape: {X_train.shape}")
print(f"Original test data shape: {X_test.shape}")

# MinMaxScaler
minmax_scaler = MinMaxScaler().fit(X_train)
X_train_minmax = minmax_scaler.transform(X_train)
X_test_minmax = minmax_scaler.transform(X_test)

svm_raw = SVC(C=100).fit(X_train, y_train)
print(f"Unscaled SVC test accuracy: {svm_raw.score(X_test, y_test):.2f}")

svm_minmax = SVC(C=100).fit(X_train_minmax, y_train)
print(f"MinMaxScaler + SVC test accuracy: {svm_minmax.score(X_test_minmax, y_test):.2f}")

# StandardScaler
std_scaler = StandardScaler().fit(X_train)
X_train_std = std_scaler.transform(X_train)
X_test_std = std_scaler.transform(X_test)

svm_std = SVC(C=100).fit(X_train_std, y_train)
print(f"StandardScaler + SVC test accuracy: {svm_std.score(X_test_std, y_test):.2f}")


# ---------------------------------------------------------
# 2. Dimensionality Reduction, Feature Extraction & Manifold Learning
# ---------------------------------------------------------
print("\n--- 2. Dimensionality Reduction & Manifold Learning ---")

# Principal Component Analysis (PCA)
scaler = StandardScaler().fit(cancer.data)
X_scaled = scaler.transform(cancer.data)

pca = PCA(n_components=2).fit(X_scaled)
X_pca = pca.transform(X_scaled)
print(f"Cancer original shape: {X_scaled.shape}")
print(f"Cancer PCA reduced shape: {X_pca.shape}")
print(f"PCA components shape: {pca.components_.shape}")

# Non-Negative Matrix Factorization (NMF) on Synthetic Signals
S_signals = mglearn.datasets.make_signals()
A_signals = np.random.RandomState(0).uniform(size=(100, 3))
X_signals = np.dot(S_signals, A_signals.T)

nmf = NMF(n_components=3, random_state=42)
S_nmf = nmf.fit_transform(X_signals)
print(f"NMF input signals shape: {X_signals.shape}")
print(f"NMF recovered signals shape: {S_nmf.shape}")

# Manifold Learning with t-SNE on Digits Dataset
digits = load_digits()
tsne = TSNE(random_state=42)
digits_tsne = tsne.fit_transform(digits.data)
print(f"Digits original shape: {digits.data.shape}")
print(f"Digits t-SNE shape: {digits_tsne.shape}")


# ---------------------------------------------------------
# 3. Clustering
# ---------------------------------------------------------
print("\n--- 3. Clustering ---")

# k-Means Clustering
X_blobs, y_blobs = make_blobs(random_state=1)
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X_blobs)
print(f"k-Means cluster labels on blobs dataset:\n{kmeans.labels_}")
print(f"k-Means cluster centers shape: {kmeans.cluster_centers_.shape}")

# Vector Quantization with k-Means on make_moons
X_moons, y_moons = make_moons(n_samples=200, noise=0.05, random_state=0)
kmeans_10 = KMeans(n_clusters=10, random_state=0).fit(X_moons)
distance_features = kmeans_10.transform(X_moons)
print(f"Vector Quantization 10-cluster distance feature shape: {distance_features.shape}")

# Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=3)
agg_labels = agg.fit_predict(X_blobs)
print(f"Agglomerative Clustering cluster labels:\n{agg_labels}")

# Hierarchical Clustering Linkage with SciPy
X_12, _ = make_blobs(random_state=0, n_samples=12)
linkage_array = ward(X_12)
print(f"SciPy Ward Linkage array shape: {linkage_array.shape}")

# DBSCAN
X_moons_scaled = StandardScaler().fit_transform(X_moons)
dbscan = DBSCAN().fit(X_moons_scaled)
print(f"DBSCAN cluster labels on scaled moons dataset:\n{dbscan.labels_}")
print(f"Unique DBSCAN cluster labels: {np.unique(dbscan.labels_)}")

# ---------------------------------------------------------
# 4. Comparing and Evaluating Clustering Algorithms
# ---------------------------------------------------------
print("\n--- 4. Clustering Evaluation (ARI & Silhouette Score) ---")

algorithms = [
    KMeans(n_clusters=2, random_state=0),
    AgglomerativeClustering(n_clusters=2),
    DBSCAN(eps=0.5)
]

for alg in algorithms:
    labels = alg.fit_predict(X_moons_scaled)
    ari = adjusted_rand_score(y_moons, labels)
    sil = silhouette_score(X_moons_scaled, labels)
    print(f"{alg.__class__.__name__:25s} -> Adjusted Rand Index (ARI): {ari:.2f}, Silhouette Score: {sil:.2f}")

print("\nFinished running Chapter 3 code successfully!")
