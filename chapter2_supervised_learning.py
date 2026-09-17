"""
Chapter 2: Supervised Learning
Examples and algorithms from "Introduction to Machine Learning with Python" by Andreas C. Müller & Sarah Guido.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer, load_iris, make_blobs, make_moons, make_circles

# Set non-blocking plot display mode for headless/script execution
plt.ion()

print("="*60)
print("CHAPTER 2: SUPERVISED LEARNING")
print("="*60)

# ---------------------------------------------------------
# 1. Sample Datasets
# ---------------------------------------------------------
print("\n--- 1. Sample Datasets ---")

# Forge dataset
X_forge, y_forge = mglearn.datasets.make_forge()
print(f"Forge dataset shape: {X_forge.shape}")

# Wave dataset
X_wave, y_wave = mglearn.datasets.make_wave(n_samples=40)
print(f"Wave dataset shape: {X_wave.shape}")

# Breast Cancer dataset
cancer = load_breast_cancer()
print(f"Breast Cancer dataset shape: {cancer.data.shape}")
print(f"Sample counts per class: {{'benign': {np.bincount(cancer.target)[1]}, 'malignant': {np.bincount(cancer.target)[0]}}}")

# Extended Boston dataset
X_boston, y_boston = mglearn.datasets.load_extended_boston()
print(f"Extended Boston dataset shape: {X_boston.shape}")


# ---------------------------------------------------------
# 2. k-Nearest Neighbors (k-NN)
# ---------------------------------------------------------
print("\n--- 2. k-Nearest Neighbors ---")

# k-NN Classification on Forge dataset
X_train, X_test, y_train, y_test = train_test_split(X_forge, y_forge, random_state=0)
clf_knn = KNeighborsClassifier(n_neighbors=3)
clf_knn.fit(X_train, y_train)
print(f"k-NN (n_neighbors=3) Forge test predictions: {clf_knn.predict(X_test)}")
print(f"k-NN (n_neighbors=3) Forge test accuracy: {clf_knn.score(X_test, y_test):.2f}")

# k-NN on Breast Cancer dataset across different n_neighbors
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=66
)
training_accuracy = []
test_accuracy = []
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train_c, y_train_c)
    training_accuracy.append(clf.score(X_train_c, y_train_c))
    test_accuracy.append(clf.score(X_test_c, y_test_c))

print(f"Breast Cancer KNN n_neighbors=1 -> Train: {training_accuracy[0]:.3f}, Test: {test_accuracy[0]:.3f}")
print(f"Breast Cancer KNN n_neighbors=6 -> Train: {training_accuracy[5]:.3f}, Test: {test_accuracy[5]:.3f}")
print(f"Breast Cancer KNN n_neighbors=10 -> Train: {training_accuracy[9]:.3f}, Test: {test_accuracy[9]:.3f}")

# k-NN Regression on Wave dataset
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_wave, y_wave, random_state=0)
reg_knn = KNeighborsRegressor(n_neighbors=3)
reg_knn.fit(X_train_w, y_train_w)
print(f"k-NN Regressor (n_neighbors=3) test predictions:\n{reg_knn.predict(X_test_w)}")
print(f"k-NN Regressor (n_neighbors=3) test R^2 score: {reg_knn.score(X_test_w, y_test_w):.2f}")


# ---------------------------------------------------------
# 3. Linear Models
# ---------------------------------------------------------
print("\n--- 3. Linear Models ---")

# Linear Regression (Ordinary Least Squares) on Wave dataset
X_wave60, y_wave60 = mglearn.datasets.make_wave(n_samples=60)
X_train_w60, X_test_w60, y_train_w60, y_test_w60 = train_test_split(X_wave60, y_wave60, random_state=42)
lr = LinearRegression().fit(X_train_w60, y_train_w60)
print(f"LinearRegression coef_: {lr.coef_}")
print(f"LinearRegression intercept_: {lr.intercept_:.6f}")
print(f"LinearRegression Wave train score: {lr.score(X_train_w60, y_train_w60):.2f}")
print(f"LinearRegression Wave test score: {lr.score(X_test_w60, y_test_w60):.2f}")

# Linear Regression on Extended Boston (Demonstrating Overfitting)
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_boston, y_boston, random_state=0)
lr_b = LinearRegression().fit(X_train_b, y_train_b)
print(f"LinearRegression Extended Boston train score: {lr_b.score(X_train_b, y_train_b):.2f}")
print(f"LinearRegression Extended Boston test score: {lr_b.score(X_test_b, y_test_b):.2f}")

# Ridge Regression on Extended Boston
ridge = Ridge().fit(X_train_b, y_train_b)
print(f"Ridge (alpha=1.0) train score: {ridge.score(X_train_b, y_train_b):.2f}, test score: {ridge.score(X_test_b, y_test_b):.2f}")
ridge10 = Ridge(alpha=10).fit(X_train_b, y_train_b)
print(f"Ridge (alpha=10) train score: {ridge10.score(X_train_b, y_train_b):.2f}, test score: {ridge10.score(X_test_b, y_test_b):.2f}")
ridge01 = Ridge(alpha=0.1).fit(X_train_b, y_train_b)
print(f"Ridge (alpha=0.1) train score: {ridge01.score(X_train_b, y_train_b):.2f}, test score: {ridge01.score(X_test_b, y_test_b):.2f}")

# Lasso Regression on Extended Boston
lasso = Lasso().fit(X_train_b, y_train_b)
print(f"Lasso (alpha=1.0) train score: {lasso.score(X_train_b, y_train_b):.2f}, test score: {lasso.score(X_test_b, y_test_b):.2f}, features used: {np.sum(lasso.coef_ != 0)}")
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train_b, y_train_b)
print(f"Lasso (alpha=0.01) train score: {lasso001.score(X_train_b, y_train_b):.2f}, test score: {lasso001.score(X_test_b, y_test_b):.2f}, features used: {np.sum(lasso001.coef_ != 0)}")

# Linear Models for Classification (Logistic Regression & LinearSVC)
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42
)
logreg = LogisticRegression(max_iter=5000).fit(X_train_bc, y_train_bc)
print(f"LogisticRegression (C=1.0) train acc: {logreg.score(X_train_bc, y_train_bc):.3f}, test acc: {logreg.score(X_test_bc, y_test_bc):.3f}")
logreg100 = LogisticRegression(C=100, max_iter=5000).fit(X_train_bc, y_train_bc)
print(f"LogisticRegression (C=100) train acc: {logreg100.score(X_train_bc, y_train_bc):.3f}, test acc: {logreg100.score(X_test_bc, y_test_bc):.3f}")

# Multiclass Linear Classifier (One-vs-Rest)
X_blobs, y_blobs = make_blobs(random_state=42)
linear_svm = LinearSVC().fit(X_blobs, y_blobs)
print(f"Multiclass LinearSVC coef_ shape: {linear_svm.coef_.shape}")
print(f"Multiclass LinearSVC intercept_ shape: {linear_svm.intercept_.shape}")


# ---------------------------------------------------------
# 4. Naive Bayes Classifiers
# ---------------------------------------------------------
print("\n--- 4. Naive Bayes Classifiers ---")

# Feature counting logic used in BernoulliNB
X_nb = np.array([[0, 1, 0, 1],
                 [1, 0, 1, 1],
                 [0, 0, 0, 1],
                 [1, 0, 1, 0]])
y_nb = np.array([0, 1, 0, 1])
counts = {}
for label in np.unique(y_nb):
    counts[label] = X_nb[y_nb == label].sum(axis=0)
print(f"BernoulliNB feature counts per class: {counts}")


# ---------------------------------------------------------
# 5. Decision Trees
# ---------------------------------------------------------
print("\n--- 5. Decision Trees ---")

tree_unpruned = DecisionTreeClassifier(random_state=0)
tree_unpruned.fit(X_train_bc, y_train_bc)
print(f"Unpruned DecisionTree train acc: {tree_unpruned.score(X_train_bc, y_train_bc):.3f}, test acc: {tree_unpruned.score(X_test_bc, y_test_bc):.3f}")

tree_pruned = DecisionTreeClassifier(max_depth=4, random_state=0)
tree_pruned.fit(X_train_bc, y_train_bc)
print(f"Pruned (max_depth=4) DecisionTree train acc: {tree_pruned.score(X_train_bc, y_train_bc):.3f}, test acc: {tree_pruned.score(X_test_bc, y_test_bc):.3f}")
print(f"Top feature importance ('{cancer.feature_names[np.argmax(tree_pruned.feature_importances_)]}'): {np.max(tree_pruned.feature_importances_):.3f}")


# ---------------------------------------------------------
# 6. Ensembles of Decision Trees
# ---------------------------------------------------------
print("\n--- 6. Ensembles of Decision Trees ---")

# Random Forest
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(cancer.data, cancer.target, random_state=0)
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train_rf, y_train_rf)
print(f"RandomForest (100 trees) train acc: {rf.score(X_train_rf, y_train_rf):.3f}, test acc: {rf.score(X_test_rf, y_test_rf):.3f}")

# Gradient Boosted Decision Trees
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train_rf, y_train_rf)
print(f"GradientBoosting (default) train acc: {gbrt.score(X_train_rf, y_train_rf):.3f}, test acc: {gbrt.score(X_test_rf, y_test_rf):.3f}")

gbrt_depth1 = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt_depth1.fit(X_train_rf, y_train_rf)
print(f"GradientBoosting (max_depth=1) train acc: {gbrt_depth1.score(X_train_rf, y_train_rf):.3f}, test acc: {gbrt_depth1.score(X_test_rf, y_test_rf):.3f}")


# ---------------------------------------------------------
# 7. Kernelized Support Vector Machines (SVM)
# ---------------------------------------------------------
print("\n--- 7. Kernelized Support Vector Machines ---")

svc_unscaled = SVC().fit(X_train_rf, y_train_rf)
print(f"Unscaled SVC train acc: {svc_unscaled.score(X_train_rf, y_train_rf):.2f}, test acc: {svc_unscaled.score(X_test_rf, y_test_rf):.2f}")

# MinMax Rescaling manually as described in Chapter 2
min_on_training = X_train_rf.min(axis=0)
range_on_training = (X_train_rf - min_on_training).max(axis=0)
X_train_scaled = (X_train_rf - min_on_training) / range_on_training
X_test_scaled = (X_test_rf - min_on_training) / range_on_training

svc_scaled = SVC().fit(X_train_scaled, y_train_rf)
print(f"Scaled SVC (default) train acc: {svc_scaled.score(X_train_scaled, y_train_rf):.3f}, test acc: {svc_scaled.score(X_test_scaled, y_test_rf):.3f}")

svc_c1000 = SVC(C=1000).fit(X_train_scaled, y_train_rf)
print(f"Scaled SVC (C=1000) train acc: {svc_c1000.score(X_train_scaled, y_train_rf):.3f}, test acc: {svc_c1000.score(X_test_scaled, y_test_rf):.3f}")


# ---------------------------------------------------------
# 8. Neural Networks (Multilayer Perceptrons)
# ---------------------------------------------------------
print("\n--- 8. Neural Networks (MLP) ---")

mlp_unscaled = MLPClassifier(random_state=42, max_iter=1000).fit(X_train_rf, y_train_rf)
print(f"Unscaled MLP train acc: {mlp_unscaled.score(X_train_rf, y_train_rf):.2f}, test acc: {mlp_unscaled.score(X_test_rf, y_test_rf):.2f}")

# Standardizing features (mean=0, std=1)
mean_on_train = X_train_rf.mean(axis=0)
std_on_train = X_train_rf.std(axis=0)
X_train_mlp_scaled = (X_train_rf - mean_on_train) / std_on_train
X_test_mlp_scaled = (X_test_rf - mean_on_train) / std_on_train

mlp_scaled = MLPClassifier(max_iter=1000, random_state=0).fit(X_train_mlp_scaled, y_train_rf)
print(f"Scaled MLP (default) train acc: {mlp_scaled.score(X_train_mlp_scaled, y_train_rf):.3f}, test acc: {mlp_scaled.score(X_test_mlp_scaled, y_test_rf):.3f}")

mlp_alpha1 = MLPClassifier(max_iter=1000, alpha=1, random_state=0).fit(X_train_mlp_scaled, y_train_rf)
print(f"Scaled MLP (alpha=1.0) train acc: {mlp_alpha1.score(X_train_mlp_scaled, y_train_rf):.3f}, test acc: {mlp_alpha1.score(X_test_mlp_scaled, y_test_rf):.3f}")


# ---------------------------------------------------------
# 9. Uncertainty Estimates from Classifiers
# ---------------------------------------------------------
print("\n--- 9. Uncertainty Estimates from Classifiers ---")

X_circ, y_circ = make_circles(noise=0.25, factor=0.5, random_state=1)
y_named = np.array(["blue", "red"])[y_circ]

X_train_circ, X_test_circ, y_train_named, y_test_named, y_train_c2, y_test_c2 = train_test_split(
    X_circ, y_named, y_circ, random_state=0
)

gbrt_circ = GradientBoostingClassifier(random_state=0).fit(X_train_circ, y_train_named)
print(f"Decision function shape: {gbrt_circ.decision_function(X_test_circ).shape}")
print(f"First 5 decision function values:\n{gbrt_circ.decision_function(X_test_circ)[:5]}")
print(f"Predict proba shape: {gbrt_circ.predict_proba(X_test_circ).shape}")
print(f"First 5 predict proba values:\n{gbrt_circ.predict_proba(X_test_circ)[:5]}")

# Multiclass uncertainty estimates on Iris dataset
iris = load_iris()
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(iris.data, iris.target, random_state=42)
gbrt_iris = GradientBoostingClassifier(learning_rate=0.01, random_state=0).fit(X_train_i, y_train_i)
print(f"Multiclass Iris decision_function shape: {gbrt_iris.decision_function(X_test_i).shape}")
print(f"Multiclass Iris predict_proba shape: {gbrt_iris.predict_proba(X_test_i).shape}")

print("\nFinished running Chapter 2 code successfully!")
