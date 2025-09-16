import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="KNN Playground", layout="centered")

# Sidebar controls
dataset = st.sidebar.selectbox("Dataset", ["moons", "circles", "blobs"])
n_samples = st.sidebar.slider("Samples", 100, 1000, 400, 50)
noise = st.sidebar.slider("Noise", 0.0, 0.5, 0.2, 0.01)
k = st.sidebar.slider("k (neighbors)", 1, 30, 5, 1)
weights = st.sidebar.selectbox("Weights", ["uniform", "distance"])
metric = st.sidebar.selectbox("Metric", ["minkowski", "euclidean", "manhattan"])
st.sidebar.caption("Tip: try k=1 vs k=30 and uniform vs distance.")

# Generate data (no pandas)
if dataset == "moons":
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
elif dataset == "circles":
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
else:
    X, y = make_blobs(n_samples=n_samples, centers=3, cluster_std=1.2 + noise*2, random_state=42)

# Train KNN
clf = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric)
clf.fit(X, y)

# Decision boundary grid
h = 0.03
x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Plot
fig, ax = plt.subplots(figsize=(6, 5))
ax.contourf(xx, yy, Z, alpha=0.25)             # decision regions
scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", alpha=0.5)
ax.set_title(f"KNN (k={k}, weights={weights}, metric={metric})")
ax.set_xlabel("x₁")
ax.set_ylabel("x₂")
ax.set_aspect("equal", adjustable="box")

st.pyplot(fig)