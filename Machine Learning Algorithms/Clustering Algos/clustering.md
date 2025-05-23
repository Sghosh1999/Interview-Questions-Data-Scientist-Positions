# 📚 Clustering Algorithms – Comprehensive README

## I. K-Means / K-Median / K-Mode Clustering

---

### 🔹 K-Means Clustering

**Definition**: Partitions data into $K$ clusters such that the sum of squared distances between data points and their assigned cluster centroids (means) is minimized.

---

### 🔹 K-Means++ (Initialization Improvement)

**Problem in Standard K-Means**: Random initialization of centroids can lead to poor convergence or local optima.

**K-Means++ Improvement**:

- Improves centroid initialization to spread them out in feature space.
- Leads to faster convergence and more stable results.

**Algorithm**:

1. Choose the first centroid randomly from data points.
2. For each data point $x$, compute the distance $D(x)$ to the nearest chosen centroid.
3. Choose the next centroid with probability proportional to $D(x)^2$.
4. Repeat until $K$ centroids are initialized.

**Code**:

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10).fit(X)
```

**Pros**:

- More accurate than random init
- Fewer iterations to converge

---

### 🔹 K-Median Clustering

- **Centroid**: Median
- **Distance**: Manhattan (L1)
- **Robust to outliers**

---

### 🔹 K-Mode Clustering

- **Centroid**: Mode
- **Use**: Categorical data only
- **Distance**: Matching dissimilarity

```python
from kmodes.kmodes import KModes
km = KModes(n_clusters=3, init='Huang').fit(X)
```

---

## II. Hierarchical Clustering (Agglomerative)

**Definition**: Recursively merges closest clusters until a single cluster (tree-like structure).

- **Linkage Criteria**:

  - Single: min distance
  - Complete: max distance
  - Average: average distance
  - Ward: minimizes total variance

**Dendrogram** used to visualize the cluster tree.

```python
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(X)
```

---

## III. DBSCAN (Density-Based Spatial Clustering)

**Key Concepts**:

- **Core Point**: At least `minPts` in $\varepsilon$-neighborhood
- **Border Point**: Close to a core, but not core itself
- **Noise**: Not reachable from any core point

```python
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.5, min_samples=5).fit(X)
```

---

## IV. Self-Organizing Maps (SOM)

- Neural grid maps high-dimensional data to 2D.
- Competitive learning (nodes “win” to update weights)
- Used for clustering + visualization

```python
from minisom import MiniSom
som = MiniSom(10, 10, X.shape[1])
som.train_random(X, 100)
```

---

## V. Singular Value Decomposition (SVD)

- Matrix Factorization: $A = U \Sigma V^T$
- Used to reduce dimensionality before clustering

```python
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=50)
X_reduced = svd.fit_transform(X)
```

---

## VI. Elbow Method

- Plot $K$ vs Inertia (WCSS)
- "Elbow" is where adding clusters has diminishing returns

---

## VII. Gap Statistic

- Compares clustering performance to a reference null dataset
- Choose $K$ with largest **gap**

---

## VIII. Silhouette Method

- Measures cohesion vs separation

$$
s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}
$$

```python
from sklearn.metrics import silhouette_score
score = silhouette_score(X, labels)
```

---

## IX. K-Prototypes Clustering

- Combines K-Means (numerical) + K-Modes (categorical)
- Mixed data types

```python
from kmodes.kprototypes import KPrototypes
kproto = KPrototypes(n_clusters=3)
clusters = kproto.fit_predict(X, categorical=[1,2])
```

---

## X. Gaussian Mixture Models (GMM)

- Soft clustering: points belong to multiple clusters with probabilities
- Fit using Expectation-Maximization

```python
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3).fit(X)
```

---
