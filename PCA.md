# Principal Component Analysis (PCA) — Concepts Explained

## What is PCA?

**Principal Component Analysis (PCA)** is a dimensionality reduction technique. Its goal is to take a dataset with many features (columns) and compress it into fewer dimensions while retaining as much meaningful information (variance) as possible.

Think of it like summarizing a long document: you lose some detail, but keep the key ideas.

---

## Step 1 — Feature Selection (Numeric Only)

```python
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
```

PCA works with numbers only. Non-numeric columns like IP addresses are excluded because they have no mathematical meaning in this context.

---

## Step 2 — Standardization (Z-score Scaling)

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Before applying PCA, each feature is standardized using the formula:

$$x'_j = \frac{x_j - \mu_j}{\sigma_j}$$

Where:
- $x_j$ = original feature value
- $\mu_j$ = mean of that feature
- $\sigma_j$ = standard deviation of that feature

**Why this matters:** PCA is sensitive to scale. A feature measured in thousands (e.g., bytes transferred) would dominate a feature measured in single digits (e.g., port number) without standardization. After scaling, all features have a mean of 0 and a standard deviation of 1, so they contribute equally.

> ⚠️ Important: `fit_transform` is called on the **training set only**. The same learned mean/std is then applied to the test set via `transform`. This prevents **data leakage**.

---

## Step 3 — Applying PCA

```python
n_components2 = 0.4
pca = PCA(n_components2, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
```

### What does PCA actually do?

PCA finds new axes (called **Principal Components**) in the data that capture the most variance. Each principal component (PC) is a linear combination of the original features:

$$PC_i = \sum_{j=1}^{p} w_{ij} \cdot x'_j$$

Where:
- $w_{ij}$ = the **weight (loading)** of feature $j$ in component $i$
- $x'_j$ = the standardized feature value
- $p$ = total number of features

The first PC captures the most variance, the second captures the next most (orthogonal to the first), and so on.

### What does `n_components=0.4` mean?

When you pass a float between 0 and 1, sklearn keeps enough components to explain that fraction of the total variance. Here, `0.4` means **40% of the total variance** is retained.

> Note: The comment in the code says "95% variance" but the actual parameter is `0.4` (40%). This is likely a bug or leftover comment in the original code.

---

## Step 4 — The Loadings Matrix

```python
loadings = pd.DataFrame(
    pca.components_,
    columns=numeric_cols,
    index=[f"PC{i}" for i in range(pca.n_components_)]
)
```

The **loadings matrix** shows how much each original feature contributes to each principal component. A high absolute value means that feature has strong influence on that PC.

| | feature_1 | feature_2 | feature_3 | ... |
|---|---|---|---|---|
| PC0 | 0.42 | -0.31 | 0.18 | ... |
| PC1 | 0.10 | 0.67 | -0.55 | ... |

---

## Step 5 — Saving Transformers

```python
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(pca, "models/pca.pkl")
```

Both the scaler and PCA model are saved. This is critical: when making predictions on new data in the future, you must apply the **same** transformation (same mean, same components), not refit from scratch.

---

## Summary of the Pipeline

```
Raw Data
   ↓
Select numeric columns
   ↓
Standardize (Z-score): subtract mean, divide by std
   ↓
PCA: project onto principal components (axes of maximum variance)
   ↓
Reduced Dataset (fewer columns, same rows)
```

---

## Key Concepts at a Glance

| Concept | Meaning |
|---|---|
| **Variance** | How spread out the data is — PCA maximizes this |
| **Principal Component** | A new axis that captures maximum variance |
| **Loadings** | How much each original feature contributes to a PC |
| **n_components (float)** | Fraction of variance to preserve (e.g., 0.4 = 40%) |
| **Standardization** | Required pre-step so features are on equal footing |
| **fit vs transform** | `fit` learns from training data; `transform` applies to any data |
| **Data leakage** | Fitting the scaler on test data would give an unfair advantage — always fit on train only |