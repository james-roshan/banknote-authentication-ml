# PP1 – Understanding of the Notebook and the Data

## 1. The most important finding: the notebook and the CSV don't match

| | `PP1.ipynb` (as written) | `bill_authentication.csv` (the data you have) |
|---|---|---|
| Source file loaded | `C:\Users\sanju\Downloads\archive(2).zip` (a customer dataset) | Local CSV in this folder |
| Rows | 2000 (1965 after dropping missing `Profession`) | 1372 |
| Columns | CustomerID, Gender, Age, Annual Income ($), Spending Score (1-100), Profession, Work Experience, Family Size | Variance, Skewness, Curtosis, Entropy, Class |
| Problem type | **Unsupervised** customer segmentation (K-Means), then classifiers trained to predict the cluster labels | **Supervised** binary classification: is a banknote genuine (0) or forged (1)? |
| Target | No real target – uses `kmeans.labels_` as `y` | `Class` column (762 × class 0, 610 × class 1) |

The notebook was written for a **Mall-Customers-style segmentation dataset**, not for the banknote data.
None of the column names it uses (`Age`, `Income`, `Spending Score`, `Gender`, `Profession` …) exist in
`bill_authentication.csv`, so running it against this CSV fails at the first column reference.

---

## 2. What the notebook does, section by section

### 2.1 Data selection and EDA
1. Imports `pandas`, `numpy`, `matplotlib`, `seaborn`.
2. Loads the customer dataset (hard-coded path on another user's machine – `C:\Users\sanju\...`).
3. Inspects it: `head()`, `shape` → (2000, 8), `info()`, `describe()`.
4. Renames `Annual Income ($)` → `Income`, `Spending Score (1-100)` → `Spending Score`.
5. Checks nulls → 35 missing `Profession` values.
6. Cleans: `drop_duplicates()`, then drops rows with missing `Profession` (→ 1965 rows).
7. Visualisations:
   - Boxplots of Age, Income, Spending Score, Work Experience, Profession, Family Size (outlier check).
   - Bar charts of Gender and Profession counts (both drawn on the same axes).
   - Histogram of Age, histogram + KDE of Income.
   - Scatter plot Income vs Spending Score.
   - Boxplot of Spending Score by Gender.
8. Keeps only numeric columns (`select_dtypes`) → drops Gender and Profession.
9. Correlation heatmap of the numeric columns.

### 2.2 Clustering (creates the "labels")
- Sets `OMP_NUM_THREADS=8` (avoids the Windows K-Means memory-leak warning).
- **K-Means, k = 3**, on `Income` + `Spending Score` → new `Cluster` column; scatter plot coloured by cluster.
- `pd.get_dummies(df, drop_first=True)` – has no effect because categorical columns were already dropped.

### 2.3 Classification with SVM
- Features `X = [Age, Income, Spending Score]`, target `y = kmeans.labels_`.
- 70/30 train/test split (`random_state=42`) → 590 test rows.
- `SVC(kernel='rbf', gamma='scale')`, **no feature scaling**.
- Result: **accuracy 0.9915**, classification report and confusion matrix printed.

### 2.4 3D visualisation
- 3D scatter of Age / Income / Spending Score coloured by class.
  (Title says "with SVM boundaries" but no decision boundary is actually drawn.)

### 2.5 Random Forest and XGBoost
- Same X / y / split.
- `RandomForestClassifier(n_estimators=100)` → **accuracy 0.9983**.
- `XGBClassifier(eval_metric='mlogloss')` → **accuracy 0.9915**
  (`use_label_encoder` is deprecated and ignored – gives a warning).

### 2.6 Model comparison
- Puts the three accuracies in a DataFrame and draws a bar chart "SVM vs RF vs XGBoost".

---

## 3. Issues / weaknesses noticed in the current code

1. **Wrong / non-portable data path** – points to a `.zip` on someone else's PC, and to a different dataset.
2. **Circular ML task** – the classifiers learn to reproduce K-Means labels that were built from
   Income + Spending Score, which are also in X. Near-100 % accuracy is expected and says little.
3. **Boxplot on `Profession`** (a text column) – seaborn can't draw a numeric boxplot of it; this cell errors or produces nonsense.
4. **Two bar charts on one axes** (Gender and Profession) overlap.
5. **`CustomerID` kept as a numeric feature** in the correlation heatmap (meaningless).
6. **No scaling before SVM / K-Means** – Income (~10⁵) dominates Spending Score (0–100). Distance-based models need `StandardScaler`.
7. Data-quality oddities not handled: `Age = 0` and `Income = 0` rows.
8. `get_dummies` step does nothing; `LabelEncoder` and `Axes3D` imported but unused.
9. Only a single train/test split, no cross-validation, no hyper-parameter tuning.

---

## 4. Proposed plan: adapt the notebook to `bill_authentication.csv`

The same workflow (EDA → SVM → Random Forest → XGBoost → comparison) fits the banknote data very well,
and here there is a **real target** (`Class`), so no K-Means label trick is needed.

1. **Load**: `df = pd.read_csv("bill_authentication.csv")` (relative path).
2. **EDA**: `head / shape / info / describe`, null + duplicate check, class balance (`value_counts`),
   boxplots + histograms of the 4 features, pairplot coloured by `Class`, correlation heatmap.
3. **Split**: `X = Variance, Skewness, Curtosis, Entropy`, `y = Class`,
   `train_test_split(test_size=0.3, stratify=y, random_state=42)`.
4. **Scaling**: `StandardScaler` inside a `Pipeline` (fit on train only).
5. **Models**: SVM (RBF, optionally also linear), Random Forest, XGBoost (`eval_metric='logloss'`).
6. **Evaluation**: accuracy, precision/recall/F1, confusion matrix, ROC-AUC, plus 5-fold cross-validation for a fairer comparison.
7. **Visualisation**: 3D scatter (Variance / Skewness / Curtosis coloured by Class), 2D SVM decision boundary on the two strongest features, model-comparison bar chart, feature importance for RF / XGB.
8. *(Optional)* keep a K-Means section as an unsupervised exercise (k = 2) and compare clusters with the true `Class`.

---

## 5. Questions before I change anything
- Do you want the **existing notebook rewritten** for `bill_authentication.csv`, or a **new notebook** (e.g. `PP1_banknote.ipynb`) leaving the original untouched?
- Keep the K-Means part (optional step 8) or drop it?
