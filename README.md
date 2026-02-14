# Diabetes_Prediction
Contains code and assets for a multi‑class diabetes status prediction based on a Kaggle dataset. It includes data preprocessing, training and evaluation of six classification models, metric comparison tables, saved models, and an interactive Streamlit web app for uploading test data and visualizing predictions, metrics, and confusion matrices.

# Diabetes Status Prediction – Multi‑Class Classification

## 1. Problem Statement

The objective of this project is to build and compare multiple machine learning models that predict an individual’s diabetes status using demographic and clinical features.
The task is formulated as a **three‑class classification** problem:

- **N** – Non‑diabetic  
- **P** – Pre‑diabetic  
- **Y** – Diabetic  

The project covers the complete workflow: data preprocessing, model training and evaluation, comparison of six algorithms, and deployment of the final models via a Streamlit web application as required in the assignment.

---

## 2. Dataset Description

**Dataset:** Diabetes Prediction Dataset – Legit Dataset (Kaggle).

- **Source:** Public Kaggle dataset created for diabetes risk prediction.
- **Task type:** Multi‑class classification (N / P / Y).  
- **Target column:** `Class` (mapped internally to 0 = N, 1 = P, 2 = Y).  
- **Number of instances:** More than 1,000 records after cleaning, satisfying the requirement of at least 500 instances.
- **Number of features:** More than 12 input attributes after preprocessing and one‑hot encoding.

Key input features include:

- **Demographics:** Age, gender.  
- **Clinical measurements:** Urea, creatinine, HbA1c level, cholesterol, triglycerides, HDL, LDL, VLDL, BMI.  
- **Encoded attributes:** Gender indicators (`Gender_M`, `Gender_F`) after one‑hot encoding.

This dataset is moderately imbalanced: the majority of observations are Non‑diabetic, while Pre‑diabetic and Diabetic classes are less frequent, which influences model evaluation and motivates the use of macro‑averaged metrics.

---

## 3. Target Variable Distribution

After cleaning and mapping labels (`N`, `P`, `Y` → `0`, `1`, `2`), the class counts are:

- **Class 0 (N – Non‑diabetic):** majority class.  
- **Class 1 (P – Pre‑diabetic):** minority class.  
- **Class 2 (Y – Diabetic):** minority class.

A bar plot of the target distribution clearly shows that the Non‑diabetic class dominates, while the other two classes have fewer samples. This imbalance means that simple accuracy alone would be misleading, so macro Precision, Recall, F1, MCC and PR‑AUC are used to compare models.

---

## 4. Data Preprocessing

The preprocessing pipeline applied in the training notebook includes:

1. **Cleaning and type handling**

   - Loaded the raw CSV using `pandas`.  
   - Removed rows with missing or invalid values in critical numeric fields (e.g., BMI or laboratory measurements).  
   - Ensured all numeric columns were correctly typed as `float` or `int`.

2. **Target label processing**

   - Original target column: `Class` with values `{N, P, Y}`.
   - Cleaned label strings (trimmed spaces, upper‑cased) and mapped them to integers using `{"N": 0, "P": 1, "Y": 2}`.  
   - Stored the mapping in the code for consistent use during training and in the Streamlit app.

3. **Feature engineering and encoding**

   - Separated features and target:  
     - `X = df.drop(columns=["Class"])`  
     - `y = mapped integer labels`.  
   - Applied one‑hot encoding to categorical features (such as gender) using `pd.get_dummies(X, drop_first=True)`.  
   - After encoding, `X` contained 17 feature columns (more than the minimum requirement of 12).

4. **Train–test split and scaling**

   - Split data into training and test sets with `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)` to preserve class proportions.
   - Fitted a `StandardScaler` on the training features and transformed both `X_train` and `X_test` for models that need scaling (Logistic Regression and kNN).  
   - Tree‑based models and ensembles (Decision Tree, Random Forest, XGBoost) were trained on the unscaled features to preserve interpretability of splits.

5. **Imbalance handling**

   - Used `class_weight="balanced"` for Logistic Regression, Decision Tree and Random Forest to give higher weight to minority classes.  
   - For XGBoost, computed class‑based sample weights from the training distribution and passed them as `sample_weight` during fitting.  
   - Evaluation used macro‑averaged metrics so each class contributed equally regardless of its frequency.

6. **Test data for Streamlit**

   - Created a smaller `test_data.csv` from the test split by combining `X_test` with the encoded target column (`CLASS` = 0/1/2), then saved it for upload through the Streamlit interface.

---

## 5. Models Used and Evaluation Metrics (Assignment Section 3 – Step 5)

Six classification models were implemented on the same preprocessed dataset, as specified in the assignment:

1. Logistic Regression  
2. Decision Tree Classifier  
3. k‑Nearest Neighbor Classifier  
4. Naive Bayes Classifier (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

All models were evaluated on the hold‑out test set using the following metrics:

- **Accuracy**  
- **AUC (macro PR‑AUC / Average Precision)**  
- **Precision (macro)**  
- **Recall (macro)**  
- **F1 Score (macro)**  
- **Matthews Correlation Coefficient (MCC)**  

### 5.1 Comparison table of all models

| ML Model Name       | Accuracy | AUC (PR) | Precision | Recall  | F1      | MCC      |
|---------------------|----------|----------|-----------|---------|---------|----------|
| Logistic Regression | 0.935    | 0.799535 | 0.772678  | 0.852475| 0.804533| 0.784816 |
| Decision Tree       | 0.990    | 0.950228 | 0.996101  | 0.950794| 0.972365| 0.962918 |
| kNN                 | 0.920    | 0.799771 | 0.731073  | 0.696872| 0.712932| 0.694721 |
| Naive Bayes         | 0.105    | 0.368722 | 0.350970  | 0.355030| 0.074242| 0.066998 |
| Random Forest       | 0.995    | 0.996923 | 0.998039  | 0.966667| 0.981473| 0.981577 |
| XGBoost             | 0.995    | 0.999988 | 0.998039  | 0.966667| 0.981473| 0.981577 |

> AUC values above correspond to macro **PR‑AUC (Average Precision)** computed in a one‑vs‑rest manner across the three classes; this metric is more informative and stable than ROC‑AUC for imbalanced multi‑class problems.

### 5.2 Observations about model performance

| ML Model Name       | Observation about model performance |
|---------------------|--------------------------------------|
| **Logistic Regression** | Provides a strong baseline with accuracy of 0.935 and macro F1 of about 0.80. With class balancing and scaled features, the linear model captures the primary trends in the data and treats all three diabetes classes relatively fairly. However, its capacity is limited compared to non‑linear ensembles. |
| **Decision Tree**       | Achieves high accuracy (0.99) and macro F1 (~0.97), showing that a single tree can model non‑linear relationships between clinical variables and diabetes status. The macro PR‑AUC also improves significantly over the baseline, but the model may be sensitive to overfitting and to small changes in the training data or hyperparameters. |
| **kNN**                 | kNN performs reasonably (F1 ~0.71, MCC ~0.69) but lags behind tree‑based ensembles. Even after scaling, the nearest‑neighbor approach is affected by class imbalance and local density variations, leading to confusion between pre‑diabetic and diabetic classes. It is useful as a simple non‑parametric baseline. |
| **Naive Bayes**         | This model performs poorly on this dataset (accuracy ~0.10, F1 ~0.07, PR‑AUC ~0.37). The strong conditional independence assumptions of Gaussian Naive Bayes are not suitable for correlated clinical features such as BMI, HbA1c and lipid profile values, resulting in unreliable probability estimates and misclassification across all classes. |
| **Random Forest (Ensemble)** | Random Forest delivers excellent performance with accuracy near 0.995, macro F1 ~0.98, MCC ~0.98 and PR‑AUC ~0.997. The ensemble of many decision trees effectively captures complex interactions between features and, with class weighting, provides balanced performance on Non‑diabetic, Pre‑diabetic and Diabetic classes. |
| **XGBoost (Ensemble)**  | XGBoost matches Random Forest with similar accuracy and macro F1, and the highest PR‑AUC (~0.9999), indicating extremely strong ranking performance for the three classes. Gradient‑boosted trees with class‑based sample weights are able to model subtle patterns in the minority classes while maintaining excellent overall metrics. |

Overall, the ensemble models (Random Forest and XGBoost) clearly dominate all other methods in terms of macro F1, MCC and PR‑AUC, demonstrating the value of non‑linear ensemble methods for medical risk prediction on imbalanced multi‑class tabular data.
Logistic Regression and kNN serve as competitive baselines, while Naive Bayes indicates that strong independence assumptions are not appropriate for this problem.

---

## 6. Streamlit Application

### 6.1 About the app

A Streamlit web application was built and deployed on Streamlit Community Cloud to demonstrate the trained models interactively, as required in the assignment.
The app allows users to upload test data, select a model, and view predictions and evaluation metrics in real time.

Key features:

- **CSV upload (test data only):**  
  Users can upload a CSV file containing test records with the same feature columns used during training plus a `CLASS` column (0/1/2 or N/P/Y).  

- **Model selection dropdown:**  
  A sidebar dropdown lets the user switch between **Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest, and XGBoost** and instantly see how each model performs on the same test data.

- **Predictions table:**  
  The app shows predicted class codes and human‑readable labels (N, P, Y) for uploaded records, making it easy to inspect individual predictions.

- **Evaluation metrics:**  
  When the `CLASS` column is present and valid, the app computes and displays macro‑averaged **Accuracy, Precision, Recall, F1 Score, MCC, and AUC (macro PR‑AUC)** for the selected model.

- **Confusion matrix and classification report:**  
  A confusion matrix heatmap (with N / P / Y on the axes) is displayed along with the full multi‑class classification report summarizing per‑class precision, recall and F1 scores. This satisfies the assignment requirement for visualizing model performance.

### 6.2 How to use the app

1. Open the deployed Streamlit URL in a browser.  
2. In the left sidebar, choose one of the six models from the dropdown.  
3. Click **Browse files** and upload a CSV containing:  
   - All training feature columns in the same names as in the notebook.  
   - A `CLASS` column with either numeric labels 0/1/2 or string labels N/P/Y.  
4. The app will display:  
   - A preview of the uploaded data.  
   - A table of predicted codes and labels.  
   - Metric cards with Accuracy, Precision, Recall, F1, MCC and AUC (PR).  
   - A confusion matrix plot and the full classification report.  

If the `CLASS` column is missing or has unexpected values, the app still shows predictions but warns that evaluation metrics and confusion matrix cannot be computed for that file.

---

## 7. GitHub Repository Structure

The repository is organized as follows, in line with the assignment requirements


```text
## 7. GitHub Repository Structure

The repository is organized as follows:

```text
Diabetes_Prediction/
│
├── app.py                     # Streamlit application entry point
├── README.md                  # Project documentation and assignment write-up
├── requirements.txt           # Python package dependencies for Streamlit Cloud
├── test_data.csv              # Sample test subset used for Streamlit upload
│
├── model/
│   ├── Diabetes_Prediction.ipynb   # Notebook for preprocessing, training and metrics
│   ├── Diabetes_Prediction.py      # Script version of the notebook (optional)
│   ├── logistic_regression.pkl     # Saved Logistic Regression model
│   ├── decision_tree.pkl           # Saved Decision Tree model
│   ├── knn.pkl                     # Saved kNN model
│   ├── naive_bayes.pkl             # Saved Naive Bayes model
│   ├── random_forest.pkl           # Saved Random Forest model
│   ├── xgboost.pkl                 # Saved XGBoost model
│   ├── scaler.pkl                  # StandardScaler fitted on training data
│   └── feature_columns.pkl         # List of feature column names used in training
│
└── .devcontainer/             # (Optional) Dev container configuration for VS Code
    └── ...                    # Files used for local container-based development

