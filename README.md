# Diabetes_Prediction
Contains code and assets for a multi‑class diabetes status prediction based on a Kaggle dataset. It includes data preprocessing, training and evaluation of six classification models, metric comparison tables, saved models, and an interactive Streamlit web app for uploading test data and visualizing predictions, metrics, and confusion matrices.

## 1. Problem Statement

The goal of this project is to build and compare multiple machine learning models that can predict a person’s diabetes status based on demographic and clinical features. 
The target variable has three classes:

- **N** – Non‑diabetic  
- **P** – Pre‑diabetic  
- **Y** – Diabetic  

The models are trained, evaluated, and then deployed as an interactive Streamlit web application where users can upload test data and obtain predictions and evaluation metrics. 

***

## 2. Dataset Description

**Dataset:** “Diabetes Prediction Dataset – legit dataset” from Kaggle (public, gold‑rated). 

- **Source:** Kaggle – Diabetes Prediction Dataset (Legit Dataset)  
- **Task type:** Multi‑class classification (N / P / Y)  
- **Target column:** `Class` (encoded internally as 0 = N, 1 = P, 2 = Y)  
- **Number of instances:** 1,000+ rows (after cleaning, > 500 rows). 
- **Number of features:** 12+ input features after preprocessing and one‑hot encoding (original columns such as age, gender, BMI, HbA1c level, blood glucose level, and lifestyle‑related attributes). 

### Data preprocessing

Key preprocessing steps:

- Removed rows with missing or invalid values in critical fields (e.g., BMI or glucose level).  
- Standardized the target labels: `Class` values were cleaned and mapped from `{"N","P","Y"}` to integer labels `{0,1,2}`.  
- Performed one‑hot encoding for categorical features (e.g., gender, smoking status, family history) using `pd.get_dummies(..., drop_first=True)`, which expanded the feature space to 17 columns in `X`. 
- Split the data into training and test sets using `train_test_split` with `stratify=y` to preserve the class distribution in both splits.  
- Applied `StandardScaler` to numeric features for models that are sensitive to scale (Logistic Regression, kNN); tree‑based and ensemble models used unscaled features.

The dataset is imbalanced, with the majority of samples in the Non‑diabetic class and fewer samples in the Pre‑diabetic and Diabetic classes, which directly impacts evaluation and choice of metrics. 

***

## 3. Models Used and Evaluation Metrics

For this assignment, the following six models were implemented on the same preprocessed dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. k‑Nearest Neighbor (kNN) Classifier  
4. Naive Bayes Classifier (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

To address class imbalance, `class_weight="balanced"` was used where supported (Logistic Regression, Decision Tree, Random Forest), and XGBoost used class‑based sample weights computed from the training class frequencies. 

Macro‑averaged metrics were used for multi‑class evaluation:

- **Accuracy**  
- **AUC** (macro one‑vs‑rest; undefined in this split → NaN)  
- **Precision (macro)**  
- **Recall (macro)**  
- **F1 (macro)**  
- **Matthews Correlation Coefficient (MCC)**  

### 3.1 Comparison table of all models

| ML Model Name       | Accuracy | AUC  | Precision | Recall  | F1      | MCC      |
|---------------------|----------|------|-----------|---------|---------|----------|
| Logistic Regression | 0.935    | NaN  | 0.772678  | 0.852475| 0.804533| 0.784816 |
| Decision Tree       | 0.990    | NaN  | 0.996101  | 0.950794| 0.972365| 0.962918 |
| kNN                 | 0.920    | NaN  | 0.731073  | 0.696872| 0.712932| 0.694721 |
| Naive Bayes         | 0.105    | NaN  | 0.350970  | 0.355030| 0.074242| 0.066998 |
| Random Forest       | 0.995    | NaN  | 0.998039  | 0.966667| 0.981473| 0.981577 |
| XGBoost             | 0.995    | NaN  | 0.998039  | 0.966667| 0.981473| 0.981577 |

> Note: AUC is reported as `NaN` because in this specific train–test split the macro one‑vs‑rest ROC‑AUC is undefined for at least one class (only a single class present in some one‑vs‑rest slice). Other metrics (Precision, Recall, F1, MCC) provide a stable comparison.

***

## 4. Observations on Model Performance

| ML Model Name       | Observation about model performance |
|---------------------|--------------------------------------|
| Logistic Regression | Achieves strong overall macro F1 (~0.80) and MCC (~0.78), showing that a linear decision boundary with class balancing can separate the three diabetes status classes reasonably well, particularly when features are standardized. |
| Decision Tree       | Very high accuracy (0.99) and macro F1 (~0.97) indicate that a single tree can capture non‑linear relationships in the features; however, the model is more prone to overfitting, and performance may be sensitive to changes in train–test split or hyperparameters such as depth and minimum samples per leaf. |
| kNN                 | Performance is solid but lower than tree‑based ensembles, with F1 around 0.71 and MCC around 0.69; kNN benefits from scaling but is affected by class imbalance and local density variations in feature space, sometimes confusing pre‑diabetic and diabetic classes. |
| Naive Bayes         | Shows very low accuracy (~0.10) and F1 (~0.07), suggesting that the conditional independence assumptions are too restrictive for this dataset; the feature dependencies (e.g., correlations between BMI, age, and glucose measures) are not well captured by Gaussian NB, leading to poor multi‑class discrimination. |
| Random Forest (Ensemble) | Delivers excellent results with accuracy around 0.995, macro F1 ~0.98, and MCC ~0.98, indicating robust performance across all three classes; the ensemble of trees captures complex interactions and is resilient to noise, while class weighting helps it treat minority classes (pre‑diabetic and diabetic) more fairly. |
| XGBoost (Ensemble)  | Matches Random Forest with similarly high accuracy and macro F1, showing that gradient‑boosted trees are also highly effective for this dataset; XGBoost’s ability to model subtle non‑linear patterns and the use of class‑based sample weights make it particularly strong on the minority diabetes classes. |

Overall, the ensemble models (Random Forest and XGBoost) clearly outperform the simpler baselines in terms of macro F1 and MCC, while Naive Bayes performs poorly and logistic regression/kNN provide interpretable but less powerful alternatives. 
This highlights the importance of non‑linear, ensemble‑based approaches for multi‑class medical prediction problems where subtle feature interactions matter.

***

## 5. Streamlit Web Application

The trained models are deployed via a Streamlit app hosted on Streamlit Community Cloud, as required in the assignment. 

Key features implemented:

- **CSV upload (test data only):** Users can upload a test subset of the diabetes dataset containing the same feature columns and an optional `Class` column.  
- **Model selection dropdown:** A sidebar dropdown allows switching between the six models (Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest, XGBoost).  
- **Prediction display:** The app shows predicted class codes (0/1/2) and corresponding labels (N/P/Y) for uploaded records.  
- **Evaluation metrics:** When the true `Class` column is present in the uploaded CSV, the app computes and displays macro‑averaged Accuracy, Precision, Recall, F1, and MCC for the selected model.  
- **Confusion matrix & classification report:** The app visualizes a confusion matrix heatmap and prints the full multi‑class classification report (per‑class precision, recall, F1) for the chosen model.
  
This end‑to‑end workflow demonstrates model training, evaluation, and deployment for a real‑world multi‑class medical prediction problem.
