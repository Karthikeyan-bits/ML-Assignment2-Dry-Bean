# Dry Bean Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to classify dry bean varieties using their morphological characteristics and compare the performance of five supervised machine learning classifiers.

The project also demonstrates how trained classification models can be integrated into an interactive Streamlit application for evaluation on labeled test data.

## 2. Dataset Description

**Dataset:** Dry Bean Dataset  
**Source:** UCI Machine Learning Repository  
**Problem Type:** Multiclass Classification

The original dataset contains 13,611 observations, 16 numerical input features, and seven dry bean classes.

The seven classes are:

- Seker
- Barbunya
- Bombay
- Cali
- Dermason
- Horoz
- Sira

After removing 68 duplicate records, 13,543 observations remained.

A stratified 80:20 train-test split was used:

| Dataset | Observations |
|---|---:|
| Training | 10,834 |
| Testing | 2,709 |
| Total | 13,543 |

Feature scaling was incorporated into the relevant machine learning pipelines.

## 3. GitHub Repository

https://github.com/Karthikeyan-bits/ML-Assignment2-Dry-Bean

The repository contains the Streamlit application, dependency file, labeled test dataset, and five saved classification models.

## 4. Models Used and Evaluation Metrics

Five classification algorithms were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

Each model was evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

For this multiclass problem, Precision, Recall and F1 use macro averaging. AUC uses the multiclass One-vs-Rest approach with macro averaging.

### Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9192 | 0.9947 | 0.9307 | 0.9300 | 0.9302 | 0.9023 |
| Decision Tree | 0.8996 | 0.9612 | 0.9155 | 0.9141 | 0.9147 | 0.8786 |
| KNN | 0.9155 | 0.9832 | 0.9290 | 0.9256 | 0.9270 | 0.8978 |
| Naive Bayes | 0.8970 | 0.9911 | 0.9081 | 0.9082 | 0.9072 | 0.8762 |
| SVM | 0.9210 | 0.9945 | 0.9342 | 0.9317 | 0.9328 | 0.9045 |

## 5. Observations on Model Performance

| ML Model | Observation |
|---|---|
| Logistic Regression | Achieved 91.92% accuracy and the highest AUC of 0.9947. Its strong macro F1 score indicates effective classification performance across the seven bean varieties. |
| Decision Tree | Achieved 89.96% accuracy and a macro F1 score of 0.9147. It performed below Logistic Regression, KNN and SVM, although its tree-based structure offers interpretability. |
| KNN | Achieved 91.55% accuracy and a macro F1 score of 0.9270. It performed competitively, indicating that similarities among bean measurements are useful for classification. |
| Naive Bayes | Achieved 89.70% accuracy and a macro F1 score of 0.9072. It recorded the lowest accuracy and F1 among the five classifiers, although its AUC remained high at 0.9911. |
| SVM | Achieved the highest accuracy (92.10%), macro F1 (0.9328), and MCC (0.9045). It provided the strongest overall classification performance among the evaluated models. |

### Overall Winning Model: Support Vector Machine

SVM was selected as the overall best-performing model because it achieved the highest Accuracy, macro F1 and MCC.

Logistic Regression achieved a marginally higher AUC, but SVM performed better on the primary evaluation criteria used for overall model selection.

## 6. Streamlit Application

The Streamlit application supports:

- Uploading a labeled test dataset in CSV format.
- Selecting any of the five saved classifiers.
- Displaying six evaluation metrics.
- Viewing a confusion matrix and classification report.
- Comparing the performance of all five classifiers.
- Evaluating uploaded test data using saved models without retraining.

**Deployment status:** Streamlit Community Cloud deployment pending.

## 7. Conclusion

All five classification models were trained and evaluated using the same Dry Bean dataset and held-out test set.

SVM achieved the strongest overall performance, while Logistic Regression and KNN also produced competitive results.

The project demonstrates a workflow covering preprocessing, model training, evaluation, model serialization, and interactive application development.