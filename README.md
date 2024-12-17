# Online Shoppers Purchase Intent Prediction

This project aims to analyze and predict the purchase intent of online shoppers using a dataset of user activity and website interaction metrics. By leveraging machine learning techniques in Apache Spark, the project explores both supervised and unsupervised learning models to identify patterns and insights from the data.

---

## Project Structure

### Files Included:
1. **model4.py**: Python script for data preprocessing, feature engineering, and implementing machine learning models using Spark MLlib.
2. **online_shoppers_intention.csv**: Dataset containing online shopping activity metrics for analysis and prediction.

---

## Workflow and Features

### Data Preprocessing:
- Conversion of categorical data (e.g., `Month`, `VisitorType`) into numerical format using `StringIndexer`.
- Transformation of Boolean columns (`Weekend`, `Revenue`) into integer values.
- Feature vector assembly using `VectorAssembler`.
- Dimensionality reduction via Principal Component Analysis (PCA).

### Models Implemented:
#### Supervised Learning:
1. Logistic Regression
2. Naive Bayes

#### Unsupervised Learning:
1. K-Means Clustering
2. Gaussian Mixture Models

### Evaluation Metrics:
- **Supervised Models**: Accuracy using `MulticlassClassificationEvaluator`.
- **Unsupervised Models**: Silhouette Score using `ClusteringEvaluator`.

---

## How to Run

### Prerequisites:
- Python 3.x
- Apache Spark

### Steps:
1. Clone the repository and navigate to the project folder.
2. Update the `file_path` variable in `model4.py` with the absolute path to `online_shoppers_intention.csv`.
3. Run the script:
   ```bash
   python model4.py
