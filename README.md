# Credit Card Fraud Detection with MLP and Imbalance Handling

This project focuses on detecting credit card fraud using a Multi-Layer Perceptron (MLP) neural network. A key aspect of this work is evaluating different techniques to handle the significant class imbalance inherent in fraud detection datasets.

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [Dataset](#dataset)
3.  [Methodology](#methodology)
    *   [Data Preparation](#data-preparation)
    *   [Balancing Techniques](#balancing-techniques)
    *   [Model](#model)
    *   [Evaluation Metrics](#evaluation-metrics)
4.  [Setup and Installation](#setup-and-installation)
    *   [Prerequisites](#prerequisites)
    *   [Dependencies](#dependencies)
    *   [Data](#data)
5.  [Usage](#usage)
6.  [Results Summary](#results-summary)
7.  [Key Insights](#key-insights)
8.  [Conclusion](#conclusion)
9.  [Visualizations](#visualizations)
    *   [Correlation Matrix](#correlation-matrix)
    *   [Confusion Matrices](#confusion-matrices)
10. [Potential Future Work](#potential-future-work)

## Project Overview
The primary goal is to build and evaluate an MLP model for credit card fraud detection. Due to the highly imbalanced nature of the dataset (many more non-fraudulent transactions than fraudulent ones), this project investigates the impact of several data balancing techniques:
*   **SMOTE (Synthetic Minority Over-sampling Technique)**
*   **Random Oversampling**
*   **Random Undersampling**

The performance of the MLP model is assessed on the original imbalanced data and then on data preprocessed with each of these balancing methods.

## Dataset
*   **Source**: The project uses the `creditcard.csv` dataset.
*   **Size**: 284,807 transactions.
*   **Fraud Instances**: 492 fraudulent transactions.
*   **Features**: The dataset contains anonymized features (V1-V28), `Time`, and `Amount`. The target variable is `Class` (0 for non-fraud, 1 for fraud).

## Methodology

### Data Preparation
1.  **Loading Data**: The `creditcard.csv` dataset is loaded using pandas.
2.  **Initial Checks**:
    *   Data shape and null/missing value checks are performed.
    *   Descriptive statistics are generated.
3.  **Scaling**: The features (excluding 'Class') are scaled using `StandardScaler`. This is crucial for neural networks as they are sensitive to feature magnitudes. The scaler is fit on the training set and applied to both training and test sets to prevent data leakage.
4.  **Correlation Analysis**: A correlation matrix is generated and visualized to understand relationships between features.
5.  **Train-Test Split**: The data is split into training (70%) and testing (30%) sets, stratified by the 'Class' variable to ensure similar class proportions in both sets.

### Balancing Techniques
The following techniques are applied to the *training data only* to address class imbalance:
1.  **SMOTE**: Generates synthetic samples for the minority class (fraud) by interpolating between existing minority instances using k-Nearest Neighbors.
    *   *Pros*: Helps avoid overfitting compared to simple duplication.
    *   *Cons*: Can create noise if synthetic samples are generated in areas where they don't belong.
2.  **Random Oversampling**: Duplicates instances from the minority class until it is balanced with the majority class.
    *   *Pros*: Simple to implement.
    *   *Cons*: High risk of overfitting as the model sees identical fraudulent samples multiple times.
3.  **Random Undersampling**: Randomly removes instances from the majority class until it is balanced with the minority class.
    *   *Pros*: Can lead to faster training due to a smaller dataset.
    *   *Cons*: Can lead to loss of important information from the majority class.

### Model
*   **Neural Network (MLP)**: A Multi-Layer Perceptron classifier is used as the predictive model. The notebook uses a simple architecture with two hidden layers (64 and 32 units) and ReLU activation.

### Evaluation Metrics
The model's performance, especially for the fraud class (Class=1), is evaluated using:
*   **Precision**: Proportion of correctly identified frauds out of all transactions predicted as fraud (TP / (TP + FP)). Aims to minimize false alarms.
*   **Recall**: Proportion of actual frauds that were correctly identified (TP / (TP + FN)). Aims to capture as many frauds as possible.
*   **F1-Score**: The harmonic mean of precision and recall, providing a balance between the two (2 * (Precision * Recall) / (Precision + Recall)).
*   **Confusion Matrix**: Visualized for each balancing technique to show True Positives, True Negatives, False Positives, and False Negatives.
*   **Overall Test Accuracy**: While reported, less emphasis is placed on this metric due to the class imbalance.

## Setup and Installation

### Prerequisites
*   Python (3.7+ recommended)
*   pip (Python package installer)

### Dependencies
The project uses the following Python libraries. You can install them using pip:
```bash
pip install pandas scikit-learn imblearn matplotlib seaborn
```

## Requirements

Create a `requirements.txt` file:

```text
pandas
scikit-learn
imblearn
matplotlib
seaborn
```

## Data

The notebook expects a CSV file named `creditcard.csv` in the same directory as the notebook (or a specified path). This dataset is widely available (e.g., on [Kaggle](https://www.kaggle.com/)).

---

## Usage

1. Ensure all dependencies are installed.  
2. Place `creditcard.csv` in the appropriate location.  
3. Open the Jupyter Notebook: `fraud2.ipynb`.  
4. Run all cells sequentially.

The notebook will:  
- Load and preprocess the data  
- Split into training and testing sets  
- Apply SMOTE, Random Oversampling, and Random Undersampling on the training data  
- Train and evaluate an MLP model on:  
  - Original imbalanced data  
  - SMOTE-balanced data  
  - Random Oversampled data  
  - Random Undersampled data  
- Display evaluation metrics and confusion matrices for each scenario  

---

## Results Summary

| Technique                | Precision | Recall | F1-Score |
|--------------------------|:---------:|:------:|:--------:|
| Original (Imbalanced)    |   0.90    |  0.72  |   0.80   |
| SMOTE                    |   0.76    |  0.75  |   0.75   |
| Random Oversampling      |   0.73    |  0.72  |   0.73   |
| Random Undersampling     |   0.04    |  0.88  |   0.08   |

---

## Key Insights

- **Original Data**  
  - Highest F1-score (0.80) but missed 28% of fraud cases (Recall = 0.72).

- **SMOTE**  
  - Best balance (Precision = 0.76, Recall = 0.75), F1 = 0.75.

- **Random Undersampling**  
  - Highest recall (0.88) but very low precision (0.04), leading to excessive false positives.

- **Random Oversampling**  
  - Moderate performance (Precision = 0.73, Recall = 0.72), risk of overfitting duplicated samples.

---

## Conclusion

- The MLP on **original imbalanced data** achieved the highest F1-score, showing effective learning despite class imbalance.  
- **SMOTE** offers the best trade-off for detecting fraud (higher recall) without drastically lowering precision.  
- **Random Undersampling** is generally impractical due to too many false positives.  
- **Random Oversampling** is simple but may overfit.

*Choice of technique should align with business requirements (tolerance for false positives vs. missed frauds).*

---

## Visualizations

- **Correlation Matrix**: Heatmap of feature correlations.  
- **Confusion Matrices**: For MLP trained on:  
  - Original unbalanced data  
  - SMOTE  
  - Random Oversampling  
  - Random Undersampling  

---

## Potential Future Work

- **Hyperparameter Tuning**: Optimize network architecture, learning rates, etc.  
- **Alternative Models**: Try Logistic Regression, Random Forest, Gradient Boosting, SVM.  
- **Advanced Sampling**: ADASYN, Tomek Links, Edited Nearest Neighbours, hybrid methods.  
- **Cost-Sensitive Learning**: Assign different misclassification costs.  
- **Anomaly Detection**: Unsupervised/semi-supervised approaches.  
- **Feature Engineering**: Create new discriminative features.  
- **Ensemble Methods**: Combine multiple models or resampling strategies.  
