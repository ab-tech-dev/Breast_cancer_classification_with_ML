# Breast Cancer Classification Using Machine Learning

## Project Overview
This project implements a **Logistic Regression** model to classify breast tumors as benign or malignant based on biopsy data. The dataset is derived from **fine needle aspiration (FNA)** procedures, which involve extracting cell samples using a thin needle. The dataset consists of **569 samples** with **33 numerical features** describing tumor characteristics.

## Dataset
The dataset (`breast_cancer_data.csv`) contains diagnostic measurements extracted from **fine needle aspiration biopsies**. The key features include:

| Feature | Description |
|---------|-------------|
| ID | Identification number |
| Diagnosis | Malignant (M) or Benign (B) |
| radius_mean | Mean distance from center to points on the perimeter |
| texture_mean | Standard deviation of gray-scale values |
| perimeter_mean | Mean size of the core tumor boundary |
| area_mean | Mean area of the tumor |
| smoothness_mean | Mean variation in radius lengths |
| compactness_mean | Mean of (perimeter² / area) - 1.0 |
| concavity_mean | Mean severity of concave portions of the contour |
| concave points_mean | Mean number of concave portions in the contour |
| symmetry_mean | Measure of symmetry of the tumor shape |
| fractal_dimension_mean | Measure of tumor boundary complexity |
| (Other features include standard errors and worst-case values for these metrics) |

The target variable is `diagnosis`, which is encoded as **M = 1 (Malignant), B = 0 (Benign).**

## Dependencies
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

## Getting Started

### Prerequisites
- Python 3.10+
- Required libraries listed in the dependencies section

### Installation
```bash
# Clone the repository
git clone https://github.com/ab-tech-dev/Breast_cancer_classification_with_ML
cd Breast_cancer_classification_with_ML

# Install dependencies
pip install -r requirements.txt
```



## Model
The project employs **Logistic Regression**, a statistical model used for binary classification tasks. Logistic Regression predicts the probability of a sample belonging to a particular class based on input features.

## Implementation
1. **Data Loading**: Load the dataset from CSV.
2. **Data Preprocessing**: Handle missing values, feature selection, and encoding.
3. **Exploratory Data Analysis**: Analyze the dataset to understand key characteristics.
4. **Feature Selection**: Identify the most relevant features.
5. **Model Training**: Train a Logistic Regression model.
6. **Evaluation**: Assess the model's performance on test data.

### Code Implementation
```python
# Load the dataset
breast_cancer_dataset = pd.read_csv('breast_cancer_data.csv')
breast_cancer_dataset.head()

# Display dataset summary information
breast_cancer_dataset.info()

# Check for missing values and remove unnecessary columns
if 'Unnamed: 32' in breast_cancer_dataset.columns:
    breast_cancer_dataset.drop(columns=['Unnamed: 32'], inplace=True)

# Encode the 'diagnosis' column: M = 1, B = 0
breast_cancer_dataset['label'] = breast_cancer_dataset['diagnosis'].map({'M': 1, 'B': 0})
breast_cancer_dataset.drop(columns=['diagnosis'], inplace=True)

# Define features (X) and target variable (Y)
X = breast_cancer_dataset.drop(columns='label', axis=1)
Y = breast_cancer_dataset['label']

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluate model accuracy
train_accuracy = accuracy_score(Y_train, model.predict(X_train))
test_accuracy = accuracy_score(Y_test, model.predict(X_test))

print(f'Accuracy on Training Data: {train_accuracy * 100:.2f}%')
print(f'Accuracy on Test Data: {test_accuracy * 100:.2f}%')
```

## Results
- The **Logistic Regression** model achieved an accuracy of **92.09%** on training data and **92.11%** on test data.
- The model effectively classifies breast tumors based on biopsy features.
- Future improvements may include **hyperparameter tuning**, feature engineering, or exploring alternative models like **Support Vector Machines (SVM), Decision Trees, or Neural Networks**.

## Future Enhancements
- Implement **alternative classification models** (e.g., SVM, Decision Trees, Neural Networks).
- Optimize model performance through **hyperparameter tuning**.
- Deploy the model as a **web application** for real-time tumor classification.

## Contributors
[Joshua Ability](https://github.com/ab-tech-dev)

## References
- [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

