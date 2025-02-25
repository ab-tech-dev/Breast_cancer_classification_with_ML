# Breast Cancer Classification Using Machine Learning

## Overview

This project uses **Logistic Regression**, a statistical model that predicts the probability of a binary outcome, to classify breast tumors as either benign or malignant based on biopsy data. The dataset is derived from **fine needle aspiration (FNA)** procedures, a minimally invasive biopsy technique that extracts cell samples using a thin needle. The dataset contains **33 features**, which are numerical characteristics of the tumors, across **569 samples**.

## Types of Tumors

### Benign Tumor

- Non-cancerous
- Encapsulated (surrounded by a protective layer)
- Slow growing
- Non-invasive (does not invade nearby tissues)
- Do not metastasize (spread to other parts of the body)
- Cells appear normal under a microscope

### Malignant Tumor

- Cancerous
- Non-encapsulated (not surrounded by a protective layer)
- Fast growing
- Invasive (spreads into nearby tissues)
- Metastasize (can spread to other parts of the body through the bloodstream or lymphatic system)
- Cells may have abnormal shapes (large, dark nuclei indicating rapid and uncontrolled growth)

## Dataset Details

The dataset contains diagnostic measurements extracted from **fine needle aspiration biopsies**. The features include:

- **ID**: Identification number
- **Diagnosis**: Malignant (M) or Benign (B)
- **radius_mean**: Mean of distances from center to points on the perimeter
- **texture_mean**: Standard deviation of gray-scale values (variation in texture)
- **perimeter_mean**: Mean size of the core tumor boundary
- **area_mean**: Mean area of the tumor
- **smoothness_mean**: Mean variation in radius lengths (how smooth or irregular the tumor is)
- **compactness_mean**: Mean of (perimeter² / area) - 1.0 (a measure of shape compactness)
- **concavity_mean**: Mean severity of concave portions of the contour (measuring the depth of indentations)
- **concave points_mean**: Mean number of concave portions in the contour
- **symmetry_mean**: Measure of symmetry of the tumor shape
- **fractal_dimension_mean**: Measure of tumor boundary complexity (higher values indicate rougher edges)
- **radius_se**: Standard error for the mean radius
- **texture_se**: Standard error for texture measurement
- **perimeter_se**: Standard error for perimeter
- **area_se**: Standard error for area
- **smoothness_se**: Standard error for smoothness measure
- **compactness_se**: Standard error for compactness measure
- **concavity_se**: Standard error for concavity measure
- **concave points_se**: Standard error for concave points count
- **symmetry_se**: Standard error for symmetry measure
- **fractal_dimension_se**: Standard error for fractal dimension measure
- **radius_worst**: Largest mean radius value recorded
- **texture_worst**: Largest mean texture value recorded
- **perimeter_worst**: Largest mean perimeter value recorded
- **area_worst**: Largest mean area value recorded
- **smoothness_worst**: Largest mean smoothness value recorded
- **compactness_worst**: Largest mean compactness value recorded
- **concavity_worst**: Largest mean concavity value recorded
- **concave points_worst**: Largest mean number of concave points recorded
- **symmetry_worst**: Largest mean symmetry value recorded
- **fractal_dimension_worst**: Largest mean fractal dimension value recorded

## Workflow

1. Import necessary libraries
2. Load and explore the dataset
3. Preprocess the data (handle missing values, feature selection, and encoding)
4. Split the data into training and testing sets
5. Train a **Logistic Regression** model
6. Evaluate the model’s accuracy
7. Use the model to predict tumor types on new data

## Implementation

```python
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load and view the dataset
breast_cancer_dataset = pd.read_csv('breast_cancer_data.csv')
breast_cancer_dataset.head()

# Display dataset summary information
breast_cancer_dataset.info()

#checking for missing values
breast_cancer_dataset.isnull().sum()

# Check for missing values and remove unnecessary columns
if 'Unnamed: 32' in breast_cancer_dataset.columns:
    breast_cancer_dataset.drop(columns=['Unnamed: 32'], inplace=True)

#checking for missing values
breast_cancer_dataset.isnull().sum()

# Encode the 'diagnosis' column: M = 1, B = 0
breast_cancer_dataset['label'] = breast_cancer_dataset['diagnosis'].map({'M': 1, 'B': 0})
breast_cancer_dataset.drop(columns=['diagnosis'], inplace=True)

# Determining how many Malignant and Benign tumors are in the dataset
breast_cancer_dataset['label'].value_counts()

#Grouping the dataset by the label/diagnosis value (Benign | Malinant)
breast_cancer_dataset.groupby('label').mean()

# Define features (X) and target label (Y)
X = breast_cancer_dataset.drop(columns='label', axis=1)
Y = breast_cancer_dataset['label']

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluate model accuracy
train_accuracy = accuracy_score(Y_train, model.predict(X_train))
test_accuracy = accuracy_score(Y_test, model.predict(X_test))

print(f'Accuracy on Training Data: {train_accuracy * 100:.2f}%')
print(f'Accuracy on Test Data: {test_accuracy * 100:.2f}%')

# Actual values for the last 30 samples
actual_values = breast_cancer_dataset['label'].tail(30)
# Test model on the last 30 samples
input_features = breast_cancer_dataset.tail(30).drop(columns=['label'])
predictions = model.predict(input_features)

# Display prediction results
benign_actual = sum(actual_values == 0)
benign_pred = sum(predictions == 0)
malignant_actual = sum(actual_values == 1)
malignant_pred = sum(predictions == 1)
print(f'Actual Benign Tumors: {benign_actual}')
print(f'Predicted Benign Tumors: {benign_pred}')
print(f'Actual Malignant Tumors: {malignant_actual}')
print(f'Predicted Malignant Tumors: {malignant_pred}')
```


## Results and Conclusion  

- The **Logistic Regression** model achieved an accuracy of **92.09%** on training data and **92.11%** on test data, demonstrating strong predictive performance.  
- The model successfully distinguishes between **benign** and **malignant** tumors based on key biopsy features.  
- While the results are promising, further improvements can be made through **hyperparameter tuning**, feature engineering, or employing more complex models such as **Support Vector Machines (SVM), Decision Trees, or Neural Networks**.  
- Additionally, integrating **cross-validation** and **ensemble learning** techniques could enhance model robustness and generalization to new data.  

This study highlights the potential of machine learning in **early breast cancer detection**, which can aid healthcare professionals in making informed decisions.   


## Future Enhancements

- Implement **alternative classification models** (e.g., SVM, Decision Trees, Neural Networks).
- Optimize model performance through **hyperparameter tuning**.
- Deploy the model as a **web application** for real-time tumor classification.

## References

- [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

