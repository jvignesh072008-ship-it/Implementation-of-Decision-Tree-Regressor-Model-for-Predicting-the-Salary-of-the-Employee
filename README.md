# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Initial Setup and Data Loading: The program starts by importing all necessary Python libraries for data manipulation (pandas, numpy), machine learning (sklearn), and visualization (matplotlib). It then loads the Salary.csv dataset, the foundation for the entire process.
2. Exploratory Data Analysis: Basic checks are performed on the loaded data to understand its structure, including displaying the first five rows (data.head()), checking data types and non-null counts (data.info()), and confirming the absence of missing values (data.isnull().sum()).
3. Feature Engineering and Preprocessing: To make the data suitable for the machine learning model, the categorical Position column is transformed into a numerical format using LabelEncoder. The features (X) and target variable (y) are then defined.
4. Data Splitting: The dataset is split into a training set and a testing set. This is a critical step that ensures the model can be evaluated on data it has never seen, providing an unbiased assessment of its performance.
5. Model Training: A DecisionTreeRegressor is initialized and trained using the training data. The model learns the rules and patterns from this data to make future predictions.
6. Model Evaluation: The trained model predicts salaries for the test set, and its performance is quantitatively assessed using key metrics: the R² score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
7. Visualization and Feature Importance: The results are visualized using a scatter plot comparing actual versus predicted salaries. The performance metrics are overlaid, and the relative importance of each feature in the model's decision-making process is calculated and displayed, offering valuable insights into the model's behavior.
   
                                                              \
   
1. Data Loading and Inspection : The program begins by loading the Salary.csv dataset, and then prints the first few rows and summary information (data.head(), data.info()) to understand its structure, data types, and check for missing values. 
2. Data Preprocessing : It preprocesses the raw data by converting the categorical Position column into numerical format using LabelEncoder. It also defines the feature variables (X) and the target variable (y) for the model. 
3. Data Splitting : The dataset is split into separate training and testing sets. This ensures the model is evaluated on data it has not seen during training, preventing overfitting. 
4. Model Training : A DecisionTreeRegressor model is instantiated and then trained on the training data (X_train, y_train), which allows the model to learn the relationships between the features and the salary. 
5. Model Evaluation : The trained model makes predictions on the test data (X_test), and its performance is evaluated using several metrics: R² score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE). 
6. Visualization and Interpretation : The program visualizes the model's performance by plotting the predicted salary against the actual salary. It also calculates and displays the relative importance of each feature in the decision-making process. 

 
 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: VIGNESH J
RegisterNumber:  25014705
*/

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("Salary.csv")

# Display the first few rows
print("First 5 rows of the dataset:")
print(data.head())

# Check dataset information
print("\nDataset information:")
data.info()

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Encode categorical variables
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
print("\nAfter encoding Position column:")
print(data.head())

# Define features and target
X = data[["Position", "Level"]]
y = data["Salary"]

print("\nFeatures head:")
print(X.head())
print("\nTarget head:")
print(y.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Train the Decision Tree Regressor model
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

# Make predictions
y_pred = dt.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nModel Evaluation:")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

# Visualize actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual vs Predicted Salary')
plt.tight_layout()
plt.show()

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)
#testing data evaluation
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Test Set: Actual vs Predicted Salary')
plt.text(min(y_test), max(y_pred)*0.9, f"R² = {r2:.4f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}")
plt.show()

```

## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)
<img width="1233" height="727" alt="Screenshot 2025-10-06 192254" src="https://github.com/user-attachments/assets/46c1e4a5-1a55-4222-b31e-e35c393432fe" />
<img width="792" height="462" alt="Screenshot 2025-10-06 192310" src="https://github.com/user-attachments/assets/f4e5bc71-1b80-437e-b6cf-fd5cd39741cc" />
<img width="1379" height="735" alt="Screenshot 2025-10-06 192321" src="https://github.com/user-attachments/assets/32fd985a-4852-4e04-8190-b319258d1e59" />
<img width="1137" height="677" alt="Screenshot 2025-10-06 192337" src="https://github.com/user-attachments/assets/a4fec150-2ada-48fe-aba3-91c257c6a0c7" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
