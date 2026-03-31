# BLENDED_LEARNING
# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Start**

2. **Import Libraries**

   * Import required libraries:

     * `pandas`, `numpy`
     * `matplotlib`, `seaborn`
     * `train_test_split`
     * `Ridge`, `Lasso`, `ElasticNet`
     * `PolynomialFeatures`, `StandardScaler`, `Pipeline`
     * Evaluation metrics (`MSE`, `MAE`, `R²`)

3. **Load Dataset**

   * Read dataset `encoded_car_data (1).csv` into a DataFrame

4. **Data Preprocessing**

   * Convert categorical variables into numerical form:

     * Apply one-hot encoding using `pd.get_dummies()`

5. **Define Features and Target**

   * Define input features `X` (all columns except `price`)
   * Define target variable `y` (`price`)

6. **Feature Scaling**

   * Initialize `StandardScaler`
   * Scale feature matrix `X`
   * Reshape and scale target variable `y`

7. **Split Dataset**

   * Split data into:

     * Training set (80%)
     * Testing set (20%)
   * Use `random_state = 42`

---

### **Model Setup**

8. **Initialize Models**

   * Define a dictionary of models:

     * Ridge Regression (`alpha = 1.0`)
     * Lasso Regression (`alpha = 1.0`)
     * ElasticNet (`alpha = 1.0`, `l1_ratio = 0.5`)

9. **Create Results Storage**

   * Initialize an empty dictionary `results` to store evaluation metrics

---

### **Model Training and Evaluation**

10. **For Each Model in the Dictionary:**

* Create a pipeline consisting of:

  * Polynomial feature transformation (`degree = 2`)
  * Selected regression model (Ridge/Lasso/ElasticNet)

11. **Train Model**

* Fit pipeline using training data (`X_train`, `y_train`)

12. **Make Predictions**

* Predict values using test data (`X_test`)

13. **Evaluate Model**

* Compute:

  * Mean Squared Error (MSE)
  * Mean Absolute Error (MAE)
  * R² Score

14. **Store Results**

* Save metrics in the `results` dictionary for each model

---

### **Display Results**

15. **Print Model Performance**

* Display MSE, MAE, and R² for each model

16. **Convert Results to DataFrame**

* Convert `results` dictionary into a DataFrame
* Reset index and rename columns for clarity

---

### **Visualization**

17. **Plot Performance Metrics**

* Create bar plots using `seaborn`:

  * Plot 1: Models vs MSE
  * Plot 2: Models vs R² Score
* Add titles, labels, and rotate x-axis labels

18. **Adjust Layout and Display Plots**

---

19. **End**

## Program:
```
/*
Program to implement Ridge, Lasso, and ElasticNet regularization using pipelines.
Developed by: PRARTHANA D
RegisterNumber:  21225230213

/*
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("caras.csv")
print(data.head())

data = data.drop(['CarName', 'car_ID'], axis=1)
data = pd.get_dummies(data, drop_first=True)

X = data.drop('price', axis=1)
y = data['price']

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
    "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5)
}
results = {}

for name, model in models.items():

    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results[name] = {'MSE': mse, 'R² Score': r2}

print("\nName: PRARTHANA D")
print("Reg No: 212225530213")

for model_name, metrics in results.items():
    print(f"\n{model_name} Model")
    print(f"MSE: {metrics['MSE']:.2f}")
    print(f"R² Score: {metrics['R² Score']:.2f}")

results_df = pd.DataFrame(results).T
results_df.reset_index(inplace=True)
results_df.rename(columns={'index': 'Model'}, inplace=True)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.barplot(x='Model', y='MSE', data=results_df)
plt.title("Mean Squared Error Comparison")
plt.xticks(rotation=45)

plt.subplot(1,2,2)
sns.barplot(x='Model', y='R² Score', data=results_df)
plt.title("R² Score Comparison")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()



*/
```

## Output:

<img width="1051" height="569" alt="image" src="https://github.com/user-attachments/assets/07219258-0bc0-4132-8abe-49056def1036" />
<img width="1046" height="229" alt="image" src="https://github.com/user-attachments/assets/3b55c9fc-f89a-4391-aad6-0fbc750bb90a" />

<img width="1047" height="414" alt="image" src="https://github.com/user-attachments/assets/dea4058f-7f2c-41ed-a702-97c0ce5a8e5b" />



## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
