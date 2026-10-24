# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Set variables for assigning dataset values. 3.Import linear regression from sklearn.

3.Assign the points for representing in the graph.

4.Predict the regression for marks by using the representation of the graph. 6.Compare the graphs and hence we obtained the linear regression for the given dates.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SUJIN M L
RegisterNumber:  212225040435
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv('exp_2_dataset_student_scores.csv')  
print("First 5 rows:\n", df.head(), "\n")
print("Last 5 rows:\n", df.tail(), "\n")
```

<img width="257" height="401" alt="image" src="https://github.com/user-attachments/assets/3787bb9f-a76e-466d-9554-90a95c134e7e" />

```
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)
print("\nTraining samples:", len(X_train), " Testing samples:", len(X_test))
```

<img width="483" height="85" alt="image" src="https://github.com/user-attachments/assets/1985506c-f511-4871-af6e-5e6b4609d94c" />

```
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values
print("X (features):", X.flatten())
print("Y (targets):", Y)
```

<img width="911" height="133" alt="image" src="https://github.com/user-attachments/assets/6029168d-a3c3-4d7e-b521-88ba059c08fd" />



```
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
```

<img width="311" height="92" alt="image" src="https://github.com/user-attachments/assets/744bb6a4-94aa-4c94-a1af-02c587e5350f" />

```
Y_pred = regressor.predict(X_test)
print("\nPredicted values:", np.round(Y_pred, 2))
print("Actual values   :", Y_test)
```

<img width="773" height="103" alt="image" src="https://github.com/user-attachments/assets/676ea780-a4a6-4919-885d-85fb4b4f0c47" />

```
plt.figure(figsize=(6,4))
plt.scatter(X_train, Y_train, color="orange", label="Training data")
plt.plot(X_train, regressor.predict(X_train), color="red", label="Fitted line")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()
```

<img width="696" height="530" alt="image" src="https://github.com/user-attachments/assets/b8cb96eb-17ac-4c2a-ad80-b65f5b281db5" />

```
order = np.argsort(X_test.flatten())
X_test_sorted = X_test.flatten()[order]
Y_test_sorted = Y_test[order]
Y_pred_sorted = Y_pred[order]
plt.figure(figsize=(6,4))
plt.scatter(X_test, Y_test, color="blue", label="Test data")
plt.plot(X_test_sorted, Y_pred_sorted, color="green", label="Predictions")
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()
```

<img width="697" height="516" alt="image" src="https://github.com/user-attachments/assets/dbaf2b21-913f-4c1c-a56a-56409607e495" />


```
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print("\nMean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
```

<img width="571" height="108" alt="image" src="https://github.com/user-attachments/assets/4d00030b-4a2d-4e79-bb7a-a4cdfda652ed" />


```
new_hours = np.array([[2.5], [8.0]])
pred_new = regressor.predict(new_hours)
print("\nPredictions for new hours", new_hours.flatten(), "=>", np.round(pred_new,2))
```

<img width="642" height="68" alt="image" src="https://github.com/user-attachments/assets/33b3a583-f8d2-49dc-8b2a-6b6388ac4440" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
