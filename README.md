# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
`# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Sample dataset (Study Hours vs Marks)
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Marks': [35, 40, 50, 55, 65, 70, 75, 80, 85, 95]
}
df = pd.DataFrame(data)

# Split into features (X) and target (y)
X = df[['Hours']]  # Independent variable
y = df['Marks']    # Dependent variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Display results
print("Slope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)
print("Predicted Marks for 7.5 hours study:", model.predict([[7.5]])[0])

# Plotting
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Study Hours')
plt.ylabel('Marks Scored')
plt.title('Study Hours vs Marks')
plt.legend()
plt.show()
``
/*
Developed by: dhanalakshmi.c
RegisterNumber: 25018616  
*/
```

## Output:
<img width="1449" height="676" alt="Screenshot (146)" src="https://github.com/user-attachments/assets/76188d15-927c-43ae-bc26-a9b1397a0fb1" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
