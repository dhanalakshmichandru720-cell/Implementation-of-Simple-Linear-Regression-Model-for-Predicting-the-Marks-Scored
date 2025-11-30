m# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

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
*/
# Implementation of Simple Linear Regression

## Program:

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.array(eval(input()))
y = np.array(eval(input()))

x_mean = np.mean(x)
y_mean = np.mean(y)

num = 0
den = 0

for i in range(len(x)):
    num += (x[i] - x_mean) * (y[i] - y_mean)
    den += (x[i] - x_mean) ** 2

m = num / den
b = y_mean - m * x_mean

print(m, b)

y_predicted = m * x + b
print(y_predicted)

plt.scatter(x, y)
plt.plot(x, y_predicted, color='red')
plt.show()
Developed by: dhanalakshmi.c
RegisterNumber: 25018616  
*/
```




## Output:
<img width="645" height="849" alt="Screenshot 2025-11-30 104559" src="https://github.com/user-attachments/assets/7f04b83f-8c3b-4484-a5c9-dcb9b8c00c5e" />




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.





