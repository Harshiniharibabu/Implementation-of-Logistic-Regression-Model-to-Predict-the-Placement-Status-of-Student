# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program and import required libraries (pandas, sklearn).
2. Create or load dataset and separate it into

 * Features (CGPA, IQ, Projects) → X
 
 * Target (Placement status) → y

3.Split the dataset into training and testing sets.

4.Train the Logistic Regression model using the training data.

5.Predict and evaluate the model using test data and display the placement result.
 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: HARSHINI H
RegisterNumber: 212225040119 
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.DataFrame({
    'cgpa': [7.5, 6.2, 8.0, 5.5, 7.8, 6.5, 7.0, 6.8, 8.2, 5.9],
    'iq': [120, 100, 130, 90, 125, 105, 115, 110, 135, 95],
    'projects': [2, 1, 3, 0, 2, 1, 2, 1, 3, 0],
    'placed': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
})

X = data[['cgpa', 'iq', 'projects']]
y = data['placed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

new_student = pd.DataFrame([[7.2, 115, 2]], columns=['cgpa', 'iq', 'projects'])
result = model.predict(new_student)

if result[0] == 1:
    print("Student is likely to be Placed")
else:
    print("Student is NOT likely to be Placed")
```

## Output:

<img width="401" height="127" alt="image" src="https://github.com/user-attachments/assets/f6b64556-aa9e-4dd6-bf6f-366e47b05095" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
