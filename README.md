# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the CSV file into a Pandas DataFrame and inspect the data using .head(), .tail(), and .info().
2. Drop unnecessary columns (e.g., 'sl_no') and convert categorical columns to appropriate data types.
3. Convert categorical variables (e.g., 'gender', 'ssc_b', etc.) into numerical codes using astype('category') and .cat.codes.
4. Split the data into training and testing sets using train_test_split().
5. Initialize and train a Logistic Regression model on the training data using clf.fit().
6. Predict the target variable for the test set using clf.predict().
7. Calculate and display the confusion matrix and accuracy score using confusion_matrix() and accuracy_score().

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SANJEV R M
RegisterNumber: 212223040186
*/
from google.colab import drive
drive.mount('/content/gdrive')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
```
## Read the Dataset:
```
a=pd.read_csv('/content/Placement_Data_Full_Class (1).csv')
a
```
## Output:
![image](https://github.com/user-attachments/assets/cb81a177-2b4d-4139-b8b3-87896d323419)
## Info :
```
a.head()
a.tail()
a.info()

```
## Output:

![Screenshot 2024-10-16 094514](https://github.com/user-attachments/assets/dc2d6eb9-e900-4c03-bf7a-6107d63e72ab)
![Screenshot 2024-10-16 094655](https://github.com/user-attachments/assets/fdbc132c-9454-4142-b5c2-2d6fef972b89)
![image](https://github.com/user-attachments/assets/9ad47c29-1ea9-4738-a858-c26db8ed6da9)

## Drop unnecessary columns
```
a=a.drop(['sl_no'],axis=1)
a
```
## Output:
![Screenshot 2024-10-16 094934](https://github.com/user-attachments/assets/689b4970-82a2-48cb-80ad-f0f8742c1869)

## Encoding Categorical Variables:
```
a['gender']=a['gender'].astype('category')
a['ssc_b']=a['ssc_b'].astype('category')
a['hsc_b']=a['hsc_b'].astype('category')
a['hsc_s']=a['hsc_s'].astype('category')
a['degree_t']=a['degree_t'].astype('category')
a['workex']=a['workex'].astype('category')
a['specialisation']=a['specialisation'].astype('category')
a['status']=a['status'].astype('category')
a.info()

a['gender']=a['gender'].cat.codes
a['ssc_b']=a['ssc_b'].cat.codes
a['hsc_b']=a['hsc_b'].cat.codes
a['hsc_s']=a['hsc_s'].cat.codes
a['degree_t']=a['degree_t'].cat.codes
a['workex']=a['workex'].cat.codes
a['specialisation']=a['specialisation'].cat.codes
a['status']=a['status'].cat.codes
a.info()
```
## Output:
![Screenshot 2024-10-16 095131](https://github.com/user-attachments/assets/492de0b1-f138-4ece-993d-fdaf91dd1105)
![image](https://github.com/user-attachments/assets/c6a67e9c-c407-4c35-a9f6-196cccea0570)

## Splitting Data:
```
x=a.iloc[:,:-1].values
y=a.iloc[:,-1].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

```
## Logistic Regression Model:
```
clf=LogisticRegression()
clf.fit(x_train,y_train)
```
## Output:
![Screenshot 2024-10-16 095604](https://github.com/user-attachments/assets/dfb4f0ef-4d3a-4d42-9aef-97fb7e3d1d1d)

## Model Evaluation:
```
ypred=clf.predict(x_test)
print(ypred)
from sklearn.metrics import confusion_matrix,accuracy_score
cf=confusion_matrix(y_test,ypred)
print(cf)
accuracy=accuracy_score(y_test,ypred)
print(accuracy*100)

```
## Output :
![Screenshot 2024-10-16 095740](https://github.com/user-attachments/assets/0465fa3d-423e-4df1-8704-884c3cb27317)






## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
