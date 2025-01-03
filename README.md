# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.  Use the chardet library to detect the file's encoding by reading a portion of the file.
2.  Read the CSV file using pandas.read_csv() and store the dataset in a DataFrame (data).
3.  Use data.info() to check the structure (e.g., data types and number of non-null values).
4.  Data Preparation (Feature and Label Extraction)
5.  Split the dataset into training and testing sets using train_test_split from sklearn
6.  cv.fit_transform(x_train): Fit and transform the training data into a bag-of-words representation.
7.  Initialize the Support Vector Classifier (SVC) and train the model using svc.fit(x_train, y_train)
8. Use the trained SVC model to make predictions on the test data (y_pred = svc.predict(x_test)).
9. Evaluate the model's performance using the accuracy_score function from sklearn.metrics.
```

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: M.Mahalakshmi
RegisterNumber: 24900868 
*/
```
```
import chardet 
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
print(result)
import pandas as pd 
data = pd.read_csv("spam.csv",encoding='Windows-1252')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
print(y_pred)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print(accuracy)
```

## Output:

{'encoding':	'Windows-1252',	'confidence':	0.7270322499829184,	'language':	''}

![Screenshot 2024-12-26 081808](https://github.com/user-attachments/assets/e2a57b77-5253-44e9-b6fa-22c93438f4e9)

![Screenshot 2024-12-26 081817](https://github.com/user-attachments/assets/a3aa04d4-e01f-4904-ada5-e24e5516d071)

v1										0					
v2										0					
Unnamed:	2				5522
Unnamed:	3				5560
Unnamed:	4				5566
dtype:	int64
 
 array(["Sorry,	I'll	call	later",	"Sorry,	I'll	call	later",
        "Sorry,	I'll	call	later",	...,	"Sorry,	I'll	call	later",
        "Sorry,	I'll	call	later",	"Sorry,	I'll	call	later"],	dtype=object)

0.003587443946188341

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
