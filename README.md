<H3>ENTER YOUR NAME: KESAV DEEPAK SRIDHARAN</H3>
<H3>ENTER YOUR REGISTER NO. : 212223230104</H3>
<H3>EX. NO.1</H3>
<H3>DATE: </H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("/content/Churn_Modelling.csv")
df.head()

df.isnull().sum()

df.duplicated()

df.describe()

df.info()

encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])
df['Geography'] = encoder.fit_transform(df['Geography'])
df['Surname'] = encoder.fit_transform(df["Surname"])

scaler = StandardScaler()
scaler.fit_transform(df)

X=df.iloc[:,:-1].values
X

y=df.iloc[:,-1].values
y

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
```


## OUTPUT:
<img width="1356" height="223" alt="image" src="https://github.com/user-attachments/assets/4d81ed77-7c51-4f5a-ab12-3f8e6f7242b8" />

<img width="174" height="587" alt="image" src="https://github.com/user-attachments/assets/ecdbb5b8-52ad-42f8-9f6b-095c29e76572" />

<img width="186" height="506" alt="image" src="https://github.com/user-attachments/assets/d823537c-5334-48b8-af58-2da5cc2999b0" />

<img width="1401" height="310" alt="image" src="https://github.com/user-attachments/assets/8eadda4f-742a-4b53-b754-f2b627d371a6" />

<img width="417" height="419" alt="image" src="https://github.com/user-attachments/assets/e7305138-51e1-48fd-af64-fa5f9beeeff7" />

<img width="574" height="269" alt="image" src="https://github.com/user-attachments/assets/f7f31def-b302-4dda-9c27-25e5849bb93f" />

<img width="630" height="259" alt="image" src="https://github.com/user-attachments/assets/f4b58dd2-e02a-4e36-93a5-9b54cfa24b4b" />

<img width="278" height="34" alt="image" src="https://github.com/user-attachments/assets/8f63c636-7714-48f3-9ab8-a251de72b509" />

<img width="189" height="106" alt="image" src="https://github.com/user-attachments/assets/fb69299d-56f0-4871-86ac-bb8bb1c4598c" />


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


