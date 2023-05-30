# Ex-06-Feature-Transformation 
# Aim:
To read and perform feature transformation for the given dataset.

# Explanation:
Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column (feature) and transform the values, which are useful for our further analysis. It is a technique by which we can boost our model performance.

# Algorithm:

# STEP 1
Read the given Data

# STEP 2
Clean the Data Set using Data Cleaning Process

# STEP 3
Apply Feature Transformation techniques to all the features of the data set

# STEP 4
Save the data to the file

# PROGRAM :
```
Name : Thanika sree B
Register numnber : 212222100055

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

df = pd.read_csv("/content/Data_to_Transform.csv")
df

df.head()

df.isnull().sum()

df.info()

df.describe()

df1 = df.copy()

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = 1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()
```
# OUTPUT:
# Dataset:
![image](https://github.com/Thanikasreeb/Ex-06-Feature-Transformation/assets/119557910/2392384f-0556-4b19-be22-7a1ac4b277ee)

# HEAD :
![image](https://github.com/Thanikasreeb/Ex-06-Feature-Transformation/assets/119557910/84db3077-1bf2-46a9-89d2-64ed0ffa47e4)

# Null data:
![image](https://github.com/Thanikasreeb/Ex-06-Feature-Transformation/assets/119557910/44048fd3-2108-4219-acda-8fa43632dc17)

# Information:
![image](https://github.com/Thanikasreeb/Ex-06-Feature-Transformation/assets/119557910/9e3efbfa-46d7-498a-a369-66452376f46b)

# Description:
![image](https://github.com/Thanikasreeb/Ex-06-Feature-Transformation/assets/119557910/f52b1f3e-0b0b-4716-ba13-e7f3959bb38a)

# Highly Positive Skew:
![image](https://github.com/Thanikasreeb/Ex-06-Feature-Transformation/assets/119557910/170d89dd-312e-4523-aba4-e2f8400e2490)

# Highly Negative Skew:
![image](https://github.com/Thanikasreeb/Ex-06-Feature-Transformation/assets/119557910/c3873981-9180-45d5-bd9d-fee4b08271a6)

# Moderate Positive Skew:
![image](https://github.com/Thanikasreeb/Ex-06-Feature-Transformation/assets/119557910/5910be82-f575-4af7-8eaf-4fd57b400b93)

# Moderate Negative Skew:
![image](https://github.com/Thanikasreeb/Ex-06-Feature-Transformation/assets/119557910/6bbb7eed-5f95-4d12-828c-d1a273187aca)

# Log of Highly Positive Skew:
![image](https://github.com/Thanikasreeb/Ex-06-Feature-Transformation/assets/119557910/0156c90b-f2d6-4bc5-b025-b097fa530059)

# Log of Moderate Positive Skew:
![image](https://github.com/Thanikasreeb/Ex-06-Feature-Transformation/assets/119557910/d0003104-d4bf-4b4c-8b37-df9fb569c991)

# Reciprocal of Highly Positive Skew:
![image](https://github.com/Thanikasreeb/Ex-06-Feature-Transformation/assets/119557910/933fffc1-9a0a-4f1a-9843-339d698af502)

# Square root tranformation:
![image](https://github.com/Thanikasreeb/Ex-06-Feature-Transformation/assets/119557910/e1dd3868-2970-4a6d-a106-2d8883cf4814)

# Power transformation of Moderate Positive Skew:
![image](https://github.com/Thanikasreeb/Ex-06-Feature-Transformation/assets/119557910/2939b0d7-b05f-440c-9de3-f1317b8e6eb1)

# Power transformation of Moderate Negative Skew:
![image](https://github.com/Thanikasreeb/Ex-06-Feature-Transformation/assets/119557910/71a2e0e6-2b64-41b2-8b41-f0c0a655acf5)

# Quantile transformation:
![image](https://github.com/Thanikasreeb/Ex-06-Feature-Transformation/assets/119557910/1d02207f-38c1-4e5d-bb70-f02957903cea)

# Result :
Thus, Feature transformation is performed and executed successfully for the given dataset.

















