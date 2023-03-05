#!/usr/bin/env python
# coding: utf-8

# # Final Project

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ## Dataset 

# In[3]:


dataset = pd.read_csv(r'C:\Fall 2022\Python for Business Analytics\insurance1.csv')
dataset.head(10)


# ## Shortisted Features from Dataset

# In[4]:


dataset = dataset[['age', 'bmi', 'children', 'charges']]

dataset


# In[ ]:





# ## Impact of Age on Insurance Charge

# Here we take Age as our independent variable and charges as our Dependent variable

# In[5]:


X_Age = dataset[['age']]
y_Ch = dataset[['charges']]


# In[6]:


X_Age


# Corelation of Age and Insurance Charges

# In[7]:


plt.scatter(X_Age, y_Ch)
plt.show()


# Scaled our data to standardize the data within a particular range and speed up the algorithm

# In[8]:


from sklearn.preprocessing import StandardScaler


# In[9]:


scaler = StandardScaler()


# In[10]:


scaled = scaler.fit_transform(X_Age)
scaled


# In[11]:


scaled_y = scaler.fit_transform(y_Ch)


# In[12]:


scaled_y


# Trained the Data

# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error


# In[14]:


X_train1, X_test1, y_train1, y_test1 = train_test_split(scaled, scaled_y, test_size = 0.20, random_state = 123)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 123)


# Performed Linear Regression 

# In[15]:


regressor = LinearRegression()


# In[16]:


regressor.fit(X_train1, y_train1)


# In[17]:


y_pred1 = regressor.predict(X_test1)


# In[44]:


yp = pd.DataFrame(y_pred1, columns=['Prediction'])
xt = pd.DataFrame(X_test1, columns=['Test'])
xypt = [yp, xt]
xyptt = pd.merge([yp], [xt])
xyptt


# In[19]:


print('Cof', regressor.coef_) 
print('Intercept', regressor.intercept_)
print('Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test1, y_pred1)))


# The Coefficient tells us the increase in value of insurance charge if we increase our age by 1
# 

# In[20]:


x = np.arange(50)

plt.rcParams["figure.figsize"] = (12,4)
plt.plot(x, y_pred1[:50], c = "red")
plt.plot(x, y_test1[:50], c = "blue")
plt.show()


# In[21]:


plt.scatter(X_test1,y_test1)
plt.plot(X_test1, y_pred1, color = 'red')
plt.show()


# In[ ]:





# ## Impact of BMI on Insurance Charge

# In[22]:


X_Bmi = dataset[['bmi']]
y_Ch = dataset[['charges']]


# In[23]:


X_Bmi


# In[24]:


scaled = scaler.fit_transform(X_Bmi)
scaled


# In[25]:


plt.scatter(X_Bmi, y_Ch)
plt.show()


# In[78]:


X_train2, X_test2, y_train2, y_test2 = train_test_split(scaled, scaled_y, test_size = 0.20, random_state = 123)


# In[79]:


regressor = LinearRegression()


# In[80]:


regressor.fit(X_train2, y_train2)


# In[81]:


y_pred2 = regressor.predict(X_test2)


# In[82]:


print('Cof', regressor.coef_)
print('Intercept', regressor.intercept_)
print('Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test2, y_pred2)))


# The Coefficient tells us the increase in value of insurance charge if we increase the value of BMI by 1

# In[145]:


x = np.arange(50)

plt.rcParams["figure.figsize"] = (12,4)
plt.plot(x, y_pred2[:50], c = "red")
plt.plot(x, y_test2[:50], c = "blue")
plt.show()


# In[146]:


plt.scatter(X_test2,y_test2)
plt.plot(X_test2, y_pred2, color = 'red')
plt.show()


# In[ ]:





# ## Impact of Age, BMI, and No. of Children on Insurance Charge

# In[28]:


X_All = dataset[['age','bmi','children']]
y_Ch = dataset[['charges']]


# In[29]:


X_All


# In[30]:


scaled = scaler.fit_transform(X_All)
scaled


# In[31]:


X_train4, X_test4, y_train4, y_test4 = train_test_split(scaled, scaled_y, test_size = 0.20, random_state = 123)


# In[32]:


regressor = LinearRegression()


# In[33]:


regressor.fit(X_train4, y_train4)


# In[34]:


y_pred4 = regressor.predict(X_test4)


# In[35]:


print('Cof', regressor.coef_)
print('Intercept', regressor.intercept_)
print('Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test4, y_pred4)))


# Here the Coefficient tells us the increase in value of insurance charge if we increase our age, BMI value, and children

# In[36]:


x = np.arange(50)

plt.rcParams["figure.figsize"] = (12,4)
plt.plot(x, y_pred4[:50], c = "red")
plt.plot(x, y_test4[:50], c = "blue")
plt.show()


# In[ ]:





# In[ ]:




