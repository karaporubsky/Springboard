#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import necessary libraries
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns


# In[5]:


from matplotlib import rcParams
sns.set_style("whitegrid")
sns.set_context("poster")


# In[6]:


#load data
from sklearn.datasets import load_boston
import pandas as pd
boston = load_boston()


# In[7]:


#investigate data
boston.keys()


# In[8]:


boston.data.shape


# In[9]:


print(boston.feature_names)


# In[10]:


print(boston.DESCR)


# In[11]:


#put data into dataframe
bos = pd.DataFrame(boston.data)
bos.head()


# In[12]:


#add column names
bos.columns = boston.feature_names
bos.head()


# In[13]:


#add price data from "target" data
print(boston.target.shape)
bos['PRICE'] = boston.target
bos.head()


# In[14]:


#explore data set
bos.describe()


# In[15]:


#create scatter plot to look at select features
plt.scatter(bos.CRIM, bos.PRICE)
plt.xlabel("Per capita crime rate by town (CRIM)")
plt.ylabel("Housing Price")
plt.title("Relationship between CRIM and Price")


# In[16]:


plt.scatter(bos.RM, bos.PRICE)
plt.xlabel("Rooms per Dwelling (RM)")
plt.ylabel("Housing Price")
plt.title("Relationship between RM and Price")


# In[17]:


plt.scatter(bos.PTRATIO, bos.PRICE)
plt.xlabel("Pupil Teacher Ratio by town (PTRATIO)")
plt.ylabel("Housing Price")
plt.title("Relationship between PTRATIO and Price")


# In[18]:


#using Seaborn
sns.regplot(y = 'PRICE', x = 'RM', data = bos, fit_reg = True)


# In[19]:


#create histogram
plt.hist(np.log(bos.CRIM))
plt.title('CRIM')
plt.xlabel('Crime rate per capita')
plt.ylabel('Frequency')
plt.show()


# In[20]:


#what would it look like if we didn't do the log
plt.hist(bos.CRIM)
plt.title('CRIM')
plt.xlabel('Crime rate per capita')
plt.ylabel('Frequency')
plt.show()


# In[21]:


#running the regression model
import statsmodels.api as sm
from statsmodels.formula.api import ols
m = ols('PRICE ~ RM', bos).fit()
print(m.summary())


# In[22]:


#fitting linear regression with sklearn
from sklearn.linear_model import LinearRegression
X = bos.drop('PRICE', axis = 1)
lm = LinearRegression()
lm


# In[25]:


#fit linear model
lm.fit(X, bos.PRICE)


# In[27]:


#look at intercept and coef
lm.intercept_
lm.coef_


# In[28]:


print('Estimated intercept coefficient: {}'.format(lm.intercept_))


# In[29]:


print('Number of coefficients: {}'.format(len(lm.coef_)))


# In[30]:


#the coefs
pd.DataFrame({'features': X.columns, 'estimatedCoefficients': lm.coef_})[['features', 'estimatedCoefficients']]


# In[31]:


#predict
lm.predict(X)[0:5]


# In[32]:


#evaluate model
print(np.sum((bos.PRICE - lm.predict(X)) ** 2))


# In[33]:


print(np.sum((lm.predict(X) - np.mean(bos.PRICE)) ** 2))


# In[ ]:




