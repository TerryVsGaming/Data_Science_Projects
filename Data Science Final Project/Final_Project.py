#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd
from datetime import datetime
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

import seaborn as sns
import plotly.express as px

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# In[58]:


#Original Data
test = pd.read_csv(r'OneDrive\Desktop\test.csv')
train = pd.read_csv(r'OneDrive\Desktop\train.csv')


# In[59]:


test


# In[60]:


train


# ## EDA

# ### Cleaning the Data

# In[61]:


train.describe()


# In[62]:


train.info()


# In[63]:


#correlation of non-numeric features
sns.pairplot(train.drop(columns=['PassengerId']))
plt.show()


# In[64]:


#remove outlier high fare, remove id, name, ticket (ticket #) /drop fare >= 400 in test and train?


# In[65]:


np.where(train.Fare >= 400, 1, 0).sum()


# In[66]:


np.where(test.Fare >= 400, 1, 0).sum()


# In[67]:


np.where(train.Fare >= 200, 1, 0).sum()


# In[68]:


np.where(test.Fare >= 200, 1, 0).sum()


# In[69]:


#Removing outliers and removing unnecessary columns (cleaning the data)
test = test.query("Fare < 400").drop(columns=['PassengerId', 'Name', 'Ticket'])
train = train.query("Fare < 400").drop(columns=['PassengerId', 'Name', 'Ticket'])


# In[70]:


test


# In[71]:


train


# ### Relation Between Socio-Economic Status and Other Features

# In[72]:


# how does class relate to age, sex and family member on board
#more old ppl in class 1 so no outliers, but in 2 and 3 there are so many more oyung ppl that the old ppl become outliers



sns.catplot(data=train, x="Pclass", y="Age", kind="box", hue="Sex").set(title="Relationship Between Socio-Economic Status, Sex, and Age")
plt.show()


# In[73]:


#
train.groupby(['Pclass'])['Fare'].mean()


# In[74]:


train.groupby(['Pclass', 'Parch'])['Fare'].mean()


# In[75]:


#Bad plot tbh

sns.catplot(data=train, x="Sex", y="Parch", kind="box")
plt.show()


# In[76]:


train.query("Sex == 'female' & Parch > 0").shape


# In[77]:


train.groupby(['Pclass','Sex']).size()


# In[78]:


#Bad table tbh
train.groupby(['Pclass','Age']).size()


# In[79]:


train.groupby(['Pclass','Parch']).size().plot.bar()
plt.show()


# ### Relation Between Survival Status and Other Features

# In[80]:



sns.catplot(data=train, x="Survived", y="Age", kind="box", hue="Sex").set(title="Relationship Between Survival Status, Sex, and Age")
plt.show()


# In[81]:


train.groupby(['Survived'])['Fare'].mean()


# In[82]:


train.groupby(['Survived','Sex']).size()


# ### Correlation Between All Numeric Features

# In[83]:


#CORRELATION STUFF
train.corr()


# In[84]:


test


# In[85]:


train


# In[86]:


#logistic, KNN and 1 more (random_forest)


# In[87]:


train.isnull().sum()


# In[88]:


train.columns


# ## Modeling and Question Answering

# ### Prepping the Data for Modeling

# In[103]:


prepped_data = prep_data(train)
prepped_data_x = prepped_data.drop(columns = ['Survived'])
prepped_data_y = prepped_data['Survived']

train_test_list = train_test_split(prepped_data)
my_train = train_test_list[0]
my_test = train_test_list[1]


# In[116]:


my_train


# ### Functions for Modeling

# In[105]:


def prep_data(data):
    droped_train = data.drop(columns=['Cabin']).dropna()
    sex_dummies = pd.get_dummies(droped_train.Sex, drop_first=True)
    embarked_dummies = pd.get_dummies(droped_train.Embarked, drop_first=True)

    new_train = pd.concat([droped_train.drop(columns=['Sex','Embarked']), sex_dummies, embarked_dummies], axis = 1)
    return new_train

def build_model(train_x, train_y, test_x, model_type, data_for_cv_x, data_for_cv_y):
    if model_type == "logistic_regression":
        model = LogisticRegression()
        cv_scores = cross_val_score(model, data_for_cv_x, data_for_cv_y, cv=5)
    
    elif model_type == "KNN":
        model  = KNeighborsClassifier(n_neighbors=3)
        cv_scores = cross_val_score(model, data_for_cv_x, data_for_cv_y, cv=5)  
    else:
        model = RandomForestClassifier(random_state = 0)
        cv_scores = cross_val_score(model, data_for_cv_x, data_for_cv_y, cv=5)
        
    model.fit(train_x, train_y)
    res = model.predict(test_x)
    return res, cv_scores


# ### Creating the 3 Models: Logistic Regression, K-Nearest-Neighbors, and Random Forest

# In[106]:


my_train_x = my_train.drop(columns=['Survived'])
my_train_y = my_train['Survived']

my_test_x = my_test.drop(columns=['Survived'])
my_test_y = my_test['Survived']

lr_model = build_model(my_train_x, my_train_y, my_test_x, "logistic_regression", prepped_data_x, prepped_data_y)
lr_pred, lr_cv_scores = lr_model


knn_model = build_model(my_train_x, my_train_y, my_test_x, "KNN", prepped_data_x, prepped_data_y)
knn_pred, knn_cv_scores = knn_model

rf_model = build_model(my_train_x, my_train_y, my_test_x, "Random Forest", prepped_data_x, prepped_data_y)
rf_pred, rf_cv_scores = rf_model


# In[107]:


lr_pred


# In[108]:


knn_pred


# In[109]:


rf_pred


# In[110]:


def print_all(test_y, y_prediction):
    my_f1_score = round(f1_score(test_y, y_prediction) , 4)
    my_recall_score = round(recall_score(test_y, y_prediction), 4)
    my_precision_score = round(precision_score(test_y, y_prediction), 4)
    my_accuracy_score = round(accuracy_score(test_y, y_prediction), 4)
    
    
    print('f1_score: {}'.format(my_f1_score))
    print('recall_score: {}'.format(my_recall_score))
    print('precision_score: {}'.format(my_precision_score))
    print('accuracy_score: {}'.format(my_accuracy_score))


# ### Evaluating the Performance of Each Model

# In[111]:


print_all(my_test_y, lr_pred)


# In[112]:


print_all(my_test_y, knn_pred)


# In[113]:


print_all(my_test_y, rf_pred)


# ### Evaluating Performance After Cross Validation

# In[114]:


#Accuracy of Model with Cross Validation
lr_cv_mean = round(np.mean(lr_cv_scores),4)
knn_cv_mean = round(np.mean(knn_cv_scores),4)
rf_cv_mean = round(np.mean(rf_cv_scores),4)

print('Mean Accuracy with Cross Validation Set of Logistic Regression Model: {}'.format(lr_cv_mean))
print('Mean Accuracy with Cross Validation Set of K-Nearest-Neighbors Model: {}'.format(knn_cv_mean))
print('Mean Accuracy with Cross Validation Set of Random Forest Model: {}'.format(rf_cv_mean))


# In[ ]:





# <!-- odds = probability / 1-probability
# 0 < odds <= 1 
# -inf <= log (odds) <= inf
# 
# in linerar regression wed have y = Bo + B1x1...
# 
# the algorithm, logistic regression, picks the probability and coresponding liine so that the probability of that point is as close to 0 or 1
# -> for any x, minimizes distance between that point and 0 or 1 depending on if its closer to 0 or 1
# with logistic we have log(odds) = B0 + B1x1... which goes from -inf to inf 
# we convert this back to probability
# 
# the line becomes less linear and becomes a better fit
# -> you do log of odds to get the actual probability
# 
# you input a bunch of x's, the slope of x, and the interecept, and get a y for each x. And then you can this logistic regression on the y to get the probability. 
# 
# we dont have probabilities, but we have this mechanism of B0 + B1X1... to predict probabilities, which gives odds
# so you log(odds) -->
