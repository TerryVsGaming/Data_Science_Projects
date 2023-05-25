#!/usr/bin/env python
# coding: utf-8

# In[73]:


import pandas as pd
from datetime import datetime
import numpy as np
import datetime as dt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score
import seaborn as sns
import plotly.express as px


# ### Q1

# In[74]:


weather_data = pd.read_csv('OneDrive\Desktop\weather_data.csv')
energy_data = pd.read_csv('OneDrive\Desktop\energy_data.csv',parse_dates=True)


# In[75]:


display(weather_data)


# In[76]:


display(energy_data)


# In[77]:


weather_data['time'] = pd.to_datetime(weather_data['time'], unit='s', origin='unix')


# In[78]:


display(weather_data)


# In[79]:


display(energy_data)


# In[80]:


energy_data['Date & Time'] = pd.to_datetime(energy_data['Date & Time'])


# In[81]:


display(energy_data['Date & Time'])


# In[82]:


#sum of energy per day
energy_data.groupby(energy_data['Date & Time'].dt.date)['use [kW]'].sum()


# In[83]:


energy_sum_per_day = energy_data.groupby(energy_data['Date & Time'].dt.date)['use [kW]'].sum().reset_index()


# In[84]:


energy_sum_per_day['Date & Time'] = pd.to_datetime(energy_sum_per_day['Date & Time'])


# In[85]:


#summary is just less detailed version of icon so we drop it
merged = pd.merge_asof(weather_data, energy_sum_per_day, left_on='time', right_on='Date & Time').drop(columns=['Date & Time', 'summary'])


# In[86]:


merged = pd.concat([merged.drop(columns=['icon']),pd.get_dummies(merged.icon, drop_first=True)], axis = 1)


# In[87]:


merged


# In[88]:


weather_data.index


# In[89]:


weather_data.info()


# In[90]:


energy_sum_per_day.info()


# ### Q2

# In[91]:


train = merged.query("time < '2014-12-01'" )


# In[92]:


train


# In[93]:


test = merged.query("time >= '2014-12-01'" )


# In[94]:


test


# ### Q3

# In[95]:


#train the model
train = train.dropna()
test = test.dropna()

x_train = train.drop(columns=['time', 'use [kW]']) 
y_train = train['use [kW]']

x_test = test.drop(columns=['time', 'use [kW]']) 
y_test = test['use [kW]']

linear_regressor = LinearRegression()  # create object
linear_regressor.fit(x_train, y_train)  #linear regression

Y_pred = linear_regressor.predict(x_test)  #makes predictions


# In[96]:


rmse = mean_squared_error(y_test, Y_pred)


# In[97]:


rmse


# In[98]:


energy_sum_per_day['use [kW]'].mean()


# #### 3) The model is quite bad. As one can see from the root mean squared error (rmse) value calculated above, the model doesn't work very well at all. I think that this makes some sense as the model uses data that works in a somewhat backwards day. It seems to use the daily values to estimate the hourly usage, which seems somewhat backwards. Due to this reverse nature of the model, it makes perfect sense for the root mean squared error to indicate a poor model.

# In[99]:


prediction_df = pd.DataFrame({'date':test.time, 'prediction':Y_pred})


# In[100]:


prediction_df


# In[101]:


prediction_df.to_csv("cse351_hw2_Shvartsman_Terrence_114311609_linear_regression.csv", index=False)


# In[ ]:





# ### Q4

# In[102]:


merged['high/low'] = np.where(merged.temperature >= 35, 1, 0)


# In[103]:


merged


# In[ ]:





# In[104]:


#train the model
test = merged.query("time >= '2014-12-01'" )
train = merged.query("time < '2014-12-01'" )

train = train.dropna()
test = test.dropna()

x_train = train.drop(columns=['time', 'use [kW]', 'high/low']) 
y_train = train['high/low']

x_test = test.drop(columns=['time', 'use [kW]','high/low']) 
y_test = test['high/low']

logistic_regressor = LogisticRegression()  # create object for the class
logistic_regressor.fit(x_train, y_train)  # perform linear regression

Y_pred = logistic_regressor.predict(x_test)  # make predictions


# In[105]:


Y_pred


# In[106]:


my_f1_score = f1_score(y_test, Y_pred)


# In[107]:


my_f1_score


# In[108]:


energy_data['Date & Time'].dt.hour #if hr >= 6 &&


# In[109]:


classification_df = pd.DataFrame({'date':test.time, 'prediction':Y_pred})


# In[110]:


classification_df


# In[111]:


classification_df.to_csv("cse351_hw2_Shvartsman_Terrence_114311609_logistic_regression.csv", index=False)


# ### Q5

# In[112]:


energy_data['time_of_day'] = np.where((energy_data['Date & Time'].dt.hour >= 6) & (energy_data['Date & Time'].dt.hour < 19), 'day', 'night')


# In[113]:


energy_data


# In[114]:


furnace_plot = px.scatter(energy_data, x='Date & Time', y='Furnace [kW]', color = 'time_of_day', title="Analyzing Energy Usage of the Furnace[kW] throughout 2014")
furnace_plot.update_layout(xaxis_title='Time', title_x=0.5)
furnace_plot.show()


# In[115]:


#select a device and then plot it
first_floor_plot = px.scatter(energy_data, x='Date & Time', y='First Floor lights [kW]', color = 'time_of_day', title="Analyzing Energy Usage of the First Floor Lights[kW] throughout 2014")
first_floor_plot.update_layout(xaxis_title='Time', title_x=0.5)
first_floor_plot.show()


# ### I think that its very interesting that there is at large number of points in both plots that are near the bottom. Furthermore, it is interesting that all of these points are during the day. However, upon consideration, this makes sense since most people are out during the day so most appliances are either off or barely used during the day. Of course this isnt always true as we have certain points that are outliers in the y axis, meaning they use an abnormal amount of kW. However, this also makes sense since we define day to be 6am-7pm and most people are still home at 6am and get back home around 5pm, leaving 2 hours from 5-7 where a lot of energy would be used. 
# 
# ### The most interesting thing about these two graphs, though, is how the first floor light usage vs time is a constant nearly bell plot shaped curve, while the furnace vs time graph has these rectangular clumps. This makes sense as during the cold months the furnace would be used throughout the entire day and month to keep the houses warm. As such, there is a large concentration of points ranging from 0kW to half a kW from about October to April. Then from May to June there is nearly no usage at all. I am not entirely sure what the reason behind the points in July-September is. However, due to it being almost entirely "night" points, it might be relatively cold at night, or at least cold enough to require the furnace to be used somewhat. 

# In[ ]:





# In[ ]:




