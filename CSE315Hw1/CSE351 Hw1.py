#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install wordcloud')


# In[2]:


pip install nbconvert


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[4]:


originalData = pd.read_csv('OneDrive\Desktop\AB_NYC_2019.csv')
data = pd.read_csv('OneDrive\Desktop\AB_NYC_2019.csv').dropna(axis = 0, how = 'any')
display(data)


# In[5]:


data['price'].mean()


# In[6]:


data['price'].std()


# In[7]:


data['latitude'].min()


# In[8]:


data['latitude'].max()


# In[9]:


data['longitude'].min()


# In[10]:


data['longitude'].max()


# In[11]:


data['price'].min()


# In[12]:


data['price'].max()


# In[13]:


data['minimum_nights'].min()


# In[14]:


data['minimum_nights'].max()


# In[15]:


data['number_of_reviews'].min()


# In[16]:


data['number_of_reviews'].max()


# In[17]:


data['last_review'].min()


# In[18]:


data['last_review'].max()


# In[19]:


data['reviews_per_month'].min()


# In[20]:


data['reviews_per_month'].max()


# In[21]:


data['calculated_host_listings_count'].min()


# In[22]:


data['calculated_host_listings_count'].max()


# In[23]:


data['availability_365'].min()


# In[24]:


data['availability_365'].max()


# 1) First I obtained the uncleaned data set and printed its contents to have a priliminary understanding of the data.  To begin cleaning, I dropped every row with at least one missing column using the pandas dropna method. Afterwards, I printed the minumum and maximum values of each numeric (non word) feature of the data set to examine whether the numbers were reasonable without further inspection. For instance, I inspected the latitude and longitutde to ensure that they were within valid range of NYC. There were a number of rows that had relatively unlikely values, but I think that having a couple strange values could prove to be useful and interesting. It would be a diservice to myself to remove artifacts thinking they were errors.

# In[ ]:





# In[25]:


all_neighbourhoods = data.groupby('neighbourhood').filter(lambda neighbourhood: 5 < len(neighbourhood))


# In[26]:


neighbourhoods = all_neighbourhoods.groupby('neighbourhood').agg('mean')


# In[27]:


sortedByPrice = neighbourhoods.sort_values(by='price')['price']


# Bottom 5 Based on Price

# In[28]:


sortedByPrice.head()


# Top 5 Based on Price

# In[29]:


sortedByPrice.tail()


# In[30]:


plt.title('Price Variation Between Different Neighborhood Groups', fontsize = 15)
plt.xlabel('neighbourhoods', fontsize=10, color='red')
plt.ylabel('price', fontsize=10, color='blue')

data['price'] = data['price'].astype('float')
sortedByPrice.plot(kind = 'bar')
plt.show()


# In[ ]:





# In[31]:


reviewsVsAvailability = data['reviews_per_month'].corr(data['availability_365']) 
print(reviewsVsAvailability)


# In[83]:


dataplot = sns.heatmap(data.corr(method='pearson'), cmap="rainbow_r", annot=True)
plt.title('Heatmap Depicting Corelation between Aspects of NYC AirBnbs')


# In[ ]:





# In[138]:


sns.scatterplot(data=data, x='longitude', y='latitude', hue='neighbourhood_group')
plt.title('Latitudinal and Longitudinal Location of the 5 Boroughs')
plt.xlabel('longitude', fontsize=10, color='red')
plt.ylabel('latitude', fontsize=10, color='blue')


# In[149]:


lessThan1000 = data[(data['price'] < 1000)]
sns.scatterplot(data=lessThan1000, x='longitude', y='latitude', hue='price')
plt.title('Price of the AirBnb in Each Area')
plt.xlabel('longitude', fontsize=10, color='red')
plt.ylabel('latitude', fontsize=10, color='blue')


# By looking at the two scatterplots side by side, it is evident that the majority of the most expensive Airbnbs are in the upperbrooklyn to lower manhattan area, as is demonstrated by the dark purple points. However, it is interesting to consider that there are both very expensive and very inexpensive Airbnbs spread across all five boroughs of NYC. 

# In[ ]:





# In[101]:


words = data['name'].values
wordCloud = WordCloud().generate(str(words))
plt.title('Word Cloud Using Words in AirBnb Names')
plt.imshow(wordCloud)
plt.show()


# 

# In[152]:


data.groupby('neighbourhood').agg('mean').sort_values(by='calculated_host_listings_count')['calculated_host_listings_count']


# In[153]:


data.groupby('host_id').agg('count').sort_values(by='calculated_host_listings_count')['calculated_host_listings_count']


# In[140]:


busiest_host_ids = data.groupby('host_id').agg('mean').sort_values(by='calculated_host_listings_count')['calculated_host_listings_count'].tail().index.values


# In[141]:


busiest_host_listings = data.loc[data['host_id'].isin(busiest_host_ids)]


# In[142]:


busiest_host_listings.groupby('neighbourhood').agg('count')


# In[143]:


busiest_host_listings.groupby('host_id').agg('mean').sort_values('price')[['price']]


# In[144]:


data['price'].mean()


# In[145]:


busiest_host_listings.groupby('host_id').agg('mean').sort_values('availability_365')[['availability_365']]


# In[146]:


data['availability_365'].mean()


# In[147]:


busiest_host_listings.groupby('host_id').agg('mean').sort_values('reviews_per_month')[['reviews_per_month']]


# In[148]:


data['reviews_per_month'].mean()


# First, I grouped the data by neighbourhood and sorted by calculated_host_listings_count to determine which neighbourhoods had the largest number of listings. After that, I did grouped the data by host_id and sorted it by calculated_host_listings_count similarly to the previous step. However, I also used the index and the values in the index to create a seperate table that used the host_id as a basis. From this, I was able to display a table depicting host_id vs various other factors. Finally, I printed the average values of each aspect of the original data set. From this, I now had the values of each element of the busiest regions and hosts, and that of the entire data set. I was able to use this to make predictions on why these hosts are the busiest. For instance, the average availablity of the averge AirBnb is 114.88629865279101. However, for the busiest hosts, the availablity was about 200-300. Furthermore, this explains why the price of the busiest host is approximately 100$ more than that of the dataset as a whole.

# In[ ]:





# In[ ]:





# In[114]:


dataplot = sns.heatmap(data.corr(method='pearson'), cmap="rainbow_r", annot=True)
plt.title('Heatmap Depicting Corelation between Aspects of NYC AirBnbs')


# In[132]:


minNightsPerPrice = all_neighbourhoods.groupby('minimum_nights').agg('mean').sort_values(by='price')['price']


# In[154]:


sns.scatterplot(data=data, x='price', y='minimum_nights')
plt.title('Price of AirBnb Versus Minimum Nights Stayed')
plt.xlabel('price', fontsize=10, color='red')
plt.ylabel('minimum_nights', fontsize=10, color='blue')


# The first graph is a heatmap that demonstrates the corelation between every element of the dataset. I think that this is the most interesting model of the entire project. Before starting the assignment, I expected for there to be significant correlations between price and neighbourhood or price and availability. However, I was shocked to discover that the correlation between the elements was mostly approximately 0, meaning that there was no correlation between the elements. The maximum correlation was -0.33 between number of reviews and id. I expected a much larger correlation between a number of the elements. Of course, it is possible that I did something wrong, but if not then its a very interesting observation. The second graph represents how price may affect the minimum nights stayed at an AirBnb. It is evident that the vast majority of the points are in the 0-2000$ range, especially leaning towards the 0. It is logical to conclude that the cheaper an AirBnb is the liklier it is for people to stay there. The interesting thing about this graph is the outliers. Of course, most of the time outliers are a bad thing. However, I think its interesting to consider the outliers in this graph. There are some people who payed 10,000$ for 100-200 nights and others who payed nearly nothing but stayed for 1000+ nights. As such, this is a perfect demosntration of how a single outlier in either axis can skew the data.
