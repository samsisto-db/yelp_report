
# coding: utf-8

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pprint import pprint
import re
import os
import glob
from bs4 import BeautifulSoup
import requests
get_ipython().magic(u'matplotlib inline')
from sklearn import metrics
import numpy as np
import unicodedata
import time


# In[16]:

#yelp data line breaks:
#130873
#460945


# In[2]:

path = '/Users/samsisto/Desktop/Yelp Users/yelp_users.json'
path2 = '/Users/samsisto/Desktop/Yelp Business/yelp_businesses.json'
path3 = '/Users/samsisto/Desktop/Yelp Reviews/yelp_reviews.json'

data = []
with open(path) as f:
    for line in f:
        data.append(json.loads(line))
users = pd.DataFrame(data)

data = []
with open(path2) as f:
    for line in f:
        data.append(json.loads(line))
business = pd.DataFrame(data)

data = []
with open(path3) as f:
    for line in f:
        data.append(json.loads(line))
reviews = pd.DataFrame(data)


# In[83]:

#business.schools

#objs = [df, pd.DataFrame(df['lists'].tolist()).iloc[:, :2]]
#pd.concat(objs, axis=1).drop('lists', axis=1)

#new_business = [business, pd.DataFrame(business['schools'].tolist()).iloc[:, :2]]
pd.DataFrame(business['schools'].tolist()).iloc[:]
newest_bus = pd.concat(new_business, axis=1).drop('schools', axis=1)

#business.state.value_counts().plot(kind="bar")

#business.schools.value_counts().plot(kind="bar")

y = sns.countplot(x='state', data=business, orient = "v")
y.set_title('Yelp Business Counts by State')
#y.set(xlabel='Content Rating', ylabel='Count')

#sns.boxplot(x='review_count', y='state', data=business)
x = sns.barplot(x='stars', y=0, data=newest_bus)
x.set_title('Mean Star Rating for 250 Nearest Businesses by College')
x.set(xlabel='Mean Star Rating', ylabel='Colleges')

#newest_bus.groupby(0).stars.mean()


# In[3]:

reviews_business = pd.merge(reviews, business, on='business_id', suffixes=('_review','_business'))
yelp = pd.merge(reviews_business, users, on='user_id')


# In[4]:

yelp.columns.values
#yelp['name_y'].head(10)


# In[4]:

yelp=yelp.rename(columns = {'votes_x':'review_votes', 'name_x':'business_name', 'review_count_x':'business_review_cnt'})
yelp=yelp.rename(columns = {'url_x':'business_url', 'name_y':'reviewer_name', 'review_count_y':'user_review_cnt'})
yelp=yelp.rename(columns = {'votes_y':'user_votes', 'url_y':'user_url'})


# In[5]:

def strip_votes(df, desired_col):
    funny,useful,cool = [],[],[]
    for instance in df[desired_col]:
        funny.append(instance[u'funny'])
        useful.append(instance[u'useful'])
        cool.append(instance[u'cool'])
    
    votes = pd.DataFrame([funny,useful,cool]).T
    df = pd.merge(df, votes, left_index=True, right_index=True)
    return df


# In[9]:

#yelp = strip_votes(yelp, 'review_votes')
yelp = strip_votes(yelp, 'user_votes')
yelp


# In[5]:

funny,useful,cool = [],[],[]
funny2,useful2,cool2 = [],[],[]
for review_votes in yelp['review_votes']:
    funny.append(review_votes[u'funny'])
    useful.append(review_votes[u'useful'])
    cool.append(review_votes[u'cool'])
    
for user_votes in yelp['user_votes']:
    funny2.append(user_votes[u'funny'])
    useful2.append(user_votes[u'useful'])
    cool2.append(user_votes[u'cool'])
df = pd.DataFrame([funny,useful,cool,funny2,useful2,cool2]).T


# In[6]:

yelp = pd.merge(yelp, df, left_index=True, right_index=True)


# In[11]:

yelp.rename(columns = {'0_x':'review_votes_funny', '1_x':'review_votes_useful', '2_x':'review_votes_cool'}, inplace=True)
yelp.rename(columns = {'0_y':'user_votes_funny', '1_y':'user_votes_useful', '2_y':'user_votes_cool'}, inplace=True)
yelp.drop('type_review', axis=1, inplace=True)
yelp.drop('type_business', axis=1, inplace=True)


# In[33]:

yelp[yelp.state.isin(['TX', 'NY'])]


# In[34]:

len(yelp[yelp.state.isin(['TX', 'NY'])].user_id.unique())
#len(yelp[yelp.state == 'NY'].user_id.unique())


# In[9]:

yelp.columns.values
#average_stars = averages stars given by that particular user
#stars_business =  #of stars a business received
#stars_review = # of stars given for that particular review
#type_review, type and type_business = drop this column, brings no useful information to the table


# ## Funny, useful and cool votes from a users review
# ## stars_business vs. review_votes_funny, useful, cool
# #### Is there a relationship between the number of funny, useful or cool votes that a review receives and the average number of stars for a particular business?

# In[12]:

#review_votes_funny, review_votes_useful, review_votes_cool

feature_cols = ['review_votes_funny', 'review_votes_useful', 'review_votes_cool']
sns.pairplot(yelp, x_vars=feature_cols, y_vars='stars_business', kind='reg')


# Contrary to the findings that we had in the Yelp homework that we did for class, it actually looks like there is a negative trend between "funny" votes and the number of stars that a business receives. Whereas we see a positive trend for "useful" and "cool" reviews and number of stars that a business receives.

# In[13]:

#Assembling the linear regression model

X = yelp[feature_cols]
y = yelp.stars_business

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X, y)
zip(feature_cols, linreg.coef_)


# In[20]:

#Compute RMSE with training and testing data sets

X = yelp[feature_cols]
y = yelp.stars_business

#train/test split?
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 123)
    
#create a model; fit
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
    
#return the error
np.sqrt(metrics.mean_squared_error(y_test, y_pred))


# ## User data frame exploration

# In[9]:

funny3,useful3,cool3 = [],[],[]
for votes in users['votes']:
    funny3.append(votes[u'funny'])
    useful3.append(votes[u'useful'])
    cool3.append(votes[u'cool'])

df3 = pd.DataFrame([funny3,useful3,cool3]).T
users = pd.merge(users, df3, left_index=True, right_index=True)
users.rename(columns = {0:'funny', 1:'useful', 2:'cool'}, inplace=True)


# In[13]:

feature_cols = ['funny', 'useful', 'cool']
sns.pairplot(users, x_vars=feature_cols, y_vars='average_stars', kind='reg')


# In[53]:

#user_votes_funny, user_votes_useful, user_votes_cool

feature_cols = ['user_votes_funny', 'user_votes_useful', 'user_votes_cool']
sns.pairplot(yelp, x_vars=feature_cols, y_vars='average_stars', kind='reg')


# In[14]:

users.columns.values


# In[15]:

sns.pairplot(users, x_vars='review_count', y_vars='average_stars', kind='reg')
#As review count increases, average star rating decreases


# ## Business data frame exploration

# In[16]:

business.columns.values


# In[32]:

#business.groupby('city').stars.mean()


# In[38]:

feat_cols = ['latitude', 'longitude', 'review_count']
sns.pairplot(business, x_vars=feat_cols, y_vars='stars', kind='reg')
#higher review count leads to higher average star rating for a business


# In[37]:

business.neighborhoods


# ## Reviews data frame exploration

# In[42]:

reviews.columns.values


# In[41]:

funny3,useful3,cool3 = [],[],[]
for votes in reviews['votes']:
    funny3.append(votes[u'funny'])
    useful3.append(votes[u'useful'])
    cool3.append(votes[u'cool'])

df3 = pd.DataFrame([funny3,useful3,cool3]).T
reviews = pd.merge(reviews, df3, left_index=True, right_index=True)
reviews.rename(columns = {0:'funny', 1:'useful', 2:'cool'}, inplace=True)


# In[43]:

feat_cols = ['funny', 'useful', 'cool']
sns.pairplot(reviews, x_vars=feat_cols, y_vars='stars', kind='reg')


# In[ ]:

## Categories


# In[26]:

#categories
#breaking out a list here to see different category ratings

#sns.barplot(x='stars_business', y='categories', data=yelp)
yelp.categories[0]


# In[18]:

cat1,cat2,cat3 = [],[],[]
for cat in yelp['categories']:
    if not cat:
        cat1.append('')
        cat2.append('')
        cat3.append('')
    elif len(cat) < 2:
        cat1.append(cat[0])
    elif len(cat) < 3:
        cat1.append(cat[0])
        cat2.append(cat[1])
    else:
        cat1.append(cat[0])
        cat2.append(cat[1])
        cat3.append(cat[2])
    
df = pd.DataFrame([cat1,cat2,cat3]).T
yelp = pd.merge(yelp, df, left_index=True, right_index=True)
yelp.rename(columns = {0:'cat1', 1:'cat2', 2:'cat3'}, inplace=True)


# In[26]:

yelp.groupby('cat1').stars_business.mean()


# ## Text

# In[39]:

#text
#bag of words approach here?

yelp.text


# ## Web scraping to get additional user information

# In[8]:

url1 = users.url[2]
url1


# In[9]:


print b.prettify()


# In[88]:

b.find('h4', text='Rating Distribution')
new_list


# In[29]:

url_list = users.url[2]
url_list


# In[32]:

users.head()


# In[46]:

users2 = users.iloc[:,4:6]
users2
new_dict = users2.set_index('user_id')['url'].to_dict()
new_dict


# In[8]:

users2 = users.iloc[:,4:6]
users2
new_dict = users2.set_index('user_id')['url'].to_dict()
new_dict

newest_dict = { key:value for key, value in new_dict.items() }
newest_dict
len(newest_dict)


# In[ ]:

users2 = users.iloc[:,4:6]
new_dict = users2.set_index('user_id')['url'].to_dict()

newest_dict = { key:value for key, value in new_dict.items() }

#r = requests.get(url1)
#b = BeautifulSoup(r.text, 'html.parser')

def get_header_text(b):
    new_list = b.html.find_all("h4")
    list_guy = []
    for item in new_list:
        list_guy.append(item.getText())
    return list_guy

def get_user_info(list):
    user_info = {}
    for i in list:
        header = b.find('h4', text=i)
        for item in header.next_siblings:
            if item != '\n': 
                user_info[i] = item.string
    return user_info


user_info = {}
for item in newest_dict:
    #time.sleep(5)
    r = requests.get(newest_dict[item])
    b = BeautifulSoup(r.text, 'html.parser')
    
    header_text = []
    
    header_text = get_header_text(b)
    user_info[item] = get_user_info(header_text)
    
    print user_info


# In[16]:

user_info_df = pd.DataFrame(user_info)
user_info_df


# In[19]:

file_name = '/Users/samsisto/Desktop/user_info.csv'
user_info_df.to_csv(file_name, encoding='utf-8')


# In[70]:

b.html.find_all("h4")


# In[ ]:



