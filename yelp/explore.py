# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 21:24:10 2016

@author: samsisto
"""
#imports
import seaborn as sns
import wrangle

#subset datasets to look at just the state of Texas
texas = wrangle.yelp[wrangle.yelp.state.isin(['TX'])]
texas_bus = wrangle.business[wrangle.business.state.isin(['TX'])]

#clean city values
texas_bus['city'][texas_bus.city == 'West University Place'] = 'Houston'
texas_bus['city'][texas_bus.city == 'Liberty Hill'] = 'Austin'
texas['city'][texas.city == 'West University Place'] = 'Houston'
texas['city'][texas.city == 'Liberty Hill'] = 'Austin'

#Clean categories column, use get dummies to create columns
texas_bus['categories'] = texas_bus['categories'].apply(str)
texas_bus['categories'] = texas_bus['categories'].str.strip('[]')
texas_bus['categories'] = texas_bus['categories'].str.replace("u'", "'")
texas_bus['categories'] = texas_bus['categories'].str.replace('u"', "'")
texas_bus['categories'] = texas_bus['categories'].str.replace('"', "'")
texas_bus['categories'] = texas_bus['categories'].str.replace("'", '')
texas_bus['categories'] = texas_bus['categories'].str.replace(", ", ",")
texas_bus['categories'] = texas_bus['categories'].str.replace(" ", "_")

df = texas_bus['categories'].str.get_dummies(sep=',')
texas_bus = texas_bus.merge(df, left_index = True, right_index = True)

#Create binary city mapping
texas_bus['city_binary'] = texas_bus.city.map({'Houston':0, 'Austin':1})

#New texas_bus dataframe containing 5 star reviews and 2.5 star reviews and
#lower
tb = texas_bus[(texas_bus.stars == 5) | (texas_bus.stars == 1) | \
(texas_bus.stars == 1.5) | (texas_bus.stars == 2) | (texas_bus.stars == 2.5)]

#map star ratings
tb['stars'] = tb.stars.map({1:0, 1.5:0, 2:0, 2.5:0, 5:1})

#Columns to drop
col_drop = ['business_id', 'categories', 'city', 'full_address', 'latitude',\
'longitude', 'name', 'neighborhoods', 'open', 'photo_url', 'review_count', \
'schools', 'stars', 'state', 'type', 'url']

categories_dtm = tb.drop(col_drop, 1)

#count of business captured in each city
print 'Count of businesses captured in each city:'
print texas_bus.city.value_counts()

#average star rating by city
print 'Average star rating by city:'
print texas_bus.groupby('city').stars.mean()

#count of star ratings by city
print 'Count of star ratings by city:'
print texas_bus.groupby('city').stars.value_counts()

#boxplot and scatter plot of number of review counts for each business, by city
sns.plt.title('Number of review counts for each business, by city')
sns.plt.ylim(0,200)
sns.stripplot(x="city", y="review_count", data=texas_bus, jitter=True)
sns.boxplot(x="city", y="review_count", data=texas_bus)