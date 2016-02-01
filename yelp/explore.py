# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 21:24:10 2016

@author: samsisto
"""
#imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wrangle

#subset datasets to look at just the state of Texas
texas = wrangle.yelp[wrangle.yelp.state.isin(['TX'])]
texas_bus = wrangle.business[wrangle.business.state.isin(['TX'])]

#clean city values
texas_bus['city'][texas_bus.city == 'West University Place'] = 'Houston'
texas_bus['city'][texas_bus.city == 'Liberty Hill'] = 'Austin'

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