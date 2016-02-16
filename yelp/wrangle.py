#imports
import pandas as pd
import ingest

#Call ingest.py and create initial dataframes
path1 = '/Users/samsisto/Desktop/Yelp Users/yelp_users.json'
path2 = '/Users/samsisto/Desktop/Yelp Business/yelp_businesses.json'
path3 = '/Users/samsisto/Desktop/Yelp Reviews/yelp_reviews.json'

users = ingest.load_json_to_df(path1)
business = ingest.load_json_to_df(path2)
reviews = ingest.load_json_to_df(path3)

#Merge dataframes based on business_id and user_id
reviews_business = pd.merge(reviews, business, on='business_id', suffixes=('_review','_business'))
yelp = pd.merge(reviews_business, users, on='user_id')

#Rename columns in merged yelp dataframe
yelp=yelp.rename(columns = {'votes_x':'review_votes', 'name_x':'business_name', 'review_count_x':'business_review_cnt'})
yelp=yelp.rename(columns = {'url_x':'business_url', 'name_y':'reviewer_name', 'review_count_y':'user_review_cnt'})
yelp=yelp.rename(columns = {'votes_y':'user_votes', 'url_y':'user_url'})

#Strip out the cool, funny and useful votes
def strip_votes(df, desired_col):
    funny,useful,cool = [],[],[]
    for instance in df[desired_col]:
        funny.append(instance[u'funny'])
        useful.append(instance[u'useful'])
        cool.append(instance[u'cool'])
    
    votes = pd.DataFrame([funny,useful,cool]).T
    df = pd.merge(df, votes, left_index=True, right_index=True)
    return df

yelp = strip_votes(yelp, 'review_votes')
yelp = strip_votes(yelp, 'user_votes')

#Rename newly created columns, drop types columns
yelp.rename(columns = {'0_x':'review_votes_funny', '1_x':'review_votes_useful', '2_x':'review_votes_cool'}, inplace=True)
yelp.rename(columns = {'0_y':'user_votes_funny', '1_y':'user_votes_useful', '2_y':'user_votes_cool'}, inplace=True)
yelp.drop('type_review', axis=1, inplace=True)
yelp.drop('type_business', axis=1, inplace=True)