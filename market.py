import math
import numpy as np
import pandas as pd
from sklearn import *
from catboost import Pool, CatBoostRegressor
from collections import Counter
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cluster import KMeans

pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows', 300)
np.set_printoptions(threshold=100)

def extract_features(data):
    data['name'] = pd.Categorical.from_array(data.name).codes
    data['category'] = pd.Categorical.from_array(data.category_name).codes
    data['brand'] = pd.Categorical.from_array(data.brand_name).codes
    data['description'] = pd.Categorical.from_array(data.item_description).codes
    df = pd.concat([data.name, data.category, data.brand, data.shipping, data.item_condition_id, data.description], axis=1)
    return df


train_data = pd.read_csv('train.tsv', sep='\t', header=0)
print(train_data)
values = {'item_description': '', 'brand' :0}
train_data = train_data.fillna(value=values)

#text = train_data['item_description']
#vectorizer = HashingVectorizer()
#vectorizer.fit(text)
#print(vectorizer.vocabulary_)
#vector = vectorizer.transform(text)

#print(vector[0,:].toarray()[0])

#kmeans = KMeans(n_clusters=2, random_state=0).fit(vector)
#print(kmeans.labels_)
#print(train_data)

train_data['price'] = train_data['price'].apply(lambda x: math.log(1+x))
train_label = train_data['price']
train_data = extract_features(train_data)
print(train_data)

train_pool = Pool(train_data, train_label)
model = CatBoostRegressor(iterations=1000, loss_function='RMSE', random_seed=21, learning_rate=0.15)
model.fit(train_pool)

