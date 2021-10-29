import pandas as pd
import numpy as np
from models import User_CF


customer_vendor_initial = pd.read_csv("dataset/train_df.csv")[
    ['customer_id', 'vendor_id', 'vendor_rating', 'vendor_tag_name', 'order_rating']]
# a = customer_vendor_ratings['order_rating'].value_counts()
customer_vendor_initial['order_rating'].replace(0, np.nan, inplace=True)
valid_rating_mean = round(customer_vendor_initial['order_rating'].mean(), 4)
customer_vendor_initial['order_rating'].replace(np.nan,
                                                valid_rating_mean,
                                                inplace=True)
cf = User_CF(customer_vendor_initial, valid_rating_mean)
recommendation, _ = cf.recommend_vendors(customer_id='ZZV76GY', recommendation_num=5, neighbors_num=30)
# recommendation, _ = cf.recommend_vendors(customer_id='new_user', recommendation_num=5, neighbors_num=30)
'''
test_full = pd.read_csv("original_data/test_full.csv")[
    ['customer_id', 'id', 'vendor_rating']]

test_full.rename(columns={"id": "vendor_id"}, inplace=True)
test_df = test_full.groupby(['customer_id', 'vendor_id']).mean().reset_index()
ids = test_df['customer_id'].drop_duplicates().values.tolist()
results = []
# for index, row in test_df.iterrows():
for id in ids:
    result = []
    result.append(id)
    recommendation, is_existing = cf.recommend_vendors(
        customer_id=id, recommendation_num=10, neighbors_num=30)
    result.append(recommendation)
    if is_existing:
        vendor2rating = {}
        # vendors = cf.customer_vendor_matrix.columns.values.tolist()
        # ratings = cf.customer_vendor_matrix.loc[id].values.tolist()
        vendors = test_df[test_df['customer_id'] == id]['vendor_id'].values.tolist()
        ratings = test_df[test_df['customer_id'] == id]['vendor_rating'].values.tolist()
        for i in range(0, len(ratings)):
            if ratings[i] != np.nan:
                vendor2rating[vendors[i]] = ratings[i]
        vendor2rating = dict(sorted(vendor2rating.items(), key=lambda x: x[1],
                                    reverse=True))
        num = min(len(recommendation), len(vendor2rating))
        recommendation = recommendation[:num]
        keys = vendor2rating.keys()[:num]
        count = 0
        for rec in recommendation:
            if rec in keys:
                count += 1
        result.append(round(count/num, 4))
    results.append(result)
print(6)
'''