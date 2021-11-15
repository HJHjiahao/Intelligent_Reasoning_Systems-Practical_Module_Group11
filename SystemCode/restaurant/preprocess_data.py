import pandas as pd
import numpy as np


customer_train_data = pd.read_csv("original_data/train_full.csv")
# train_data = pd.read_csv("original_data/train_full.csv")[:10000]
"""
73 columns, select 10 as features based on common sense.
customer_id, gender, location_type, id, (authentication_id),
vendor_category_en/vendor_category_id, delivery_charge, serving_distance,
prepration_time, vendor_rating, primary_tags/vendor_tag_name, 
(location_number_obj(same as location_number)), (id_obj(same as id))
"""
print(customer_train_data['location_type'].value_counts())
print(customer_train_data.head())
# dataset1 = data1[['customer_id','gender','location_type','id','OpeningTime','language','vendor_rating','serving_distance','vendor_tag_name','delivery_charge']]
customer_trainset = customer_train_data[['customer_id', 'gender', 'location_type', 'id',
                                         'vendor_category_en', 'delivery_charge', 'serving_distance',
                                         'prepration_time', 'vendor_rating', 'vendor_tag_name']]
# Drop duplicates based 'all' derived variables
# dataset1.drop_duplicates(['all'], inplace=True)

orders_data = pd.read_csv("original_data/orders.csv")
# orders_data = pd.read_csv("original_data/orders.csv")[:10000]
"""
26 columns, select 6 as features based on common sense.
akeed_order_id, customer_id, item_count, grand_total, is_rated, vendor_rating, 
delivery_time, vendor_id, LOCATION_TYPE
"""
print(orders_data.head())
orderset = orders_data[['customer_id', 'grand_total', 'vendor_rating',
                        'delivery_time', 'vendor_id', 'LOCATION_TYPE']]
print(customer_trainset.shape)
print(orderset.shape)

# prepare for merge
customer_trainset.rename(columns={"id": "vendor_id"}, inplace=True)
orderset.rename(columns={"vendor_rating": "order_rating"}, inplace=True)
customer_trainset['ID'] = customer_trainset[['customer_id', 'vendor_id']].apply(
    lambda row: '_'.join(row.values.astype(str)), axis=1)
orderset['ID'] = orderset[['customer_id', 'vendor_id']].apply(
    lambda row: '_'.join(row.values.astype(str)), axis=1)
train_df = pd.merge(customer_trainset, orderset, on='ID', how='inner')
print(train_df.shape)
# to check the repeated features
print(train_df.head)
train_df.rename(columns={"customer_id_x": "customer_id"}, inplace=True)
train_df.rename(columns={"vendor_id_x": "vendor_id"}, inplace=True)
train_df.drop(['customer_id_y', 'vendor_id_y', 'LOCATION_TYPE'], axis=1, inplace=True)
# since here, (len, 14)
train_df.to_csv('dataset/train_df.csv')

# def count_null(df, feature):
#     null_num = np.count_nonzero(df[feature].isnull())
#     print(null_num)
#     return null_num/df.shape[0]
#
#
# cols = ['serving_distance', 'delivery_charge',
#         'item_count', 'grand_total', 'vendor_rating']
# for i in cols:
#     print(i, 'null ratio :', count_null(train_df, i))
print(23)
