from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


class User_CF(object):
    def __init__(self, customer_vendor_full, valid_rating_mean):
        """
        user based collaborative filtering class for restaurant recommendation
        :param customer_vendor_full: the merging result of customers' records preprocessed and
                                     orders' records preprocessed
        :param valid_rating_mean: mean of non-nan order ratings in customer_vendor_full
        """
        super(User_CF, self).__init__()
        self.customer_vendor_full = customer_vendor_full
        self.customer_vendor_ratings = self.select_features()
        self.customer_vendor_matrix = self.customer_vendor_ratings.pivot(
            index='customer_id', columns='vendor_id', values='mean_rating')  # (26779, 100)
        self.rating_matrix = self.customer_vendor_matrix.fillna(0).values.astype(np.float32)
        self.valid_rating_mean = valid_rating_mean
        self.vendor2rating = self.get_vendors_mean()
        self.customer_similarity,  = self.get_similarity()

    def select_features(self):
        """
        select the fundamental features, the orders of same customer and vendor are handled.
        :return: information to generate customer-vendor matrix
        """
        customer_vendor_ratings = self.customer_vendor_full[
            ['customer_id', 'vendor_id', 'order_rating']]
        customer_vendor_ratings = customer_vendor_ratings.groupby(['customer_id', 'vendor_id']) \
            .mean().reset_index()  # 69814 remained
        customer_vendor_ratings.rename(columns={'order_rating': 'mean_rating'}, inplace=True)
        return customer_vendor_ratings

    def get_similarity(self, ):
        """
        calculate the cosine/pearson coefficient similarity of users
        :return: similarity matrix
        """
        customer_cos_similarity = cosine_similarity(self.rating_matrix, self.rating_matrix)
        customer_cos_similarity = pd.DataFrame(customer_cos_similarity,
                                               index=self.customer_vendor_matrix.index,
                                               columns=self.customer_vendor_matrix.index)
        # customer_pearson_similarity = np.corrcoef(self.rating_matrix,
        #                                           self.rating_matrix,)
        # customer_pearson_similarity = pd.DataFrame(customer_pearson_similarity,
        #                                            index=self.customer_vendor_matrix.index,
        #                                            columns=self.customer_vendor_matrix.index)
        return customer_cos_similarity,
        # return customer_pearson_similarity  run too slowly

    def get_rmse(self, y_true, y_pred):
        """
        calculate Root Mean Squared Error
        :param y_true: ground truth
        :param y_pred: prediction
        :return:
        """
        return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

    def get_loss(self, get_rating, neighbors_num=0):
        """
        calculate loss of predicting ratings
        :param get_rating: predict rating
        :param neighbors_num: the top n similar neighbors
        :return:
        """
        id_pairs = zip(self.customer_vendor_ratings['customer_id'],
                       self.customer_vendor_ratings['vendor_id'])
        y_pred = np.array([get_rating(customer, vendor, neighbors_num) for (customer, vendor) in id_pairs])
        y_true = np.array(self.customer_vendor_ratings['mean_rating'])
        return self.get_rmse(y_true, y_pred)

    def get_rating(self, customer_id, vendor_id, neighbors_num=0):
        """
        predict rating
        :param customer_id:
        :param vendor_id:
        :param neighbors_num: the top n similar neighbors
        :return:
        """
        if vendor_id not in self.customer_vendor_matrix:
            return self.valid_rating_mean
        else:
            customer_similarities = self.customer_similarity[customer_id].copy()
            vendor_ratings = self.customer_vendor_matrix[vendor_id].copy()
            none_rating_customers = vendor_ratings[vendor_ratings.isnull()].index
            vendor_ratings.drop(none_rating_customers, inplace=True)
            customer_similarities.drop(none_rating_customers, inplace=True)

            if neighbors_num == 0:
                mean_rating = np.dot(
                    customer_similarities, vendor_ratings) / customer_similarities.sum()
            else:
                if len(customer_similarities) == 1:
                    if vendor_ratings:
                        mean_rating = (vendor_ratings.sum() + self.valid_rating_mean) / 2
                    else:
                        mean_rating = self.valid_rating_mean
                else:
                    neighbors_num = min(neighbors_num, len(customer_similarities))
                    customer_similarities = customer_similarities.values
                    vendor_ratings = vendor_ratings.values
                    customer_indices = np.argsort(customer_similarities)
                    customer_similarities = customer_similarities[customer_indices][-neighbors_num:]
                    vendor_ratings = vendor_ratings[customer_indices][-neighbors_num:]
                    mean_rating = np.dot(
                        customer_similarities, vendor_ratings) / customer_similarities.sum()
        return round(mean_rating, 4)

    def recommend_vendors(self, customer_id, recommendation_num=1, neighbors_num=0, allow_repeated=False):
        """

        :param customer_id: recommend for customer_id
        :param recommendation_num: number of vendors recommended
        :param neighbors_num: the top n similar neighbors
        :param allow_repeated: whether recommend vendors where
                               users have already bought food
        :return:
        """
        if customer_id not in self.customer_vendor_matrix.index.values:
            num = min(recommendation_num, len(self.vendor2rating))
            vendors = list(self.vendor2rating.keys())[:num]
            vendors_info = self.customer_vendor_full[self.customer_vendor_full['vendor_id'].isin(vendors)]
            vendors_info = vendors_info[['vendor_id', 'vendor_rating', 'vendor_tag_name']] \
                .drop_duplicates('vendor_id').values.tolist()
            for i in range(0, num):
                vendors_info[i].append(round(self.vendor2rating[vendors[i]], 4))
            return vendors_info, False  # vendor_id, rating, tag, predicted_rating

        customer_vendors = self.customer_vendor_matrix.loc[customer_id].copy()
        for vendor in self.customer_vendor_matrix:
            if pd.notnull(customer_vendors.loc[vendor]):
                if not allow_repeated:
                    customer_vendors.loc[vendor] = 0
            else:
                customer_vendors.loc[vendor] = self.get_rating(customer_id, vendor, neighbors_num)

        vendors = customer_vendors.sort_values(ascending=False)[:recommendation_num]
        vendors_info = self.customer_vendor_full[self.customer_vendor_full['vendor_id'].isin(vendors.index)]
        vendors_info = vendors_info[['vendor_id', 'vendor_rating', 'vendor_tag_name']]\
            .drop_duplicates('vendor_id').values.tolist()
        for vendor_info in vendors_info:
            vendor_info.append(vendors.loc[vendor_info[0]])
        return vendors_info, True  # vendor_id, rating, tag, predicted_rating

    def get_vendors_mean(self):
        """
        calculate the mean rating of each vendors,
        which could be used as the knowledge base to recommend vendors for new customers
        :return:
        """
        matrix = self.customer_vendor_matrix.mean(axis=0).to_frame()
        vendor_ids = matrix.index.values
        vendor_ratings = matrix.values
        vendor2rating = {}
        indices = min(vendor_ids.shape[0], vendor_ratings.shape[0])
        for i in range(0, indices):
            vendor2rating[vendor_ids[i]] = vendor_ratings[i][0]
        vendor2rating = dict(sorted(vendor2rating.items(), key=lambda x: x[1], reverse=True))
        return vendor2rating


