import numpy as np

from .exception import UnSupportException, ParameterException


class ZDataFrame:

    response_encode_type = ["binary", "multiclass"]

    def __init__(self, data, response_column=None, ignores=None, response_encode=None, category=None):

        self.data = data.copy()
        self.response_column = response_column
        self.ignores = ignores
        self.feature_cols = self.data.columns.difference(
            [self.response_column] if self.response_column else []).difference(
            self.ignores if self.ignores else [])
        self.category = category
        self.response_encode = response_encode

        # reconstruct the index
        origin_columns = self.data.columns
        self.data = self.data.reset_index()
        new_column = self.data.columns.difference(origin_columns)
        self.data = self.data.drop(new_column, axis=1)

        if response_column and self.category is None:
            self.category = np.unique(self.data[self.response_column])

    def features(self):
        return self.data[self.feature_cols].values

    def impute(self, method="median"):

        if method == "median":
            self.data[self.feature_cols] = self.data[self.feature_cols].fillna(self.data[self.feature_cols].median())
        elif method == "mean":
            self.data[self.feature_cols] = self.data[self.feature_cols].fillna(self.data[self.feature_cols].mean())
        else:
            raise UnSupportException(method)

        return self

    def response(self):

        if not self.response_column:
            raise UnSupportException("data set has no response.")

        if self.response_encode:
            response_category = np.array(
                [int(np.where(self.category == v)[0][0]) for v in self.data[self.response_column]])

            if self.response_encode == "binary":
                return response_category.reshape(-1, 1)
            elif self.response_encode == "multiclass":
                response = np.zeros((self.data.shape[0], len(self.category)))
                response[np.arange(response.shape[0]), response_category] = 1
                return response

            else:
                raise UnSupportException(self.response_encode)
        else:
            return self.data[self.response_column].values.reshape(-1, 1)

    def split(self, fraction=0.75):

        random_index = self.data.index.values.copy()
        np.random.shuffle(random_index)

        cut = int(len(random_index) * fraction)
        sample_index = random_index[:cut]
        remain_index = random_index[cut + 1:]

        sample = ZDataFrame(self.data.loc[sample_index].copy(), self.response_column, self.ignores,
                            self.response_encode, self.category)
        remain = ZDataFrame(self.data.loc[remain_index].copy(), self.response_column, self.ignores,
                            self.response_encode, self.category)

        return sample, remain

    def bootstrap(self, bagging_size=5):

        bootstrap = list()

        for i in range(bagging_size):
            chosen_index = np.random.choice(self.data.index, self.data.shape[0])
            new_data = ZDataFrame(self.data.loc[chosen_index].copy(), self.response_column, self.ignores,
                                  self.response_encode, self.category)
            bootstrap.append(new_data)

        return bootstrap

    def cv(self, folds=5):

        random_index = self.data.index.values.copy()
        np.random.shuffle(random_index)

        cv = list()
        sub_set_size = int(np.floor(self.data.shape[0] / folds))
        for i in range(folds):
            sample_index = np.concatenate([random_index[0:i * sub_set_size], random_index[(i + 1) * sub_set_size:]])
            remain_index = random_index[i * sub_set_size:(i + 1) * sub_set_size]
            sample = ZDataFrame(self.data.loc[sample_index].copy(), self.response_column, self.ignores,
                                self.response_encode, self.category)
            remain = ZDataFrame(self.data.loc[remain_index].copy(), self.response_column, self.ignores,
                                self.response_encode, self.category)
            cv.append((sample, remain))

        return cv

    def shape(self):
        return self.data.shape

    def reverse_response(self, response_encoded):
        if self.response_encode:
            return np.array([self.category[i] for i in np.argmax(response_encoded, axis=1)])
        else:
            return response_encoded

    def __call__(self):
        return self.data
