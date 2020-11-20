import os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

data_dir = os.path.split(os.path.abspath(__file__))[0]
raw_train_path = os.path.join(data_dir, 'train')
raw_test_path = os.path.join(data_dir, 'test')
train_path = os.path.join(data_dir, 'train_custom')
test_path = os.path.join(data_dir, 'test_custom')

class DataLoader:
    def __init__(self):
        self.label_col = 'click'
        self.train = None
        self.test = None

    def load_raw_csv(self, name):
        return pd.read_csv(data_dir + name)
    
    def x_columns(self, df):
        return set(list(df.columns)) - set(["id", "site_id", "app_id", "hour", "dt_hour", "device_id", "device_ip", "click"])

    def to_date_column(self, df):
        df['dt_hour'] = pd.to_datetime(df['hour'], format='%y%m%d%H')
        df['year'] = df['dt_hour'].dt.year
        df['month'] = df['dt_hour'].dt.month // 4
        df['day'] = df['dt_hour'].dt.day // 11
        df['hourofday'] = df['dt_hour'].dt.hour // 2  # 减小维度到12
        df['dayofweek'] = df['dt_hour'].dt.dayofweek
        return df

    def get_batch_generator(self, df, batch_size):
        x = df[self.x_columns(df)]
        y = df[self.label_col]
        total_len = len(df)
        batch_num = int(np.ceil(total_len / batch_size))
        for i in range(batch_num):
            yield {'x': x[i*batch_size:(i+1)*batch_size], 
                   'y': y[i*batch_size:(i+1)*batch_size]}

    def filter_features(self, df):
        cols = df.columns
        featdim_dict = dict(df.nunique())
        normal_feats = [c_name for c_name in cols if featdim_dict[c_name] > 1 and featdim_dict[c_name] < 20]
        return df[normal_feats]

    def use_small_dataset(self):
        if os.path.exists(train_path) and os.path.exists(test_path):
            self.train = pd.read_csv(train_path)
            self.test = pd.read_csv(test_path)
            return 

        train = pd.DataFrame()
        chunksize = 10 ** 6
        num_of_chunk = 0
        for chunk in pd.read_csv(os.path.join(data_dir, 'train'), chunksize=chunksize):
            num_of_chunk += 1
            train = pd.concat([train, chunk.sample(frac=.05, replace=False, random_state=123)], axis=0)
            print('Processing Chunk No. ' + str(num_of_chunk))     
    
        train.reset_index(inplace=True)
        len_train = len(train)

        df = pd.concat([train, pd.read_csv(raw_test_path)]).drop(['index', 'id'], axis=1)

        df = self.to_date_column(df)
        df = self.filter_features(df)
        df = pd.get_dummies(df, columns=df.columns[1:])
        train = df[:len_train]
        test = df[len_train:]

        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        
        del df
        self.train = train
        self.test = test

    @property
    def train_set(self):
        return self.train.drop(self.label_col, axis=1).values, self.train[self.label_col].values

    @property
    def test_set(self):
        return self.test.values[:, 1:]