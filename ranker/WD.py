import os
import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.models import DeepFM
from deepctr.estimator import DeepFMEstimator
from deepctr.estimator.inputs import input_fn_tfrecord
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
import time

lbe_map_path = './lbe_map3.pkl'
raw_train_path = './data/train.csv'
raw_test_path = './data/test.csv'
train_record_path = './data/train_records4'
test_record_path = './data/test_records4'
save_dir = './deepFM_log3/'
time_feats = ['day', 'hourofday', 'dayofweek']

def my_auc(labels, predictions):
    auc_metric = tf.keras.metrics.AUC(name="my_auc")
    auc_metric.update_state(y_true=labels, y_pred=predictions['pred'])
    return {'auc': auc_metric}

def to_date_column(df):
    df['dt_hour'] = pd.to_datetime(df['hour'], format='%y%m%d%H')
    df['day'] = df['dt_hour'].dt.day
    df['hourofday'] = df['dt_hour'].dt.hour
    df['dayofweek'] = df['dt_hour'].dt.dayofweek
    return df

def gen_record(is_training=True):
    data_path = raw_train_path if is_training else raw_test_path
    out_path = train_record_path if is_training else test_record_path
    chunksize = 10 ** 6

    if os.path.exists(out_path):
        print('tf_records exists!')
        return
    
    lbe_map = {}
    x_columns = []
    if os.path.exists(lbe_map_path):
        with open(lbe_map_path, 'rb') as f:
            lbe_map = pkl.load(f)
        x_columns = [k for k in lbe_map.keys()]
    else:
        train_data = pd.read_csv(raw_train_path)
        test_data = pd.read_csv(raw_test_path)
        data = pd.concat([train_data, test_data])
        feat_ndim = dict(data.drop(['click', 'hour'], axis=1).nunique())
        for key in feat_ndim.keys():
            if feat_ndim[key] < 10000:
                x_columns.append(key)
                lbe_map[key] = LabelEncoder()
                lbe_map[key].fit(data[key].unique())
        with open(lbe_map_path, 'wb') as f:
            pkl.dump(lbe_map, f)
        del data
    x_columns += time_feats
    print(x_columns)

    t0 = time.time()
    with tf.io.TFRecordWriter(out_path) as writer:
        num_of_chunk = 0
        for chunk in pd.read_csv(data_path, chunksize=chunksize):
            num_of_chunk += 1
            if is_training:
                samples = chunk.sample(frac=.7, replace=False, random_state=123).reset_index()
            else:
                samples = chunk.reset_index()
            for c in lbe_map.keys():
                samples[c] = lbe_map[c].transform(samples[c])
            
            samples = to_date_column(samples)

            random_arr = np.random.random(len(samples))
            for i in range(len(samples)):
                if is_training and samples['click'][i] == 0 and random_arr[i] < 0.8:   # 负样本采样
                    continue
                feature_map = {}
                for c in x_columns:
                    feature_map[c] = tf.train.Feature(int64_list=tf.train.Int64List(value=[samples[c][i]]))
                feature_map['label'] = tf.train.Feature(float_list=tf.train.FloatList(value=[samples['click'][i]]))
                """ 2. 定义features """
                example = tf.train.Example(features = tf.train.Features(feature = feature_map))

                """ 3. 序列化,写入"""
                serialized = example.SerializeToString()
                writer.write(serialized)
                
                if i % 20000 == 0:
                    print(f'current chunk number: {num_of_chunk}, index: {i}')
            t1 = time.time()
            print(f'processing speed: {(t1 - t0) / num_of_chunk}s/chunk')
    print('done processing tf_record')

def train():
    # 1.generate feature_column for linear part and dnn part

    with open(lbe_map_path, 'rb') as f:
        lbe_map = pkl.load(f)
    sparse_features = [k for k in lbe_map.keys()] + time_feats

    dnn_feature_columns = []
    linear_feature_columns = []
    feat_dims = {}
    max_dim = 0
    for k in lbe_map.keys():
        feat_dims[k] = len(lbe_map[k].classes_)
        max_dim = max(max_dim, feat_dims[k])
    feat_dims['day'] = 31
    feat_dims['hourofday'] = 24
    feat_dims['dayofweek'] = 7

    for i, feat in enumerate(sparse_features):
        dnn_feature_columns.append(tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_identity(feat, feat_dims[feat]), 8))
        linear_feature_columns.append(tf.feature_column.categorical_column_with_identity(feat, feat_dims[feat]))
    # for feat in dense_features:
    #     dnn_feature_columns.append(tf.feature_column.numeric_column(feat))
    #     linear_feature_columns.append(tf.feature_column.numeric_column(feat))

    # 2.generate input data for model

    feature_description = {k: tf.FixedLenFeature(dtype=tf.int64, shape=1) for k in sparse_features}
    # feature_description.update(
        # {k: tf.FixedLenFeature(dtype=tf.float32, shape=1) for k in dense_features})
    feature_description['label'] = tf.FixedLenFeature(dtype=tf.float32, shape=1)

    train_model_input = input_fn_tfrecord(train_record_path, feature_description, 'label', batch_size=256,
                                          num_epochs=3, shuffle_factor=10)
    test_model_input = input_fn_tfrecord(test_record_path, feature_description, 'label',
                                         batch_size=4000000, num_epochs=1, shuffle_factor=0)


    # 3.Define Model,train,predict and evaluate
    model = DeepFMEstimator(linear_feature_columns, dnn_feature_columns, 
                            dnn_hidden_units=(128, 64), model_dir=save_dir,
                            l2_reg_linear=0.005, task='binary', dnn_optimizer='Adam')
    model = tf.estimator.add_metrics(model, my_auc)

    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' start training!')
    # model.train(train_model_input)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' done training!')

    eval_result = model.evaluate(test_model_input)

    print(eval_result)
    os.system('say "你的代码训练完了"')

if __name__ == '__main__':
    gen_record()
    gen_record(False)
    train()