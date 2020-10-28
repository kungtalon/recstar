# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical
from data_loader.processor import DataProcessor
from os.path import join, exists

base_dir = os.path.split(os.path.abspath(__file__))[0]
data_dir = join(base_dir, 'data')
prior_raw_path = './data_loader/data/prior_raw.csv'
array_columns = [('order_list', 'int32'), ('last_order_list', 'int32'), ('sub_samples', 'int32')]

def load_tables(eval_set = 'prior'):
    path_dict = {'products': 'products.csv',
                 'orders': 'orders.csv',
                 'prior': 'order_products__prior.csv'
                }
    if eval_set == 'train':
        path_dict['train'] = 'order_products__train.csv'
    table_dict = {k : pd.read_csv(join(data_dir, table_path)) for k, table_path in path_dict.items()}
    
    orders = table_dict['orders']
    if eval_set == 'prior':
        table_dict['orders'] = orders[orders['eval_set'] == eval_set].drop('eval_set', axis=1)
    else:
        table_dict['orders'] = orders[orders['eval_set'] != 'test']
    
    return table_dict

class DataLoader:
    def __init__(self, args):
        self.args = args
        eval_set = 'prior' if args.is_training else 'train'
        self.tables = load_tables(eval_set)
        self.processor = DataProcessor(self.tables, eval_set)
        self.consts = self.processor.consts
        self.cur_inp = None
        self.batch_size = args.batch_size
        self.shard_count = args.shard_count
        self.dense_size = args.dense_size

    def load_prior_input(self):
        if exists(prior_raw_path):
            print('reading prior input dataframe...')
            prior_inp = self.load_csv_with_arrays(prior_raw_path, [('last_order_list', 'int32'), ('order_list', 'int32')])
            return prior_inp
        else:
            print('generating prior input dataframe')
            prior_inp = self.processor.get_prior_input()
            prior_inp.to_csv(prior_raw_path, sep=',', encoding='utf-8')
            return prior_inp

    def preload(self, is_training):
        if is_training:
            try:
                for i in range(self.shard_count):
                    assert exists(join(data_dir, f'prior_input_shard_{i}.csv'))
            except AssertionError:
                all_inp_data = self.load_prior_input()
                print('generating sharded input data...')
                size_per_shard = np.ceil(len(all_inp_data) / self.shard_count)
                print(f'each shard with size of {int(size_per_shard)}')
                for i in range(self.shard_count):
                    file_name = join(data_dir, f'prior_input_shard_{i}.csv')
                    sharded_inp_data = all_inp_data.loc[i*size_per_shard:(i+1)*size_per_shard].copy()
                    sharded_inp_data = self.processor.gen_subsamples(sharded_inp_data, self.args)
                    sharded_inp_data.to_csv(file_name, sep=',', encoding='utf-8')
                    print(f'writing {i}th shard to csv...')
            print('sharded input data are prepared!')
        else:
            try:
                assert exists(join(data_dir, f'train_input_data.csv'))
            except AssertionError:
                print('generating test input data...')
                all_inp_data = self.processor.get_train_input()
                file_name = join(data_dir, f'train_input_data.csv')
                all_inp_data.to_csv(file_name, sep=',', encoding='utf-8')
            print('test input data are prepared!')

    def gen_batch(self):
        for cur_shard in range(self.shard_count):
            print(f'now processing shard {cur_shard} / {self.shard_count}')
            self.cur_inp = self.load_csv_with_arrays(join(data_dir, f'prior_input_shard_{cur_shard}.csv'), array_columns)
            self.cur_inp = self.cur_inp.sample(frac=1)
            self.cur_inp = self.processor.process_sharded_data(self.cur_inp, self.args)
            cur_len = len(self.cur_inp)
            for i in range(int(np.ceil(cur_len / self.batch_size))):
                batched_data = self.cur_inp[i * self.batch_size : (i+1) * self.batch_size]
                sampled_y = np.array(batched_data['sampled_y'].to_list())
                hist_seq = np.array(batched_data['hist_seq'].to_list())
                hist_len = np.array(batched_data['hist_len'].to_list())
                sub_samples = np.array(batched_data['sub_samples'].to_list())
                dow = to_categorical(np.array(batched_data['order_dow'].to_list()), num_classes=7)
                hod = to_categorical(np.array(batched_data['order_hour_of_day'].to_list()), num_classes=24)
                dense = np.array(batched_data['days_since_prior_order'].to_list()).reshape(-1, self.dense_size)
                yield {'hist_seq': hist_seq,
                       'hist_len': hist_len,
                       'sampled_y': sampled_y,
                       'dow': dow,
                       'hod': hod,
                       'sub_samples': sub_samples,
                       'dense': dense}

    def gen_test_batch(self):
        self.cur_inp = self.load_csv_with_arrays(\
            join(data_dir, f'train_input_data.csv'), [('last_order_list', 'int32'), ('order_list', 'int32')])
        self.cur_inp = self.processor.process_test_data(self.cur_inp, self.args)
        cur_len = len(self.cur_inp)
        for i in range(int(np.ceil(cur_len / self.batch_size))):
            batched_data = self.cur_inp[i * self.batch_size : (i+1) * self.batch_size]
            order_id = np.array(batched_data['order_id'].to_list())
            y = np.array(batched_data['y'].to_list())
            hist_seq = np.array(batched_data['hist_seq'].to_list())
            hist_len = np.array(batched_data['hist_len'].to_list())
            dow = to_categorical(np.array(batched_data['order_dow'].to_list()), num_classes=7)
            hod = to_categorical(np.array(batched_data['order_hour_of_day'].to_list()), num_classes=24)
            dense = np.array(batched_data['days_since_prior_order'].to_list()).reshape(-1, self.dense_size)
            yield {'order_id': order_id,
                   'hist_seq': hist_seq,
                   'hist_len': hist_len,
                   'y': y,
                   'dow': dow,
                   'hod': hod,
                   'dense': dense}

    def parse_array(self, series, dtype='float32'):
        # transform from str type to numpy array
        def aux(mylist):
            if mylist[0] == '[':
                mylist = mylist[1:]
            if mylist[-1] == ']':
                mylist = mylist[:-1]
            if '[' == mylist[0][0]:
                mylist[0] = mylist[0][1:]
            if ']' == mylist[-1][-1]:
                mylist[-1] = mylist[-1][:-1]
            return np.array(mylist, dtype=dtype)
        return series.map(str.split).map(aux)

    def load_csv_with_arrays(self, path, columns):
        raw = pd.read_csv(path)
        for cname, dtype in columns:
            raw[cname] = self.parse_array(raw[cname], dtype)
        return raw

if __name__ == '__main__':
    print(load_tables('train'))
