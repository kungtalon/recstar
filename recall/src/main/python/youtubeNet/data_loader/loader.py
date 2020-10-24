import os
import numpy as np
import pandas as pd
import tensorflow as tf
from data_loader.processor import DataProcessor
from os.path import join, exists

base_dir = os.path.split(os.path.abspath(__file__))[0]
prior_raw_path = './data_loader/data/prior_raw.csv'
array_columns = [('hist_seq', 'int32'), ('y', 'float32'), ('sub_samples', 'int32')]
extra_columns = ['order_dow', 'order_hour_of_day', 'days_since_prior_order']

def load_tables(eval_set = 'prior'):
    path_dict = {'products': 'data/products.csv',
                 'orders': 'data/orders.csv',
                 'prior': 'data/order_products__prior.csv'
                }
    if eval_set == 'train':
        path_dict['train'] = 'data/order_products__train.csv'
    table_dict = {k : pd.read_csv(join(base_dir, table_path)) for k, table_path in path_dict.items()}
    
    orders = table_dict['orders']
    table_dict['orders'] = orders[orders['eval_set'] == eval_set].drop('eval_set', axis=1)
    
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
        if args.is_training:
            self.gen_prior_data()
        else:
            pass

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

    def gen_prior_data(self):
        try:
            for i in range(self.shard_count):
                assert exists(join(base_dir, f'prior_input_shard_{i}.csv'))
            print('sharded input data are prepared!')
        except AssertionError:
            all_inp_data = self.load_prior_input()
            print('generating sharded input data...')
            size_per_shard = np.ceil(len(all_inp_data) / self.shard_count)
            print(f'each shard with size of {int(size_per_shard)}')
            for i in range(self.shard_count):
                file_name = join(base_dir, f'data/prior_input_shard_{i}.csv')
                sharded_inp_data = all_inp_data.loc[i*size_per_shard:(i+1)*size_per_shard].copy()
                sharded_inp_data = self.processor.process_sharded_data(sharded_inp_data, self.args)
                sharded_inp_data.to_csv(file_name, sep=',', encoding='utf-8')
                print(f'writing {i}th shard to csv...')

    def gen_batch(self):
        for cur_shard in range(self.shard_count):
            print(f'now processing shard {cur_shard} / {self.shard_count}')
            self.cur_inp = self.load_csv_with_arrays(join(base_dir, f'prior_input_shard_{i}.csv'), array_columns).sample(frac=1)
            cur_len = len(self.cur_inp)
            for i in range(np.ceil(cur_len / self.batch_size)):
                batched_data = self.cur_inp[i * self.batch_size : (i+1) * self.batch_size]
                y = np.array(batched_data['y'].to_list())
                hist_seq = np.array(batched_data['hist_seq'].to_list())
                sub_samples = np.array(batched_data['sub_samples'].to_list())
                dow = np.array(batched_data['order_dow'].to_list())
                hod = np.array(batched_data['order_hour_of_day'].to_list())
                dense = np.array(batched_data['days_since_prior_order'].to_list())
                yield {'hist_seq': hist_seq, 
                       'y': y, 
                       'dow': dow, 
                       'hod': hod,
                       'sub_samples': sub_samples,
                       'dense': dense}

    def load_array(self, series, dtype='float32'):
        def aux(mylist):
            a = mylist[-1]
            b = a[:-1]
            mylist[-1] = b
            return np.array(mylist[1:], dtype=dtype)
        series = series.map(str.split).map(aux)
        return series

    def load_csv_with_arrays(self, path, columns):
        raw = pd.read_csv(path)
        for c, dtype in columns:
            raw[c] = self.load_array(raw[c], dtype)
        return raw

if __name__ == '__main__':
    print(load_tables('train'))