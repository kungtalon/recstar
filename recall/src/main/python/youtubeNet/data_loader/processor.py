import json
import numpy as np
import pandas as pd
import pickle as pkl
from os.path import exists


class DataProcessor:
    def __init__(self, tables, eval_set='prior'):
        self.mode = eval_set
        self.tables = tables
        self.consts = self.get_consts()

    def get_prior_input(self):
        '''
        每条记录所需字段：
        [user_id,] last_order_list, reorder_list, order_dow, order_hour_of_day, days_since_prior_order
        '''
        m_days_po, sd_days_po = self.consts['m_days_po'], self.consts['sd_days_po']
        origin_data = self.tables['orders']
        origin_data['days_since_prior_order'] = (origin_data['days_since_prior_order'] - m_days_po) / sd_days_po
        origin_data['order_dow'] -= 1
        
        # 拼接上一次的order_list
        order_items = self.tables['prior'].groupby('order_id')['product_id'].apply(lambda x : np.array(x))
        orders0 = pd.DataFrame({'last_order_id': origin_data['order_id'],
                                'user_id': origin_data['user_id'],
                                'order_number': origin_data['order_number'] + 1})
        origin_data = origin_data.merge(orders0,\
             left_on=['order_number', 'user_id'], right_on=['order_number', 'user_id'], how='left')
        input_x = origin_data.dropna().set_index('last_order_id').\
            join(order_items).rename(columns={'product_id': 'last_order_list'})
        input_data = input_x.set_index('order_id').join(order_items).rename(columns={'product_id': 'order_list'}).reset_index()
        return input_data

    def get_train_input(self):
        '''
        每条记录所需字段：
        [user_id,] last_order_list, reorder_list, order_dow, order_hour_of_day, days_since_prior_order
        '''
        m_days_po, sd_days_po = self.consts['m_days_po'], self.consts['sd_days_po']
        origin_data = self.tables['orders']
        origin_data['days_since_prior_order'] = (origin_data['days_since_prior_order'] - m_days_po) / sd_days_po
        origin_data['order_dow'] -= 1
        
        # 拼接上一次的order_list
        order_items = self.tables['prior'].groupby('order_id')['product_id'].apply(lambda x : np.array(x))
        orders0 = pd.DataFrame({'last_order_id': origin_data['order_id'],
                                'user_id': origin_data['user_id'],
                                'order_number': origin_data['order_number'] + 1})
        origin_data = origin_data.merge(orders0,\
             left_on=['order_number', 'user_id'], right_on=['order_number', 'user_id'], how='left')
        input_x = origin_data.dropna().set_index('last_order_id').\
            join(order_items).rename(columns={'product_id': 'last_order_list'})
        input_data = input_x.set_index('order_id').join(order_items).rename(columns={'product_id': 'order_list'})
        return input_data.reset_index()
        

    def get_consts(self):
        if exists('consts.pkl'):
            with open('consts.pkl', 'rb') as f:
                consts = pkl.load(f)
                return consts
        tables = self.tables
        if self.mode != 'prior':
            raise ValueError('Use prior data to generate consts first!')

        orders = tables['orders'].dropna()  # 去掉第一次购买的记录
        m_days_po, sd_days_po = orders['days_since_prior_order'].mean(), orders['days_since_prior_order'].std()

        aisle_list, dept_list = self.get_lists()

        # 计算各个product_id被购买的概率
        product_cnt = tables['prior'][['product_id']]
        product_cnt['freq'] = 1
        prob = product_cnt.groupby('product_id')['freq'].sum().map(lambda x : np.power(x, 0.75))
        psum = prob.sum()
        prob = dict(prob / psum)

        consts = { 'm_days_po': m_days_po, 
                   'sd_days_po':sd_days_po, 
                   'aisle_list':aisle_list,
                   'dept_list':dept_list,
                   'product_prob':prob}
        with open('consts.pkl','wb') as f:
                pkl.dump(consts, f, pkl.HIGHEST_PROTOCOL)

        return consts

    def get_lists(self):
        item_aisle = self.tables['products'][['product_id', 'aisle_id']].drop_duplicates()
        item_dept = self.tables['products'][['product_id', 'department_id']].drop_duplicates()
        aisle_list = item_aisle['aisle_id'].tolist()
        dept_list = item_dept['department_id'].tolist()
        return aisle_list, dept_list

    def transform_y(self, item_count):
        # order_list: [B, T]
        def transform(order_list):
            y = np.zeros(item_count)
            y[order_list] = 1
            return y
        return transform

    def transform_hist(self, hist_maxlen):
        def transform(order_list):
            hist = np.zeros(hist_maxlen)
            hist[:len(order_list)] = order_list
            return hist
        return transform

    def gen_subsamples(self, input_data, args):
        print('negative sampling...')
        # sub_sampler = NegativeSampling(args.item_count, self.consts, args.sample_ratio)
        sub_sampler = NegativeSamplingUniform(args.item_count, args.sample_ratio)
        input_data['sub_samples'] = input_data['order_list'].map(sub_sampler)
        input_data = input_data.explode('sub_samples')
        return input_data

    def process_sharded_data(self, input_data, args):
        # 生成hist_seq, hist_length和y
        input_data['hist_len'] = input_data['last_order_list'].map(lambda x: len(x))
        input_data['hist_seq'] = input_data['last_order_list'].map(self.transform_hist(args.hist_maxlen))
        input_data['y'] = input_data['order_list'].map(self.transform_y(args.item_count))
        return input_data.drop(['order_list', 'last_order_list'], axis=1)

class NegativeSampling:
    def __init__(self, item_count, consts, sample_ratio):
        self.item_count = item_count
        if consts is not None:
            prob_dict = consts['product_prob']
            self.probs = np.vectorize(prob_dict.get)(np.arange(0, item_count)).astype('float32')
            self.probs[np.isnan(self.probs)] = 0
        self.sample_ratio = sample_ratio

    def __call__(self, pos_list):
        sub_samples_list = []
        pos_set = set(pos_list)
        neg_list = np.array([self.sample_neg_once(pos_set) for i in range(self.sample_ratio * len(pos_list))])
        for i in range(len(pos_list)):
            index = np.random.randint(len(neg_list), size=self.sample_ratio)
            sub_samples_list.append(np.append(neg_list[index], pos_list[i]))
        return sub_samples_list

    def sample_neg_once(self, pos):
        neg = np.random.choice(np.arange(0, self.item_count), p=self.probs)
        while neg in pos:
            neg = np.random.choice(np.arange(0, self.item_count), p=self.probs)
        return neg

class NegativeSamplingUniform(NegativeSampling):
    def __init__(self, item_count, sample_ratio):
        super().__init__(item_count, None, sample_ratio)

    def __call__(self, pos_list):
        sub_samples_list = []
        all_neg_list = np.setdiff1d(np.arange(self.item_count), np.array(pos_list))
        neg_list = np.random.choice(all_neg_list, size=self.sample_ratio * len(pos_list))
        for i in range(len(pos_list)):
            sub_samples_list.append(np.append(neg_list[i*self.sample_ratio:(i+1)*self.sample_ratio], pos_list[i]))
        return sub_samples_list
