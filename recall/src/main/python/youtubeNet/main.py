#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
from model import YoutubeNet
from data_loader.loader import DataLoader

class MyArgs:
    def __init__(self):
        cmd_args = self.parse_cmd_args()
        print(cmd_args)
        self.is_training = not cmd_args.eval
        self.embedding_size = cmd_args.emb_size
        self.batch_size = cmd_args.batch_size
        self.item_count = cmd_args.item_count + 1  # 空出第一个token
        self.aisle_count = cmd_args.aisle_count + 1
        self.dept_count = cmd_args.dept_count + 1
        self.epoch = cmd_args.epoch
        self.init_lr = cmd_args.lr
        self.resume = cmd_args.resume
        self.topn = cmd_args.topn
        self.dense_size = 1
        self.shard_count = 100
        self.hist_maxlen = 145
        self.subsample_size = 20
        self.lr_decay = 0.5
        self.aisle_list = None
        self.dept_list = None
        self.clip_norm = False

    def parse_cmd_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-e', '--epoch', type=int, default=2)
        parser.add_argument('-es', '--emb_size', type=int, default=64)
        parser.add_argument('-it', '--item_count', type=int, default=49688)
        parser.add_argument('-ai', '--aisle_count', type=int, default=134)
        parser.add_argument('-dp', '--dept_count', type=int, default=21)
        parser.add_argument('-bs', '--batch_size', type=int, default=128)
        parser.add_argument('-lr', '--lr', type=float, default=0.5)
        parser.add_argument('-tn', '--topn', type=int, default=10)
        parser.add_argument('-d', '--debug', action='store_true')
        parser.add_argument('-r', '--resume', action='store_true')
        parser.add_argument('--eval', action='store_true')
        args = parser.parse_args()
        return args

def adjust_lr(lr, global_step, args):
    return lr * np.power(args.lr_decay, global_step // 160000)

def main():
    #make args
    args = MyArgs()

    #read data
    dataloader = DataLoader(args)
    dataloader.preload(args.is_training)
    args.aisle_list, args.dept_list = dataloader.consts['aisle_list'], dataloader.consts['dept_list']

    #else para
    checkpoint_dir = 'save_path/ckpt'

    with tf.Session() as sess:
        #build model
        model = YoutubeNet(args)
        #init variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        sys.stdout.flush()
        if args.is_training:
            if args.resume:
                model.restore(sess, checkpoint_dir)
            lr = adjust_lr(args.init_lr, model.global_step.eval(), args)
            start_time = time.time()
            for _ in range(args.epoch):

                loss_sum = 0.0
                for i, batched_data in enumerate(dataloader.gen_batch()):

                    loss = model.train(sess, batched_data, lr)
                    loss_sum += loss

                    if model.global_step.eval() % 1000 == 0:
                        model.save(sess, checkpoint_dir)
                        info_list = ['Epoch : {}'.format(model.global_epoch_step.eval()),
                                     'Global step : {}'.format(model.global_step.eval()),
                                     'Train loss : {:.5f}'.format(loss_sum/1000),
                                     'Learning rate : {:.4f}'.format(lr)]
                        print('\t'.join(info_list))

                        sys.stdout.flush()
                        loss_sum = 0.0

                        lr = adjust_lr(args.init_lr, model.global_step.eval(), args)

                print('Epoch %d DONE\tCost time: %.2f' %
                          (model.global_epoch_step.eval(), time.time() - start_time))
                model.save(sess, f'save_path/ckpt_epoch_{model.global_epoch_step.eval()}')
                sys.stdout.flush()
                model.global_epoch_step_op.eval()

        else:
            print('test')
            model.restore(sess, checkpoint_dir)

            with open("pred_skn.txt", "w") as out_file_skn:
                P = 0
                TP = 0
                for i, batched_data in enumerate(dataloader.gen_test_batch()):
                    output = model.test(sess, batched_data)
                    pred_index = np.argsort(-output, axis=1)[:, :args.topn]
                    y = batched_data['y'].astype('int32')
                    order_ids = batched_data['order_id']

                    P += y.sum()
                    for j in range(y.shape[0]):
                        out_file_skn.write(f'{order_ids[j]}:')
                        for k in range(10):
                            pred_order_id = pred_index[j][k]
                            out_file_skn.write("{}_{};".format(pred_index[j][k], output[j, pred_order_id]))
                        TP += y[j, pred_index[j]].sum()
                        out_file_skn.write("\n")
                    if i % 50 == 0:
                        print(f'predicting batch {i}')

                print(f'TP:{TP}  P:{P}  recall:{TP/P}')

if __name__ == '__main__':
    main()
