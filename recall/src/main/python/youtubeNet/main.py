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
        self.dense_size = 1
        self.shard_count = 100
        self.hist_maxlen = 145
        self.subsample_size = 20
        self.aisle_list = None
        self.dept_list = None
        self.clip_norm = False

    def parse_cmd_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-e', '--epoch', type=int, default=5)
        parser.add_argument('-es', '--emb_size', type=int, default=64)
        parser.add_argument('-it', '--item_count', type=int, default=49688)
        parser.add_argument('-ai', '--aisle_count', type=int, default=134)
        parser.add_argument('-dp', '--dept_count', type=int, default=21)
        parser.add_argument('-bs', '--batch_size', type=int, default=128)
        parser.add_argument('-lr', '--lr', type=float, default=0.1)
        parser.add_argument('-d', '--debug', action='store_true')
        parser.add_argument('--eval', action='store_true')
        args = parser.parse_args()
        return args

def main():
    #make args
    args = MyArgs()

    #read data
    dataloader = DataLoader(args)
    dataloader.load()
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
            lr = args.init_lr
            start_time = time.time()
            for epoch in range(args.epoch):

                loss_sum = 0.0
                for i, batched_data in enumerate(dataloader.gen_batch()):

                    loss = model.train(sess, batched_data, lr)
                    loss_sum += loss

                    if model.global_step.eval() % 1000 == 0:
                        model.save(sess, checkpoint_dir)
                        info_list = ['Epoch : {0}\tBatch num : {1}'.format(epoch, i+1),
                                     'Global step : {}'.format(model.global_step.eval()),
                                     'Train loss : {.4f}'.format(loss_sum/1000),
                                     'Learning rate : {.2f}'.format(lr)]
                        print('\t'.join(info_list))

                        sys.stdout.flush()
                        loss_sum = 0.0

                    if (i+1) % 1300000 == 0:
                        lr *= 0.7

                print('Epoch %d DONE\tCost time: %.2f' %
                          (model.global_epoch_step.eval(), time.time() - start_time))
                sys.stdout.flush()
                model.global_epoch_step_op.eval()

        else:
            print('test')
            # model.restore(sess, checkpoint_dir)
            # '''
            # saver = tf.train.Saver()
            # ckpt = tf.train.get_checkpoint_state('./save_path')
            # if ckpt and ckpt.model_checkpoint_path:
            #     print("Successfully loaded:", ckpt.model_checkpoint_path)
            #     saver.restore(sess, ckpt.model_checkpoint_path)
            # '''
            # out_file_skn = open("pred_skn.txt", "w")
    
            # for _, uij in DataInputTest(test_set, test_batch_size):
            #     output = model.test(sess, uij[0], uij[1], uij[2],uij[3])
            #     pre_index = np.argsort(-output, axis=1)[:, 0:200]
             
            #     for y in range(len(uij[0])):
            #         out_file_skn.write(str(uij[0][y]))
            #         pre_skn = pre_index[y]
            #         print(pre_skn)
            #         for k in pre_skn:
            #             out_file_skn.write("\t%i" % item_key[k])
            #         out_file_skn.write("\n")

if __name__ == '__main__':
    main()
