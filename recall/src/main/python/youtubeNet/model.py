# -*- coding: utf-8 -*-
import tensorflow as tf


class YoutubeNet:
    def __init__(self, args):
        self.is_training = args.is_training
        # self.input_size=args.input_size
        self.embedding_size = args.embedding_size
        self.dense_size = args.dense_size
        self.subsample_size = args.subsample_size + 1
        self.aisle_list = args.aisle_list
        self.dept_list = args.dept_list
        self.item_count = args.item_count
        self.aisle_count = args.aisle_count
        self.dept_count = args.dept_count
        self.clip_norm = args.clip_norm
        self.build_model()

    def build_model(self):
        # placeholder
        self.hist_i = tf.placeholder(
            tf.int32, [None, None])  # history order[B, T]
        self.sl = tf.placeholder(tf.int32, [None, ])  # history len [B]
        # self.last = tf.placeholder(tf.int32, [None, ])  # last order[B]
        # user basic feature[B,dense_size]
        self.dense = tf.placeholder(tf.float32, [None, self.dense_size])
        # soft layer (pos_click,neg_list)[B,sub_size]
        self.sub_sample = tf.placeholder(tf.int32, [None, self.subsample_size])
        self.dow = tf.placeholder(tf.float32, [None, 7])  # dow [B, 7]
        self.hod = tf.placeholder(tf.float32, [None, 24])  # hod [B, 24]
        self.y = tf.placeholder(tf.float32, [None, None])  # label one hot[B]
        self.lr = tf.placeholder(tf.float64, [])

        # emb variable
        # 每个item的embedding其实是item_id，aisle_id，dept_id三者embedding的concat
        item_emb_w = tf.get_variable(
            "item_emb_w", [self.item_count, self.embedding_size])
        item_b = tf.get_variable(
            "item_b", [self.item_count], initializer=tf.constant_initializer(0.0))
        aisle_emb_w = tf.get_variable(
            "aisle_emb_w", [self.aisle_count, self.embedding_size])
        dept_emb_w = tf.get_variable(
            "dept_emb_w", [self.dept_count, self.embedding_size])

        aisle_list = tf.convert_to_tensor(self.aisle_list, dtype=tf.int32)
        dept_list = tf.convert_to_tensor(self.dept_list, dtype=tf.int32)

        # historty seq
        hist_a = tf.gather(aisle_list, self.hist_i)
        hist_d = tf.gather(dept_list, self.hist_i)

        h_emb = tf.concat([tf.nn.embedding_lookup(item_emb_w, self.hist_i),
                           tf.nn.embedding_lookup(aisle_emb_w, hist_a),
                           tf.nn.embedding_lookup(dept_emb_w, hist_d)], axis=2)
        # historty mask
        mask = tf.sequence_mask(self.sl, tf.shape(
            h_emb)[1], dtype=tf.float32)  # [B,T]
        mask = tf.expand_dims(mask, -1)  # [B,T,1]
        mask = tf.tile(mask, [1, 1, tf.shape(h_emb)[2]])  # [B,T,3*e]

        h_emb *= mask  # [B,T,3*e]
        hist = tf.reduce_sum(h_emb, 1)  # [B,3*e]
        hist = tf.div(hist, tf.cast(tf.tile(tf.expand_dims(self.sl, 1), [
                      1, 3*self.embedding_size]), tf.float32))  # [B,3*e]
        # last
        # last_b=tf.gather(aisle_list,self.last)
        # last_m=tf.gather(dept_list,self.last)
        # l_emb=tf.concat([tf.nn.embedding_lookup(item_emb_w,self.last),
        #                 tf.nn.embedding_lookup(aisle_emb_w,last_b),
        #                 tf.nn.embedding_lookup(dept_emb_w,last_m)],axis=1)
        # net input
        ctx_v = tf.concat([self.dow, self.hod, self.dense], axis=1)

        self.input = tf.concat([hist, ctx_v], axis=1)
        # print('',)

        # dd net
        bn = tf.layers.batch_normalization(inputs=self.input, name='b1')
        layer_1 = tf.layers.dense(bn, 1024, activation=tf.nn.relu, name='f1')
        layer_2 = tf.layers.dense(
            layer_1, 512, activation=tf.nn.relu, name='f2')
        layer_3 = tf.layers.dense(
            layer_2, 3*self.embedding_size, activation=tf.nn.relu, name='f3')

        # softmax
        if self.is_training:
            sa_a = tf.gather(aisle_list, self.sub_sample)
            sa_d = tf.gather(dept_list, self.sub_sample)

            sample_w = tf.concat([tf.nn.embedding_lookup(item_emb_w, self.sub_sample),
                                  tf.nn.embedding_lookup(aisle_emb_w, sa_a),
                                  tf.nn.embedding_lookup(dept_emb_w, sa_d)], axis=2)  # [B,sample,3*e]
            # sample_w=tf.nn.embedding_lookup(item_emb_w,self.sub_sample)
            sample_b = tf.nn.embedding_lookup(
                item_b, self.sub_sample)  # [B,sample]
            user_v = tf.expand_dims(layer_3, 1)  # [B,1,3*e]
            sample_w = tf.transpose(sample_w, perm=[0, 2, 1])  # [B,3*e,sample]
            self.logits = tf.squeeze(
                tf.matmul(user_v, sample_w), axis=1)+sample_b

            # Step variable
            self.global_step = tf.Variable(
                0, trainable=False, name='global_step')
            self.global_epoch_step = tf.Variable(
                0, trainable=False, name='global_epoch_step')
            self.global_epoch_step_op = tf.assign(
                self.global_epoch_step, self.global_epoch_step + 1)
            '''
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits,
                    labels=self.y)
            )
            '''
            self.yhat = tf.nn.softmax(self.logits)

            self.loss = -tf.reduce_mean(tf.multiply(self.y, tf.log(self.yhat + 1e-24)))

            trainable_params = tf.trainable_variables()
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            gradients = tf.gradients(self.loss, trainable_params)
            if self.clip_norm:
                clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
            else:
                clip_gradients = gradients
            self.train_op = self.opt.apply_gradients(
                zip(clip_gradients, trainable_params), global_step=self.global_step)

        else:
            all_emb = tf.concat([item_emb_w,
                                 tf.nn.embedding_lookup(
                                     aisle_emb_w, aisle_list),
                                 tf.nn.embedding_lookup(dept_emb_w, dept_list)], axis=1)
            self.logits = tf.matmul(layer_3, all_emb, transpose_b=True)+item_b
            self.output = tf.nn.softmax(self.logits)

    def train(self, sess, data, lr):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            self.sub_sample: data['sub_samples'],
            self.y: data['sampled_y'],
            self.hist_i: data['hist_seq'],
            self.sl: data['hist_len'],
            self.dow: data['dow'],
            self.hod: data['hod'],
            self.dense: data['dense'],
            self.lr: lr
        })
        return loss

    def test(self, sess, data):
        out = sess.run(self.output, feed_dict={
            self.hist_i: data['hist_seq'],
            self.sl: data['hist_len'],
            self.dow: data['dow'],
            self.hod: data['hod'],
            self.dense: data['dense']
        })
        return out

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)


class YoutubeNet_reorder:
    '''
    包含reorder的版本，当reorder缺失时，使用’0‘来表示对应的embedding
    '''

    def __init__(self, args):
        self.is_training = args.is_training
        # self.input_size=args.input_size
        self.embedding_size = args.embedding_size
        # self.dense_size=args.dense_size
        self.aisle_list = args.aisle_list
        self.dept_list = args.dept_list
        self.reorder_list = args.reorder_list
        self.item_count = args.item_count
        self.aisle_count = args.aisle_count
        self.dept_count = args.dept_count
        self.dense_size = args.dense_size
        self.build_model()

    def build_model(self):
        # placeholder
        self.hist_i = tf.placeholder(
            tf.int32, [None, None])  # history order[B, T1]
        self.hist_r = tf.placeholder(
            tf.int32, [None, None])  # history reorder[B, T2]
        self.sl = tf.placeholder(tf.int32, [None, ])  # history len [B]
        # user basic feature[B,dense_size]
        self.basic = tf.placeholder(tf.float32, [None, None])
        # soft layer (pos_clict,neg_list)[B,sub_size]
        self.sub_sample = tf.placeholder(tf.int32, [None, None])
        self.y = tf.placeholder(tf.float32, [None, None])  # label one hot[B]
        self.lr = tf.placeholder(tf.float64, [])

        # emb variable
        # 每个item的embedding其实是item_id，aisle_id，dept_id三者embedding的concat
        item_emb_w = tf.get_variable(
            "item_emb_w", [self.item_count, self.embedding_size])
        item_b = tf.get_variable(
            "item_b", [self.item_count], initializer=tf.constant_initializer(0.0))
        aisle_emb_w = tf.get_variable(
            "aisle_emb_w", [self.aisle_count, self.embedding_size])
        dept_emb_w = tf.get_variable(
            "dept_emb_w", [self.dept_count, self.embedding_size])

        aisle_list = tf.convert_to_tensor(self.aisle_list, dtype=tf.int32)
        dept_list = tf.convert_to_tensor(self.dept_list, dtype=tf.int32)

        # historty seq
        hist_a = tf.gather(aisle_list, self.hist_i)
        hist_d = tf.gather(dept_list, self.hist_i)

        h_emb = tf.concat([tf.nn.embedding_lookup(item_emb_w, self.hist_i),
                           tf.nn.embedding_lookup(aisle_emb_w, hist_a),
                           tf.nn.embedding_lookup(dept_emb_w, hist_d)], axis=2)
        # historty mask
        mask = tf.sequence_mask(self.sl, tf.shape(
            h_emb)[1], dtype=tf.float32)  # [B,T]
        mask = tf.expand_dims(mask, -1)  # [B,T,1]
        mask = tf.tile(mask, [1, 1, tf.shape(h_emb)[2]])  # [B,T,3*e]

        h_emb *= mask  # [B,T,3*e]
        hist = tf.reduce_sum(h_emb, 1)  # [B,3*e]
        hist = tf.div(hist, tf.cast(tf.tile(tf.expand_dims(self.sl, 1), [
                      1, 3*self.embedding_size]), tf.float32))  # [B,3*e]
        # last
        last_b = tf.gather(aisle_list, self.last)
        last_m = tf.gather(dept_list, self.last)
        l_emb = tf.concat([tf.nn.embedding_lookup(item_emb_w, self.last),
                           tf.nn.embedding_lookup(aisle_emb_w, last_b),
                           tf.nn.embedding_lookup(dept_emb_w, last_m)], axis=1)
        # net input
        self.input = tf.concat([hist, l_emb], axis=-1)
        # print('',)

        # dd net
        bn = tf.layers.batch_normalization(inputs=self.input, name='b1')
        layer_1 = tf.layers.dense(bn, 1024, activation=tf.nn.relu, name='f1')
        layer_2 = tf.layers.dense(
            layer_1, 512, activation=tf.nn.relu, name='f2')
        layer_3 = tf.layers.dense(
            layer_2, 3*self.embedding_size, activation=tf.nn.relu, name='f3')

        # softmax
        if self.is_training:
            sa_a = tf.gather(aisle_list, self.sub_sample)
            sa_d = tf.gather(dept_list, self.sub_sample)

            sample_w = tf.concat([tf.nn.embedding_lookup(item_emb_w, self.sub_sample),
                                  tf.nn.embedding_lookup(aisle_emb_w, sa_a),
                                  tf.nn.embedding_lookup(dept_emb_w, sa_d)], axis=2)  # [B,sample,3*e]
            # sample_w=tf.nn.embedding_lookup(item_emb_w,self.sub_sample)
            sample_b = tf.nn.embedding_lookup(
                item_b, self.sub_sample)  # [B,sample]
            user_v = tf.expand_dims(layer_3, 1)  # [B,1,3*e]
            sample_w = tf.transpose(sample_w, perm=[0, 2, 1])  # [B,3*e,sample]
            self.logits = tf.squeeze(
                tf.matmul(user_v, sample_w), axis=1)+sample_b

            # Step variable
            self.global_step = tf.Variable(
                0, trainable=False, name='global_step')
            self.global_epoch_step = tf.Variable(
                0, trainable=False, name='global_epoch_step')
            self.global_epoch_step_op = tf.assign(
                self.global_epoch_step, self.global_epoch_step + 1)
            '''
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits,
                    labels=self.y)
            )
            '''
            self.yhat = tf.nn.softmax(self.logits)

            self.loss = tf.reduce_mean(-self.y * tf.log(self.yhat + 1e-24))

            trainable_params = tf.trainable_variables()
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            gradients = tf.gradients(self.loss, trainable_params)
            if self.clip_norm:
                clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
            else:
                clip_gradients = gradients
            self.train_op = self.opt.apply_gradients(
                zip(clip_gradients, trainable_params), global_step=self.global_step)

        else:
            all_emb = tf.concat([item_emb_w,
                                 tf.nn.embedding_lookup(
                                     aisle_emb_w, aisle_list),
                                 tf.nn.embedding_lookup(dept_emb_w, dept_list)], axis=1)
            self.logits = tf.matmul(layer_3, all_emb, transpose_b=True)+item_b
            self.output = tf.nn.softmax(self.logits)
