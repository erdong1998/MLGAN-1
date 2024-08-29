import tensorflow as tf
import matplotlib.pyplot as plt
import random
import argparse
import sys
import os
import logging
from time import strftime
from time import localtime

from utils.evaluation.TagMetrics import *


class PITF:
    def __init__(self, sess, data_train,  data_test, num_users, num_items, num_tags, embedding_size, reg_rate,
                 learning_rate, \
                 num_neg_tags, batch_size, epochs, verbose, is_save, pretrain_path):
        if batch_size > len(data_train):
            raise ValueError('Batch size cannot be larger than the size of data.')
        self.sess = sess
        self.data_train = data_train
        self.data_test = data_test
        self.num_users = num_users
        self.num_items = num_items
        self.num_tags = num_tags
        self.embedding_size = embedding_size
        self.reg_rate = reg_rate
        self.learning_rate = learning_rate
        self.num_neg_tags = num_neg_tags
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose  # 几次打印一次
        self.is_save = is_save
        self.pretrain_path = pretrain_path
        self.present_batch = 0
        self.num_mem = 25
        self.args = 0.3
        self.hidden_units = 64
        self.output_units = 64
        self.z_dim = 64
        # return dictionary, the key is (u,i), the value the tag set
        self.train_ui_dict = self.get_ui_dict(self.data_train)
        #print(self.train_ui_dict)
        self.test_ui_dict = self.get_ui_dict(self.data_test)

    def get_ui_dict(self, data):
        res = {}
        for i in data:
            # print(i)
            if (i[0], i[1]) not in res.keys():
                # the value of dictionary is a set, which is used to load the tag ids
                res[(i[0], i[1])] = set()
            res[(i[0], i[1])].add(i[2])
        # print(f"res : {res}\n")
        return res

    def get_neg_tags(self):
        all_tags = set(np.arange(self.num_tags))
        neg_tags_dict = {}
        for key in self.train_ui_dict.keys():
            neg_tags_dict[key] = list(all_tags - set(self.train_ui_dict[key]))
        # print(f"neg_tags_dict : {neg_tags_dict}\n")
        return neg_tags_dict

    def composition_layer_U_T(self, user_emb, pos_tag_user_emb, dist='L2', reuse=None):
        energy_U_T = pos_tag_user_emb - (user_emb)
        if ('L2' in dist):
            final_layer_U_T = -tf.sqrt(tf.reduce_sum(tf.square(energy_U_T), 1) + 1E-3)
            print(f"final_layer_U_T : {final_layer_U_T}\n")
        elif ('L1' in dist):
            final_layer_U_T = -tf.reduce_sum(tf.abs(energy_U_T), 1)
        else:
            raise Exception('Please specify distance metric')
        final_layer_U_T = tf.reshape(final_layer_U_T, [-1, 1])
        print(f"final_layer_U_T : {final_layer_U_T}")
        return final_layer_U_T


    def composition_layer_I_T(self, item_emb, pos_tag_item_emb, dist='L2', reuse=None):
        energy_I_T = pos_tag_item_emb - (item_emb)
        if ('L2' in dist):
            final_layer_I_T = -tf.sqrt(tf.reduce_sum(tf.square(energy_I_T), 1) + 1E-3)
        elif ('L1' in dist):
            final_layer_I_T = -tf.reduce_sum(tf.abs(energy_I_T), 1)
        else:
            raise Exception('Please specify distance metric')
        final_layer_I_T = tf.reshape(final_layer_I_T, [-1, 1])
        return final_layer_I_T

    def composition_layer_nU_T(self, user_emb, neg_tag_user_emb, dist='L2', reuse=None):
        energy_nU_T = neg_tag_user_emb - (user_emb)
        if ('L2' in dist):
            final_layer_nU_T = -tf.sqrt(tf.reduce_sum(tf.square(energy_nU_T), 1) + 1E-3)
        elif ('L1' in dist):
            final_layer_nU_T = -tf.reduce_sum(tf.abs(energy_nU_T), 1)
        else:
            raise Exception('Please specify distance metric')
        final_layer_nU_T = tf.reshape(final_layer_nU_T, [-1, 1])
        return final_layer_nU_T

    def composition_layer_nI_T(self, item_emb, neg_tag_item_emb, dist='L2', reuse=None):
        energy_nI_T = neg_tag_item_emb - (item_emb)
        if ('L2' in dist):
            final_layer_nI_T = -tf.sqrt(tf.reduce_sum(tf.square(energy_nI_T), 1) + 1E-3)
        elif ('L1' in dist):
            final_layer_nI_T = -tf.reduce_sum(tf.abs(energy_nI_T), 1)
        else:
            raise Exception('Please specify distance metric')
        final_layer_nI_T = tf.reshape(final_layer_nI_T, [-1, 1])
        return final_layer_nI_T

    def generator(self, z, condition_vec):
        print(f'z-s:{z}')
        print(f"con_vec:{condition_vec}")
        inputs = tf.concat([z, condition_vec], axis=1)
        print(f"inpur:{inputs}")
        hidden = tf.keras.layers.Dense(self.hidden_units, activation=tf.nn.relu)(inputs)
        print(f"hidden:{hidden}")
        hidden1 = tf.keras.layers.Dense(self.hidden_units, activation=tf.nn.relu)(hidden)
        print(f"hidden1:{hidden1}")
        output = tf.keras.layers.Dense(self.output_units, activation=tf.nn.sigmoid)(hidden1)
        print(f"out2:{output}")
        return output
        # 当前的问题，判别器需要传入一个张量

    def discriminator(self, tags_latent_vec, condition_vec):
        inputs = tf.concat([tags_latent_vec, condition_vec], axis=1)
        print(f"dinpur:{inputs}")
        hidden = tf.keras.layers.Dense(self.hidden_units, activation=tf.nn.relu)(inputs)
        print(f"Dhidden:{hidden}")
        hidden1 = tf.keras.layers.Dense(self.hidden_units, activation=tf.nn.relu)(hidden)
        print(f"Dhidden1:{hidden1}")
        output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(hidden1)
        print(f'Dout:{output}')
        return output
    def build_graph(self,norm_clip_value=1):
        # define placehoders
        self.user_id = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None], name="user_id")
        self.item_id = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None], name="item_id")
        self.pos_tag_id = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None], name="pos_tag_id")
        self.neg_tag_id = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None], name="neg_tag_id")

        self.z = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.z_dim], name="z")
        self.tag_id = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None], name="tag_id")
        # define model parameter
        self.U = tf.Variable(tf.compat.v1.truncated_normal([self.num_users, self.embedding_size], mean=0, stddev=0.01),       #tf.truncated_normal(shape, mean, stddev)释义：截断的产生正态分布的随机数，即随机数与均值的差值若大于两倍的标准差，则重新生成。shape，生成张量的维度mean，均值stddev，标准差
                             dtype=tf.float32, name="U_embed")
        self.I = tf.Variable(tf.compat.v1.truncated_normal([self.num_items, self.embedding_size], mean=0, stddev=0.01),
                             dtype=tf.float32, name="I_embed")
        self.Tu = tf.Variable(tf.compat.v1.truncated_normal([self.num_tags, self.embedding_size], mean=0, stddev=0.01),
                              dtype=tf.float32, name="Tu_embed")
        self.Ti = tf.Variable(tf.compat.v1.truncated_normal([self.num_tags, self.embedding_size], mean=0, stddev=0.01),
                              dtype=tf.float32, name="Ti_embed")


        # lookup latent vector for user ,item and tag
        self.user_emb = tf.nn.embedding_lookup(self.U, self.user_id)
        self.item_emb = tf.nn.embedding_lookup(self.I, self.item_id)
        self.pos_tag_user_emb = tf.nn.embedding_lookup(self.Tu, self.pos_tag_id)
        self.pos_tag_item_emb = tf.nn.embedding_lookup(self.Ti, self.pos_tag_id)
        self.neg_tag_user_emb = tf.nn.embedding_lookup(self.Tu, self.neg_tag_id)
        self.neg_tag_item_emb = tf.nn.embedding_lookup(self.Ti, self.neg_tag_id)

        condition_vec = tf.concat([self.user_emb, self.item_emb], axis=1)

        with tf.compat.v1.variable_scope("generator", reuse=tf.compat.v1.AUTO_REUSE):
            x_fake = self.generator(self.z, condition_vec)
            print(f'fake:{x_fake}')
        with tf.compat.v1.variable_scope("discriminator", reuse=tf.compat.v1.AUTO_REUSE):
            d_real = self.discriminator(self.pos_tag_user_emb, condition_vec)
            d_fake = self.discriminator(x_fake, condition_vec)
        U_T_final_layer = self.composition_layer_U_T(self.user_emb, self.pos_tag_user_emb)
        I_T_final_layer = self.composition_layer_I_T(self.item_emb, self.pos_tag_item_emb)
        nU_T_final_layer = self.composition_layer_nU_T(self.user_emb, self.neg_tag_user_emb)
        nI_T_final_layer = self.composition_layer_nI_T(self.item_emb, self.neg_tag_item_emb)

        nu_t_fake=self.composition_layer_nU_T(self.user_emb,x_fake)
        ni_t_fake=self.composition_layer_I_T(self.item_emb,x_fake)
        U1 = self.composition_layer_U_T(x_fake, self.pos_tag_user_emb)
        I1 = self.composition_layer_U_T(x_fake, self.pos_tag_item_emb)
        U11 = self.composition_layer_U_T(x_fake, self.neg_tag_user_emb)
        I11 = self.composition_layer_U_T(x_fake, self.neg_tag_item_emb)

        d_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        g_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        self.d_realloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real)))
        self.d_fakeloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))
        self.d_loss = self.d_fakeloss + self.d_realloss
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))

        self.d_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.d_loss,
                                                                                                       var_list=d_vars)
        self.g_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.g_loss,
                                                                                                       var_list=g_vars)

        #self.cost = tf.reduce_sum(tf.nn.relu(margin - U_T_final_layer + nU_T_final_layer-I_T_final_layer+nI_T_final_layer))
        # define the y_uitt'
        self.pred_y = U_T_final_layer + I_T_final_layer
        print(f"self.pred_y : {self.pred_y}")
        #self.pred_ny = nU_T_final_layer + nI_T_final_layer

        self.loss = tf.reduce_sum(tf.maximum(0.2- U_T_final_layer + nU_T_final_layer-I_T_final_layer + nI_T_final_layer,0))
        self.GLOSS=tf.reduce_sum(tf.maximum(0.2-U_T_final_layer+nu_t_fake-I_T_final_layer+ni_t_fake,0))
        self.rec=0.0001*tf.reduce_sum((U1+U11)+(I1+I11))
        self.zloss=self.g_loss+self.d_loss+self.loss+0.01*(self.GLOSS)+self.rec
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.zloss, var_list=[self.U,self.I,self.Tu,self.Ti,self.tag_id])

        return self

    def get_train_batch(self):
        len_data = len(self.data_train)
        if self.present_batch + self.batch_size > len_data - 1:
            res = self.data_train[self.present_batch: len_data] + \
                  self.data_train[0: self.present_batch + self.batch_size - len_data]
        else:
            res = self.data_train[self.present_batch: self.present_batch + self.batch_size]
        self.present_batch += self.batch_size
        self.present_batch %= len_data
        return res

    def get_feed_dict(self, data):
        if len(set([len(i) for i in data])) != 1:
            raise ValueError('Data must all be [u, i, t].')
        user_list = []
        item_list = []
        pos_tag_list = []
        neg_tag_list = []
        z_batch = list(np.random.normal(0, 1, (self.batch_size, self.z_dim)))
        for item in data:
            for index in range(self.num_neg_tags):
                nt = random.sample(range(self.num_tags), 1)[0]
                while nt in self.train_ui_dict[(item[0], item[1])]:
                    nt = random.sample(range(self.num_tags), 1)[0]
                user_list.append(item[0])
                item_list.append(item[1])
                pos_tag_list.append(item[2])
                neg_tag_list.append(nt)
        # return as the feed_dict form
        #print(f"user_list : {user_list}\n ")
        #print(f"self.user_id : {self.user_id}\n ")
        return {self.user_id: user_list,
                self.item_id: item_list,
                self.pos_tag_id: pos_tag_list,
                self.neg_tag_id: neg_tag_list,
                self.z:z_batch}

    def train(self):
        self.saver = tf.compat.v1.train.Saver()

        random.shuffle(self.data_train)
        # with tf.Session() as sess:
        self.sess.run(tf.compat.v1.global_variables_initializer())
        epoch_indexs = []
        loss_history = []
        pre3_history = []
        rec3_history = []
        ndcg3_history = []
        pre5_history = []
        rec5_history = []
        ndcg5_history = []
        pre10_history = []
        rec10_history = []
        ndcg10_history = []
        pre20_history = []
        rec20_history = []
        ndcg20_history = []

        for i in range(1, self.epochs + 1):
            batch_dict = self.get_feed_dict(self.get_train_batch())
            loss, _ = self.sess.run([self.zloss, self.optimizer], feed_dict=batch_dict)

            # Train the discriminator

            train_dict = self.get_feed_dict(self.get_train_batch())
            _, d_loss = self.sess.run([self.d_optimizer, self.d_loss],
                                      feed_dict=train_dict)

            # Train the generator

            train_dict = self.get_feed_dict(self.get_train_batch())
            _, g_loss = self.sess.run([self.g_optimizer, self.g_loss],
                                      feed_dict=train_dict)

            if i % self.verbose == 0:
                sys.stdout.flush()
                epoch_indexs.append(i)
                loss_history.append(loss)
                pre3,rec3,ndcg3,pre5,rec5,ndcg5,pre10,rec10,ndcg10,pre20,rec20,ndcg20 = self.test()
                pre3_history.append(pre3)
                rec3_history.append(rec3)
                ndcg3_history.append(ndcg3)
                pre5_history.append(pre5)
                rec5_history.append(rec5)
                ndcg5_history.append(ndcg5)
                pre10_history.append(pre10)
                rec10_history.append(rec10)
                ndcg10_history.append(ndcg10)
                pre20_history.append(pre20)
                rec20_history.append(rec20)
                ndcg20_history.append(ndcg20)
                print(
                    "Epoch: %04d \t Loss: %.2f \t Pre@3:%.5f \t Rec@3:%.5f \t ndcg@3:%.5f \t Pre@5:%.5f \t  %.5f \t %.5f \t Pre@10:%.5f \t Rec@10: %.5f \t ndcg@10:%.5f \t Pre@20: %.5f \t  Rec@20: %.5f \t ndcg@20:%.5f \t" \
                    % (i, loss, pre3, rec3, ndcg3, pre5, rec5, ndcg5, pre10, rec10, ndcg10, pre20, rec20, ndcg20))
                logging.info(
                    "Epoch: %04d \t Loss: %.2f \t Pre@3:%.5f \t Rec@3:%.5f \t ndcg@3:%.5f \t Pre@5:%.5f \t  %.5f \t %.5f \t Pre@10:%.5f \t Rec@10: %.5f \t ndcg@10:%.5f \t Pre@20: %.5f \t  Rec@20: %.5f \t ndcg@20:%.5f \t" \
                    % (i, loss, pre3, rec3, ndcg3, pre5, rec5, ndcg5, pre10, rec10, ndcg10, pre20, rec20, ndcg20))

        best_pre3 = np.max(pre3_history)
        best_rec3 = np.max(rec3_history)
        best_ndcg3 = np.max(ndcg3_history)
        best_pre5 = np.max(pre5_history)
        best_rec5 = np.max(rec5_history)
        best_ndcg5 = np.max(ndcg5_history)
        best_pre10 = np.max(pre10_history)
        best_rec10 = np.max(rec10_history)
        best_ndcg10 = np.max(ndcg10_history)
        best_pre20 = np.max(pre20_history)
        best_rec20 = np.max(rec20_history)
        best_ndcg20 = np.max(ndcg20_history)
        print("Best Pre@3:%.5f \t Rec@3: %.5f \t ndcg@3: %.5f" % (best_pre3, best_rec3, best_ndcg3))
        print("Best Pre@5:%.5f \t Rec@5: %.5f \t ndcg@5: %.5f" % (best_pre5, best_rec5, best_ndcg5))
        print("Best Pre@10:%.5f \t Rec@10: %.5f \t ndcg@10: %.5f" % (best_pre10, best_rec10, best_ndcg10))
        print("Best Pre@20:%.5f \t Rec@20: %.5f \t ndcg@20: %.5f" % (best_pre20, best_rec20, best_ndcg20))
        if self.is_save:
            print("Save model to file as pretrain.")
            print(self.pretrain_path)
            self.saver.save(self.sess, self.pretrain_path)
        logging.info("Best Pre@5:%.5f \t Rec@5: %.5f" % (best_pre5, best_rec5))
        self.show(loss_history, pre5_history, rec5_history)

    def show(self, loss_history, pre5_history, rec5_history):
        fig1 = plt.figure('LOSS')
        x = range(len(loss_history))
        plt.plot(x, pre5_history, marker='o', label='pre5')
        plt.plot(x, rec5_history, marker='v', label='rec5')
        plt.title('The MovieLens Dataset Learning Curve')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Pre&Rec')
        plt.legend()
        plt.grid()
        plt.show()

    def test(self):
        pre3, rec3, ndcg3, pre5, rec5, ndcg5, pre10, rec10, ndcg10, pre20, rec20, ndcg20 = evaluate(self)
        return pre3, rec3, ndcg3, pre5, rec5, ndcg5, pre10, rec10, ndcg10, pre20, rec20, ndcg20

    def predict(self, user_id, item_id, tag_id):
        result = self.sess.run([self.pred_y],
                               feed_dict={self.user_id: user_id, self.item_id: item_id, self.pos_tag_id: tag_id})
        # print("result;",result)
        # print(result[0])
        return result[0]
        # return self.sess.run([self.pred_y], feed_dict={self.user_id: user_id, self.item_id: item_id,self.pos_tag_id:tag_id})[0]

    def save_tag_ratings(self):
        pass


def load_uit(path):
    num_users = -1
    num_items = -1
    num_tags = -1
    data = []  # 列表
    with open(path) as f:
        for line in f:
            line = [int(i) for i in line.split('\t')[:3]]
            # print(line)
            data.append(line)
            num_users = max(line[0], num_users)
            num_items = max(line[1], num_items)
            num_tags = max(line[2], num_tags)
    num_users, num_items, num_tags = num_users + 1, num_items + 1, num_tags + 1
    return data, num_users, num_items, num_tags


def load_data(path):
    print('Loading train and test data...', end='')
    sys.stdout.flush()  # ?????
    data_train, num_users, num_items, num_tags = load_uit(path + '.train')
    data_test, num_users2, num_items2, num_tags2 = load_uit(path + '.test')
    num_users = max(num_users, num_users2)
    num_items = max(num_items, num_items2)
    num_tags = max(num_tags, num_tags2)
    print('Done.')
    print()
    print('Number of users: %d' % num_users)
    print('Number of items: %d' % num_items)
    print('Number of tags: %d' % num_tags)
    print('Number of train data: %d' % len(data_train))
    print('Number of test data: %d' % len(data_test))

    logging.info('Number of users: %d' % num_users)
    logging.info('Number of items: %d' % num_items)
    logging.info('Number of tags: %d' % num_tags)
    logging.info('Number of train data: %d' % len(data_train))
    logging.info('Number of test data: %d' % len(data_test))
    sys.stdout.flush()
    return data_train, data_test, num_users, num_items, num_tags


def parse_args():
    parser = argparse.ArgumentParser(description='Run Pairwise Interaction Tensor Factorization.')
    parser.add_argument('--dataset_path', nargs='?', default='./data/clean-data/',
                        help='Data path.')
    parser.add_argument('--dataset', nargs='?', default='lastfm-uit-10',
                        help='Name of the dataset.')
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--reg_rate', type=float, default=0.001,
                        help='Regularization coefficient for user and item embeddings.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--num_neg_tags', type=int, default=1,
                        help='number of negtative tags')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of iterations.')
    parser.add_argument('--verbose', type=int, default=10,
                        help='Interval of evaluation.')
    parser.add_argument('--is_save', type=bool, default=True,
                        help='Save the model or not')
    parser.add_argument('--pretrain_path', nargs='?', default='pretrain/',
                        help='Save path.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    log_path = "log/%s_%s/" % ("PITF", strftime('%Y-%m-%d', localtime()))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_path = log_path + "%s_embed_size%.4f_reg%.5f_lr%0.5f_%s" % (
        args.dataset, args.embedding_size, args.reg_rate, args.learning_rate, strftime('%Y_%m_%d_%H', localtime()))
    logging.basicConfig(filename=log_path,
                        level=logging.INFO)
    logging.info(args)
    #
    pretrain_path = args.pretrain_path
    if args.is_save and not os.path.exists(pretrain_path):
        os.makedirs(pretrain_path)

    pretrain_path = pretrain_path + 'PITF_%s_%d' % (args.dataset, args.embedding_size)
    print(pretrain_path)
    # data_train and data_test are lists.
    data_train, data_test, num_users, num_items, num_tags = load_data(args.dataset_path + args.dataset)
    # print("data:",data_train)列表嵌套

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config) as sess:
        model = PITF(sess, data_train, data_test, num_users, num_items, num_tags, args.embedding_size, args.reg_rate, \
                     args.learning_rate, args.num_neg_tags, args.batch_size, args.epochs, args.verbose, args.is_save,
                     pretrain_path)
        model.build_graph()
        model.train()