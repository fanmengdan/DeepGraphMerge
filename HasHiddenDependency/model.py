# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v1 as tf
import numpy as np
import time
import os
from utils import read_data
from EvaluationFuncs import *


class graph2graph(object):
    def __init__(self, sess, Ds, Ne, Nc, Ner, Ncr, Dr, De_e, De_er, Mini_batch, checkpoint_dir, epoch, Ds_inter,
                 Dr_inter, Step, Repo):
        self.sess = sess
        self.Ds = Ds
        self.Ne = Ne
        self.Nc = Nc
        self.Ner = Ner
        self.Ncr = Ncr
        self.Dr = Dr
        self.Ds_inter = Ds_inter
        self.Dr_inter = Dr_inter
        self.De_e = De_e
        self.De_er = De_er
        self.mini_batch_num = Mini_batch
        self.epoch = epoch
        self.checkpoint_dir = checkpoint_dir
        self.Step = Step
        self.Repo = Repo
        self.build_model()

    def variable_summaries(self, var, idx):
        with tf.name_scope('summaries_' + str(idx)):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def build_model(self):
        self.E_node_train = tf.placeholder(tf.float32, [self.mini_batch_num, self.Ds, self.Ne],
                                           name="E_node_train")
        self.Es = tf.placeholder(tf.float32, [self.mini_batch_num, self.Ne, self.Ner],
                                 name="Es_data")
        self.Et = tf.placeholder(tf.float32, [self.mini_batch_num, self.Ne, self.Ner],
                                 name="Et_data")
        self.E_edge_train = tf.placeholder(tf.float32, [self.mini_batch_num, self.Dr, self.Ner],
                                           name="E_edge_train")
        self.Cs = tf.placeholder(tf.float32, [self.mini_batch_num, self.Nc, self.Ncr],
                                 name="Cs_label")
        self.Ct = tf.placeholder(tf.float32, [self.mini_batch_num, self.Nc, self.Ncr],
                                 name="Ct_label")
        self.C_edge_train = tf.placeholder(tf.float32, [self.mini_batch_num, self.Dr, self.Ncr],
                                           name="C_edge_train")
        self.Esc = tf.placeholder(tf.float32, [self.mini_batch_num, self.Nc, self.Ner], name="Esc_data")
        self.Etc = tf.placeholder(tf.float32, [self.mini_batch_num, self.Nc, self.Ner], name="Etc_data")

        self.B_1 = self.marshalling_B1(self.E_node_train, self.Es, self.Et, self.E_edge_train)
        self.e_E_node_train = self.mlp_entity_B1(self.B_1)
        self.a_E_node_train = self.agg_entity_B1(self.e_E_node_train, self.Es, self.E_node_train)
        self.E_node_train2 = self.mlp2_entity_B1(self.a_E_node_train)
        self.e_E_edge = self.mlp_entityedge_B1(self.B_1)
        self.a_E_edge = self.agg_entityedge_B1(self.e_E_edge, self.E_edge_train)
        self.E_edge_train2, _ = self.mlp2_entityedge_B1(self.a_E_edge)
        self.B_2 = self.marshalling_B1(self.E_node_train2, self.Es, self.Et, self.E_edge_train2)
        self.B_3 = self.marshalling_B2(self.B_2, self.Esc, self.Etc, self.Cs, self.Ct, self.C_edge_train)
        self.C_edge_output = self.mlp_hunk_B2(self.B_3)
        self.a_C_edge_output = self.agg_entityedge_B1(self.C_edge_output, self.C_edge_train)
        self.C_edge_output2, self.C_edge_output2_logits = self.mlp_hunkedge_B2(self.a_C_edge_output)

        self.loss_Hedge_mse = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.C_edge_output2_logits,
            labels=self.C_edge_train,
            dim=1))
        self.loss_map, self.theta = self.map_loss(2)
        self.loss_E_HR = 0.001 * tf.nn.l2_loss(self.C_edge_output)
        params_list = tf.global_variables()

        for i in range(len(params_list)):
            self.variable_summaries(params_list[i], i)
        self.loss_para = 0
        for i in params_list:
            self.loss_para += 0.001 * tf.nn.l2_loss(i)

        tf.summary.scalar('hunk_edge_mse', self.loss_Hedge_mse)
        tf.summary.scalar('map_mse', self.loss_map)

        t_vars = tf.trainable_variables()
        self.vars = [var for var in t_vars]

        self.saver = tf.train.Saver()

    def marshalling_B1(self, E_node_train, Es, Et, E_edge_train):
        return tf.concat([tf.matmul(E_node_train, Es), tf.matmul(E_node_train, Et), E_edge_train], 1)

    def marshalling_B2(self, B_1, Esc, Etc, Cs, Ct, C_edge_train):
        B_t = tf.transpose(B_1, [0, 2, 1])
        Cs_neighboEt_e = tf.matmul(Esc, B_t)
        Ct_neighboEt_e = tf.matmul(Etc, B_t)
        neighbors = Cs_neighboEt_e + Ct_neighboEt_e
        Cs_t = tf.transpose(Cs, [0, 2, 1])
        Ct_t = tf.transpose(Ct, [0, 2, 1])
        Cs_neighboEt_h = tf.matmul(Cs_t, neighbors)
        Ct_neighboEt_h = tf.matmul(Ct_t, neighbors)
        Cs_neighboEt_ht = tf.transpose(Cs_neighboEt_h, [0, 2, 1])
        Ct_neighboEt_ht = tf.transpose(Ct_neighboEt_h, [0, 2, 1])
        B_2 = tf.concat([Cs_neighboEt_ht, Ct_neighboEt_ht, C_edge_train], 1)
        return B_2

    def mlp_entity_B1(self, B):
        with tf.variable_scope("phi_E_O1") as scope:
            h_size = 20
            B_trans = tf.transpose(B, [0, 2, 1])
            B_trans = tf.reshape(B_trans, [self.mini_batch_num * self.Ner, (2 * self.Ds + self.Dr)])
            w1 = tf.Variable(tf.truncated_normal([(2 * self.Ds + self.Dr), h_size], stddev=0.1), name="r1_w1o",
                             dtype=tf.float32)
            b1 = tf.Variable(tf.zeros([h_size]), name="r1_b1o", dtype=tf.float32)
            h1 = tf.nn.relu(tf.matmul(B_trans, w1) + b1)
            w5 = tf.Variable(tf.truncated_normal([h_size, self.De_e], stddev=0.1), name="r1_w5o", dtype=tf.float32)
            b5 = tf.Variable(tf.zeros([self.De_e]), name="r1_b5o", dtype=tf.float32)
            h5 = tf.matmul(h1, w5) + b5
            h5_trans = tf.reshape(h5, [self.mini_batch_num, self.Ner, self.De_e])
            h5_trans = tf.transpose(h5_trans, [0, 2, 1])
            return (h5_trans)

    def agg_entity_B1(self, E, Es, O):
        E_bar = tf.matmul(E, tf.transpose(Es, [0, 2, 1])) + tf.matmul(E, tf.transpose(self.Et, [0, 2, 1]))
        return (tf.concat([O, E_bar], 1))

    def mlp2_entity_B1(self, C):
        with tf.variable_scope("phi_U_O1") as scope:
            h_size = 20
            C_trans = tf.transpose(C, [0, 2, 1])
            C_trans = tf.reshape(C_trans, [self.mini_batch_num * self.Ne, (self.Ds + self.De_e)])
            w1 = tf.Variable(tf.truncated_normal([(self.Ds + self.De_e), h_size], stddev=0.1), name="o1_w1o",
                             dtype=tf.float32)
            b1 = tf.Variable(tf.zeros([h_size]), name="o1_b1o", dtype=tf.float32)
            h1 = tf.nn.relu(tf.matmul(C_trans, w1) + b1)
            w2 = tf.Variable(tf.truncated_normal([h_size, self.Ds], stddev=0.1), name="o1_w2o", dtype=tf.float32)
            b2 = tf.Variable(tf.zeros([self.Ds_inter]), name="o1_b2o", dtype=tf.float32)
            h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
            h2_trans = tf.reshape(h2, [self.mini_batch_num, self.Ne, self.Ds_inter])
            h2_trans = tf.transpose(h2_trans, [0, 2, 1])
            return (h2_trans)

    def mlp_entityedge_B1(self, B):
        with tf.variable_scope("phi_E_R1") as scope:
            h_size = 20
            B_trans = tf.transpose(B, [0, 2, 1])
            B_trans = tf.reshape(B_trans, [self.mini_batch_num * self.Ner, (2 * self.Ds + self.Dr)])
            w1_1 = tf.Variable(tf.truncated_normal([(self.Ds), h_size], stddev=0.1), name="r1_w1r1", dtype=tf.float32)
            w1_2 = tf.Variable(tf.truncated_normal([(self.Dr), h_size], stddev=0.1), name="r1_w1r2", dtype=tf.float32)
            w1 = tf.concat([w1_1, w1_1, w1_2], 0)
            b1 = tf.Variable(tf.zeros([h_size]), name="r1_b1r", dtype=tf.float32)
            h1 = tf.nn.relu(tf.matmul(B_trans, w1) + b1)
            w2 = tf.Variable(tf.truncated_normal([h_size, self.De_er], stddev=0.1), name="r1_w2r", dtype=tf.float32)
            b2 = tf.Variable(tf.zeros([self.De_er]), name="r1_b2r", dtype=tf.float32)
            h2 = tf.matmul(h1, w2) + b2
            h2_trans = tf.reshape(h2, [self.mini_batch_num, self.Ner, self.De_er])
            h2_trans = tf.transpose(h2_trans, [0, 2, 1])
            h2_trans_bar1 = tf.matmul(h2_trans, tf.transpose(self.Es, [0, 2, 1]))
            h2_trans_bar2 = tf.matmul(h2_trans, tf.transpose(self.Et, [0, 2, 1]))
            effects = tf.matmul(h2_trans_bar1, self.Es) + tf.matmul(h2_trans_bar2, self.Et)
            return effects

    def mlp_hunk_B2(self, B2):
        with tf.variable_scope("mlp_hunk_B2") as scope:
            h_size = 20
            B_trans = tf.transpose(B2, [0, 2, 1])
            B_trans = tf.reshape(B_trans, [self.mini_batch_num * self.Ncr, B2.shape[1]])
            w1 = tf.Variable(tf.truncated_normal([B2.shape[1], h_size], stddev=0.1), name="w1",
                             dtype=tf.float32)
            b1 = tf.Variable(tf.zeros([h_size]), name="b1", dtype=tf.float32)
            h1 = tf.nn.relu(tf.matmul(B_trans, w1) + b1)
            w2 = tf.Variable(tf.truncated_normal([h_size, self.De_er], stddev=0.1), name="r1_w2r", dtype=tf.float32)
            b2 = tf.Variable(tf.zeros([self.De_er]), name="b2", dtype=tf.float32)
            h2 = tf.matmul(h1, w2) + b2
            h2_trans = tf.reshape(h2, [self.mini_batch_num, self.Ncr, self.De_er])
            h2_trans = tf.transpose(h2_trans, [0, 2, 1])
            h2_trans_bar1 = tf.matmul(h2_trans, tf.transpose(self.Cs, [0, 2, 1]))
            h2_trans_bar2 = tf.matmul(h2_trans, tf.transpose(self.Ct, [0, 2, 1]))
            effects = tf.matmul(h2_trans_bar1, self.Cs) + tf.matmul(h2_trans_bar2, self.Ct)
            return effects

    def agg_entityedge_B1(self, E, Ra):
        C_R = tf.concat([Ra, E], 1)
        return (C_R)

    def mlp2_entityedge_B1(self, C_R):
        with tf.variable_scope("phi_U_R1") as scope:
            h_size = 20
            C_trans = tf.transpose(C_R, [0, 2, 1])
            C_trans = tf.reshape(C_trans, [self.mini_batch_num * self.Ner, (self.De_er + self.Dr)])
            w1 = tf.Variable(tf.truncated_normal([(self.De_er + self.Dr), h_size], stddev=0.1), name="o1_w1r",
                             dtype=tf.float32)
            b1 = tf.Variable(tf.zeros([h_size]), name="o1_b1r", dtype=tf.float32)
            h1 = tf.nn.relu(tf.matmul(C_trans, w1) + b1)
            w2 = tf.Variable(tf.truncated_normal([h_size, self.Dr_inter], stddev=0.1), name="o1_w2r", dtype=tf.float32)
            b2 = tf.Variable(tf.zeros([self.Dr_inter]), name="o1_b2r", dtype=tf.float32)
            h2_trans = tf.reshape(tf.matmul(h1, w2) + b2, [self.mini_batch_num, self.Ner, self.Dr])
            h2_trans_logits = tf.transpose(h2_trans, [0, 2, 1])
            h2 = tf.nn.softmax(h2_trans_logits, dim=1)
            return h2, h2_trans_logits

    def mlp_hunkedge_B2(self, hunkedge):
        with tf.variable_scope("phi_U_R1") as scope:
            h_size = 20
            C_edge_trans = tf.transpose(hunkedge, [0, 2, 1])
            C_edge_trans2 = tf.reshape(C_edge_trans, [self.mini_batch_num * self.Ncr, hunkedge.shape[1]])
            w1 = tf.Variable(tf.truncated_normal([hunkedge.shape[1], h_size], stddev=0.1), name="C_edge_w1",
                             dtype=tf.float32)
            b1 = tf.Variable(tf.zeros([h_size]), name="C_edge_b1", dtype=tf.float32)
            h1 = tf.nn.relu(tf.matmul(C_edge_trans2, w1) + b1)
            w2 = tf.Variable(tf.truncated_normal([h_size, self.Dr_inter], stddev=0.1), name="o1_w2r",
                             dtype=tf.float32)
            b2 = tf.Variable(tf.zeros([self.Dr_inter]), name="o1_b2r", dtype=tf.float32)
            h2_trans = tf.reshape(tf.matmul(h1, w2) + b2, [self.mini_batch_num, self.Ncr, self.Dr])
            h2_trans_logits = tf.transpose(h2_trans, [0, 2, 1])
            h2 = tf.nn.softmax(h2_trans_logits, dim=1)
            return h2, h2_trans_logits

    def map_loss(self, k):
        with tf.variable_scope("map_conv") as scope:
            theta1 = tf.Variable(tf.truncated_normal([1, k, 1, 1], stddev=0.1), name="map_theta1", dtype=tf.float32)
            theta2 = tf.Variable(tf.truncated_normal([1, k, 1, 1], stddev=0.1), name="map_theta2", dtype=tf.float32)
            loss2 = tf.sqrt(2 * tf.nn.l2_loss(tf.reshape(theta2, [k]))) + tf.sqrt(
                tf.nn.l2_loss(tf.reshape(theta1, [k])) * 2)
            return 0.01 * loss2, theta2

    def train(self, args):
        train_loss = 10 * self.loss_Hedge_mse + 0.1 * self.loss_map + self.loss_para
        optimizer = tf.train.AdamOptimizer(0.0003)
        trainer = optimizer.minimize(train_loss)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        E_node_train, E_node_test, \
            E_edge_train, E_edge_test, \
            C_edge_train, C_edge_test, \
            Es_data, Et_data, \
            Cs_label, Ct_label, \
            Esc_data, Etc_data = read_data(self, self.Step)

        max_epoches = self.epoch
        counter = 1

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./logs/")

        for i in range(max_epoches):
            tr_loss_Hedge = 0
            tr_loss_map = 0
            C_edge_t = []

            for j in range(int(len(E_node_train) / self.mini_batch_num)):
                batch_O = E_node_train[j * self.mini_batch_num:(j + 1) * self.mini_batch_num]
                batcC_edge_output = E_edge_train[j * self.mini_batch_num:(j + 1) * self.mini_batch_num]
                batch_C_edge_train = C_edge_train[j * self.mini_batch_num:(j + 1) * self.mini_batch_num]

                Merge, C_edge_t_batch, tr_loss_part_Hedge, tr_loss_part_map, theta, _ = \
                    self.sess.run(
                        [merged, self.C_edge_output2, self.loss_Hedge_mse, self.loss_map, self.theta, trainer],
                        feed_dict={
                            self.E_node_train: batch_O,
                            self.E_edge_train: batcC_edge_output,
                            self.C_edge_train: batch_C_edge_train,
                            self.Es: Es_data[:self.mini_batch_num],
                            self.Et: Et_data[:self.mini_batch_num],
                            self.Cs: Cs_label[:self.mini_batch_num],
                            self.Ct: Ct_label[:self.mini_batch_num],
                            self.Esc: Esc_data[:self.mini_batch_num],
                            self.Etc: Etc_data[:self.mini_batch_num]
                        }
                    )

                tr_loss_Hedge += tr_loss_part_Hedge
                tr_loss_map += tr_loss_part_map
                C_edge_t.append(C_edge_t_batch)
                writer.add_summary(Merge, j)

            acc_top = top_ACC(C_edge_train, np.array(C_edge_t).reshape(C_edge_train.shape[0], C_edge_train.shape[1],
                                                                       C_edge_train.shape[2]))
            theta = theta.reshape([2])

            resultString = "Epoch " + str(i + 1) + \
                           " acc: " + str(acc_top)[0:6] + \
                           " Hedge loss: " + str(tr_loss_Hedge / (int(len(E_node_train) / self.mini_batch_num)))[0:6] + \
                           " map MSE: " + str(tr_loss_map / (int(len(E_node_train) / self.mini_batch_num)))[0:6] + \
                           " theta: " + str(theta[0]) + ' ' + str(theta[1]) + '\n'

            filepath = r'outputSelf/{}/model_4/{}/result_{}.npy'.format(args.Repo, self.Step, self.Step)
            directory = os.path.dirname(filepath)
            if not os.path.exists(directory):
                os.makedirs(directory)

            with open(filepath, "a", encoding='utf-8') as f:
                f.write(resultString)

            print(resultString)

            counter += 1
            self.save(args.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        model_name = "g2g.model"
        model_dir = "%s" % (self.Repo + '/model_4/' + str(self.Step))
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s" % (self.Repo + '/model_4/' + str(self.Step))
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self, args):
        E_node_train, E_node_test, \
            E_edge_train, E_edge_test, \
            C_edge_train, C_edge_test, \
            Es_data, Et_data, \
            Cs_label, Ct_label, \
            Esc_data, Etc_data = read_data(self, self.Step)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.Repo)
        if self.load(checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        te_loss_Hedge = 0
        te_loss_map = 0
        C_edge_t = []

        start_time = time.time()
        for j in range(int(len(E_node_test) / self.mini_batch_num)):
            batch_O = E_node_test[j * self.mini_batch_num:(j + 1) * self.mini_batch_num]
            batcC_edge_output = E_edge_test[j * self.mini_batch_num:(j + 1) * self.mini_batch_num]
            batch_C_edge_train = C_edge_test[j * self.mini_batch_num:(j + 1) * self.mini_batch_num]

            te_loss_part_Hedge, te_loss_part_map, C_edge_part = \
                self.sess.run([self.loss_Hedge_mse,
                               self.loss_map,
                               self.C_edge_output2
                               ],
                              feed_dict={
                                  self.E_node_train: batch_O,
                                  self.E_edge_train: batcC_edge_output,
                                  self.C_edge_train: batch_C_edge_train,
                                  self.Es: Es_data[:self.mini_batch_num],
                                  self.Et: Et_data[:self.mini_batch_num],
                                  self.Cs: Cs_label[:self.mini_batch_num],
                                  self.Ct: Ct_label[:self.mini_batch_num],
                                  self.Esc: Esc_data[:self.mini_batch_num],
                                  self.Etc: Etc_data[:self.mini_batch_num]
                              }
                              )

            end_time = time.time()

            te_loss_Hedge += te_loss_part_Hedge
            te_loss_map += te_loss_part_map
            C_edge_t.append(C_edge_part)

        C_edge_t1 = np.array(C_edge_t).reshape(len(E_node_test), self.Dr, self.Ncr)

        base_dir = 'outputSelf/' + args.Repo + '/model_4/'
        step_dir = base_dir + str(self.Step) + '/'
        file_C_edge_t = step_dir + 'C_edge_t' + str(self.Ne) + '.npy'
        file_C_edge_y = step_dir + 'C_edge_y' + str(self.Ne) + '.npy'

        os.makedirs(step_dir, exist_ok=True)

        np.save(file_C_edge_t, np.array(C_edge_t1).reshape(len(E_node_test), self.Dr, self.Ncr))
        np.save(file_C_edge_y, C_edge_test.reshape(len(E_node_test), self.Dr, self.Ncr))

        C_edge_t2 = process_edge(C_edge_t1)

        print('topol_acc: ' + str(top_ACC(C_edge_test, C_edge_t2)))
        print('prec: ' + str(prec(C_edge_test, C_edge_t2)))
        print('recall: ' + str(recall(C_edge_test, C_edge_t2)))
        print('F1-score: ' + str(f1(C_edge_test, C_edge_t2)))
        print('AUC-score: ' + str(AUC(C_edge_test, C_edge_t2)))
        print('test time:' + str(end_time - start_time))