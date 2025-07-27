# -*- coding: utf-8 -*-
import numpy as np
from numpy import float32
import time
from tqdm import *
import os
import joblib

def read_data(self, step):

    repo = self.Repo
    dir = r'./Intermediate_products/'+repo+'/Esc_data_' + str(step) + '.pkl'

    if not os.path.exists(dir):

        diagnal_is0_x = np.ones((self.Ne, self.Ne)) - np.eye(self.Ne)
        diagnal_is0_y = np.ones((self.Nc, self.Nc)) - np.eye(self.Nc)

        x = np.load(r'./Adjset/'+repo+'/Cutting_Adjs/CAdjs_' + str(step) + '.npy', allow_pickle=True)
        y = np.load(r'./Adjset/'+repo+'/Cutting_Adjs/CHunkAdjs_' + str(step) + '.npy', allow_pickle=True)
        IndexPathList = open(r'./dataset/'+repo+'/IndexPathList/IndexPathList_' + str(step) + '.pkl', 'rb')
        HunkIDmaps_path = open(r'./dataset/'+repo+'/HunkIDdict/HunkIDmap_' + str(step) + '.pkl', 'rb')
        IndexPaths = joblib.load(IndexPathList)
        HunkIDmaps = joblib.load(HunkIDmaps_path)

        node_x = np.zeros((x.shape[0], x.shape[1]))

        bar1 = trange(x.shape[0])
        for i in bar1:
            time.sleep(0.01)
            bar1.set_description('node_x[%i]' % i)
            node_x[i] = sum(x[i] * np.eye(self.Ne))
        node_x = node_x.reshape(x.shape[0], 1, self.Ne)

        x_data = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
        y_label = np.zeros((y.shape[0], y.shape[1], y.shape[1]))

        bar2 = trange(x.shape[0])
        for i in bar2:
            time.sleep(0.01)
            bar2.set_description('x_data[i,:,:] and y_label[i,:,:]')
            x_data[i, :, :] = x[i, :, :] * diagnal_is0_x
            y_label[i, :, :] = y[i, :, :] * diagnal_is0_y

        Es_data = np.zeros((100, self.Ne, self.Ner), dtype=float32)
        Et_data = np.zeros((100, self.Ne, self.Ner), dtype=float32)
        E_edge = np.zeros((100, self.Dr, self.Ner), dtype=float32)

        Cs_label = np.zeros((100, self.Nc, self.Ncr), dtype=float32)
        Ct_label = np.zeros((100, self.Nc, self.Ncr), dtype=float32)
        C_edge = np.zeros((100, self.Dr, self.Ncr), dtype=float32)

        Esc_data = np.zeros((100, self.Nc, self.Ner), dtype=float32)
        Etc_data = np.zeros((100, self.Nc, self.Ner), dtype=float32)

        cnt = 0
        bar3 = trange(self.Ne)
        for i in bar3:
            time.sleep(0.01)
            bar3.set_description('i in entity')
            for j in range(self.Ne):
                if (i != j):
                    Es_data[:, i, cnt] = 1.0
                    Et_data[:, j, cnt] = 1.0
                    for k in range(x_data.shape[0]):
                        E_edge[k, int(x_data[k, i, j]), cnt] = 1
                    cnt += 1

            cnt1 = 0
            bar6 = trange(self.Nc)
            for i in bar6:
                time.sleep(0.01)
                bar6.set_description('i in hunk')
                for j in range(self.Nc):
                    if (i != j):
                        Cs_label[:, i, cnt1] = 1.0
                        Ct_label[:, j, cnt1] = 1.0
                        for k in range(y_label.shape[0]):
                            C_edge[k, int(y_label[k, i, j]), cnt1] = 1
                        cnt1 += 1

        bar8 = trange(x.shape[0])
        for index1 in bar8:
           time.sleep(0.01)
           bar8.set_description('IndexPath and HunkIDmap')
           IndexPath = IndexPaths[index1]
           HunkIDmap = HunkIDmaps[index1]

           index_txt = open(IndexPath)
           indexLines = index_txt.readlines()[:self.Ne]
           cnt2 = 0
           for i in range(len(indexLines)):
               for j in range(len(indexLines)):
                   if (i != j):
                       Cs_key = indexLines[i].strip()
                       Ct_key = indexLines[j].strip()
                       if (Cs_key != 'null'):
                           Cs_num = HunkIDmap[Cs_key]
                           if (Cs_num < self.Nc):
                               Esc_data[index1, Cs_num, cnt2] = 1.0
                       if (Ct_key != 'null'):
                           Ct_num = HunkIDmap[Ct_key]
                           if (Ct_num < self.Nc):
                               Etc_data[index1, Ct_num, cnt2] = 1.0
                       cnt2 += 1

        E_node_train = node_x[0:int(x.shape[0] / 2)]
        E_node_test = node_x[int(x.shape[0] / 2):x.shape[0]]

        E_edge_train = E_edge[0:int(x.shape[0] / 2)]
        E_edge_test = E_edge[int(x.shape[0] / 2):x.shape[0]]

        C_edge_train = C_edge[0:int(x.shape[0] / 2)]
        C_edge_test = C_edge[int(x.shape[0] / 2):x.shape[0]]

        with open(r'./Intermediate_products/'+repo+'/E_node_train_' + str(step) + '.pkl', 'wb') as f21:
            joblib.dump(E_node_train, f21)

        with open(r'./Intermediate_products/'+repo+'/E_node_test_' + str(step) + '.pkl', 'wb') as f22:
            joblib.dump(E_node_test, f22)

        with open(r'./Intermediate_products/'+repo+'/E_edge_train_' + str(step) + '.pkl', 'wb') as f23:
            joblib.dump(E_edge_train, f23)

        with open(r'./Intermediate_products/'+repo+'/E_edge_test_' + str(step) + '.pkl', 'wb') as f24:
            joblib.dump(E_edge_test, f24)

        with open(r'./Intermediate_products/'+repo+'/C_edge_train_' + str(step) + '.pkl', 'wb') as f25:
            joblib.dump(C_edge_train, f25)

        with open(r'./Intermediate_products/'+repo+'/C_edge_test_' + str(step) + '.pkl', 'wb') as f26:
            joblib.dump(C_edge_test, f26)

        with open(r'./Intermediate_products/'+repo+'/Es_data_' + str(step) + '.pkl', 'wb') as f27:
            joblib.dump(Es_data, f27)

        with open(r'./Intermediate_products/'+repo+'/Et_data_' + str(step) + '.pkl', 'wb') as f28:
            joblib.dump(Et_data, f28)

        with open(r'./Intermediate_products/'+repo+'/Cs_label_' + str(step) + '.pkl', 'wb') as f29:
            joblib.dump(Cs_label, f29)

        with open(r'./Intermediate_products/'+repo+'/Ct_label_' + str(step) + '.pkl', 'wb') as f30:
            joblib.dump(Ct_label, f30)

        with open(r'./Intermediate_products/'+repo+'/Esc_data_' + str(step) + '.pkl', 'wb') as f31:
           joblib.dump(Esc_data, f31)

        with open(r'./Intermediate_products/'+repo+'/Etc_data_' + str(step) + '.pkl', 'wb') as f32:
           joblib.dump(Etc_data, f32)

    else:

        E_node_train = joblib.load(open(r'./Intermediate_products/'+repo+'/E_node_train_' + str(step) + '.pkl', 'rb'))
        E_node_test = joblib.load(open(r'./Intermediate_products/'+repo+'/E_node_test_' + str(step) + '.pkl', 'rb'))

        E_edge_train = joblib.load(open(r'./Intermediate_products/'+repo+'/E_edge_train_' + str(step) + '.pkl', 'rb'))
        E_edge_test = joblib.load(open(r'./Intermediate_products/'+repo+'/E_edge_test_' + str(step) + '.pkl', 'rb'))

        C_edge_train = joblib.load(open(r'./Intermediate_products/'+repo+'/C_edge_train_' + str(step) + '.pkl', 'rb'))
        C_edge_test = joblib.load(open(r'./Intermediate_products/'+repo+'/C_edge_test_' + str(step) + '.pkl', 'rb'))

        Es_data = joblib.load(open(r'./Intermediate_products/'+repo+'/Es_data_' + str(step) + '.pkl', 'rb'))
        Et_data = joblib.load(open(r'./Intermediate_products/'+repo+'/Et_data_' + str(step) + '.pkl', 'rb'))

        Cs_label = joblib.load(open(r'./Intermediate_products/'+repo+'/Cs_label_' + str(step) + '.pkl', 'rb'))
        Ct_label = joblib.load(open(r'./Intermediate_products/'+repo+'/Ct_label_' + str(step) + '.pkl', 'rb'))

        Esc_data = joblib.load(open(r'./Intermediate_products/'+repo+'/Esc_data_' + str(step) + '.pkl', 'rb'))
        Etc_data = joblib.load(open(r'./Intermediate_products/'+repo+'/Etc_data_' + str(step) + '.pkl', 'rb'))

    return E_node_train,E_node_test,\
           E_edge_train,E_edge_test,\
           C_edge_train,C_edge_test,\
           Es_data,Et_data,\
           Cs_label,Ct_label,\
           Esc_data,Etc_data