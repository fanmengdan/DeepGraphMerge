from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def process_edge(Ra):
    for i in range(len(Ra)):
        for j in range(len(Ra[i])):
            for k in range(len(Ra[i][j])):
                Ra[i][j][k] = Ra[i][j][k]
    return Ra

def process_node(O):
    for i in range(len(O)):
        for j in range(len(O[i])):
            for k in range(len(O[i][j])):
                O[i][j][k] = round(O[i][j][k])
    return O

def top_ACC(Ra, Ra_t):
    count = 0
    for i in range(Ra.shape[0]):
        for j in range(Ra.shape[2]):
            Ra_t[i, 1, j] = Ra_t[i, 1, j]
            Ra_t[i, 0, j] = Ra_t[i, 0, j]
            if np.argmax(Ra_t[i, :, j]) == np.argmax(Ra[i, :, j]):
                count += 1
    return float(count / (Ra.shape[0] * Ra.shape[2]))

def node_ACC(O, O_t):
    O = O.reshape(O.shape[0], O.shape[2])
    O_t = O_t.reshape(O_t.shape[0], O_t.shape[2])
    count = 0
    for i in range(len(O)):
        for j in range(len(O[i])):
            if O_t[i][j] > 5:
                O_t[i][j] = 1
            else:
                O_t[i][j] = 0
            if O[i][j] > 5:
                O[i][j] = 1
            else:
                O[i][j] = 0
            if O_t[i][j] == O[i][j]:
                count += 1
    return float(count / (O.shape[0] * O.shape[1]))

def mse(label, real):
    mse = 0.0
    for i in range(label.shape[0]):
        score = mean_squared_error(label[i, 0, :], real[i, 0, :])
        mse += score
    return mse / label.shape[0]

def r2(label, real):
    r2 = 0.0
    for i in range(label.shape[0]):
        score = r2_score(label[i, 0, :], real[i, 0, :])
        r2 += score
    return r2 / label.shape[0]

def pear(label, real):
    p = 0.0
    for i in range(label.shape[0]):
        score = pearsonr(label[i, 0, :], real[i, 0, :])[0]
        p += score
    return p / label.shape[0]

def spear(label, real):
    sp = 0.0
    for i in range(label.shape[0]):
        score = spearmanr(label[i, 0, :], real[i, 0, :])[0]
        sp += score
    return sp / label.shape[0]

def prec(label, real):
    prec = 0.0
    labelz = np.ceil(label)
    realz = np.ceil(real)
    for i in range(label.shape[0]):
        precision = precision_score(labelz[i, 0, :], realz[i, 0, :])
        prec += precision
    return prec/label.shape[0]

def recall(label, real):
    rec = 0.0
    labelz = np.ceil(label)
    realz = np.ceil(real)
    for i in range(label.shape[0]):
        recall = recall_score(labelz[i, 0, :], realz[i, 0, :])
        rec += recall
    return rec/label.shape[0]

def f1(label, real):
    F1 = 0.0
    labelz = np.ceil(label)
    realz = np.ceil(real)
    for i in range(label.shape[0]):
        F1_score = f1_score(labelz[i, 0, :], realz[i, 0, :])
        F1 += F1_score
    return F1/label.shape[0]

def AUC(label, real):
    auc = 0.0
    for i in range(label.shape[0]):
        arr_real = []
        arr_label = []
        auc = 0.0
        count = 0

        for j in range(real.shape[2]):
            c = label[i,0,j]
            d = label[i,1,j]
            arr = np.array([c,d])
            label_index = np.argmax(arr)
            real_index = np.argmin(arr)
            arr_label.append(label_index)

            a = real[i,0,j]
            b = real[i,1,j]
            score_arr = [a,b]
            real_score = score_arr[real_index]
            arr_real.append(real_score)

        try:
            Auc_score = roc_auc_score(arr_label, arr_real)
        except:
            print('ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.')
        else:
            auc += Auc_score
            count += 1

    return auc/count