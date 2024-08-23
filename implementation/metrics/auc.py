import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score


def caculate_auc_nn(label, pred):
    # 计算正样本和负样本的索引，以便索引出之后的概率值
    pos = [i for i in range(len(label)) if label[i] == 1]
    neg = [i for i in range(len(label)) if label[i] == 0]

    # 复杂度 O(m * n)
    auc = 0
    for i in pos:
        for j in neg:
            if pred[i] > pred[j]:
                auc += 1
            elif pred[i] == pred[j]:
                auc += 0.5
 
    return auc / (len(pos) * len(neg))


def caculate_auc_nlogn(label, pred):
    sorted_indices = np.argsort(pred)
    y_true = np.array(label)[sorted_indices]
    # y_pred = np.array(label)[sorted_indices]

    pos = sum(y_true)
    neg = len(y_true) - pos

    # 这个算法和上面的相比没法加上正负样本预测概率相同时的 0.5
    # 当有正负样本预测概率相同时就这个算法的结果会偏低, 与 sklearn 结果不符
    auc = 0.0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            auc += (i + 1)
    
    auc = (auc - pos * (pos + 1) / 2) / (pos * neg)

    return auc
 
 
if __name__ == '__main__':
    label = [1,0,0,0,1,0,1,0]
    pred = [0.9, 0.8, 0.3, 0.1, 0.4, 0.9, 0.66, 0.7]
    print('caculate_auc_nn: ', caculate_auc_nn(label, pred))
    print('caculate_auc_nlogn: ', caculate_auc_nlogn(label, pred))
 
    fpr, tpr, th = roc_curve(label, pred, pos_label=1)
    print('sklearn auc: ', auc(fpr, tpr))

    print('sklearn roc_auc_score: ', roc_auc_score(label, pred))