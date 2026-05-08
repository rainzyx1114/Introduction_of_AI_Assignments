from numpy.random import rand
import mnist
from answerTree import *
import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
num_tree = 30     # 树的数量
ratio_data = 0.8   # 采样的数据比例
ratio_feat = 0.3 # 采样的特征比例
hyperparams = {
    "depth":10, 
    "purity_bound":1e-4,
    "gainfunc": negginiDA
    } # 每颗树的超参数


def buildtrees(X, Y):
    """
    构建随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: n, 样本的label
    @return: List of DecisionTrees, 随机森林
    """
    # TODO: YOUR CODE HERE
    # 提示：整体流程包括样本扰动、属性扰动和预测输出
    trees = []
    data_num = int(X.shape[0] * ratio_data)
    feat_num = int(X.shape[1] * ratio_feat)
    for _ in range(num_tree):
        rows_idx = np.sort(np.random.choice(X.shape[0], data_num, replace=False))
        cols_idx = np.sort(np.random.choice(X.shape[1], feat_num, replace=False))
        X_rand = X[rows_idx, :]
        Y_rand = Y[rows_idx]
        tree = buildTree(X_rand, Y_rand, list(cols_idx), **hyperparams)
        trees.append(tree)
    return trees    

def infertrees(trees, X):
    """
    随机森林预测
    @param trees: 随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @return: n, 预测的label
    """
    pred = [inferTree(tree, X)  for tree in trees]
    pred = list(filter(lambda x: not np.isnan(x), pred))
    upred, ucnt = np.unique(pred, return_counts=True)
    return upred[np.argmax(ucnt)]
