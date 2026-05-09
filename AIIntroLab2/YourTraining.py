'''
Softmax 回归。计算accuracy。
'''
import mnist
import numpy as np
import pickle
from scipy import ndimage
from autograd.utils import PermIterator
from util import setseed
from typing import List
from autograd.BaseGraph import Graph
from autograd.BaseNode import *

setseed(0) # 固定随机数种子以提高可复现性

save_path = "model/mymodel.npy"
lr = 1e-3   # 学习率
wd1 = 0  # L1正则化
wd2 = 1e-3  # L2正则化
batchsize = 128

def data_trans(dataset, labels):
    def rot(img):
        img_rot = ndimage.rotate(img, np.random.uniform(-15, 15), reshape=False)
        return img_rot
    def shif(img):
        dx = np.random.uniform(-4.5, 4.5)
        dy = np.random.uniform(-4.5, 4.5)
        img_shif = ndimage.shift(img, shift=(dx, dy), mode='nearest')
        return img_shif
    # def noi(img):
    #     img_noi = img + np.random.normal(0, 0.05, img.shape)
    #     img_noi = np.clip(img_noi, 0, 1)
    #     return img_noi
    # def zoo(img):
    #     scal = np.random.uniform(0.9, 1.1)
    #     img_zoo = ndimage.zoom(img, scal)
    #     h, w = img_zoo.shape
    #     if scal > 1:
    #         h_start = (h - 28) // 2
    #         w_start = (w - 28) // 2
    #         img_zoo = img_zoo[h_start:h_start+28,w_start:w_start+28]
    #     if scal < 1:
    #         pad_h = (28 - h) // 2
    #         pad_w = (28 - w) // 2
    #         img_zoo = np.pad(img_zoo,((pad_h, 28 - h - pad_h),(pad_w, 28 - w - pad_w)),mode='constant')
    #     img_zoo = np.clip(img_zoo, 0, 1)
    #     return img_zoo
    ret_data = []
    ret_labels = []
    # idx_manipulate = np.random.choice(len(dataset), int(len(dataset) * 0.5), replace=False)
    for i, img in enumerate(dataset):
        # ret_data.append(img)
        # ret_labels.append(labels[i])
        # if (i not in idx_manipulate):
        #     continue

        img = img.reshape(28, 28)
        ret_data.append(rot(img).reshape(-1))
        ret_labels.append(labels[i])

        # trans = [shif, noi]
        # idx_tran = np.random.choice(2, 1, replace=False)
        # img_tran = trans[idx_tran[0]](img)
        ret_data.append(shif(img).reshape(-1))
        ret_labels.append(labels[i])

        # if np.random.rand() < 0.1:
        #     ret_data.append(zoo(img).reshape(-1))
        #     ret_labels.append(labels[i])

    return np.array(ret_data), np.array(ret_labels)

if __name__ == "__main__":
    _X = np.concatenate((mnist.trn_X, mnist.val_X), axis=0)
    _Y = np.concatenate((mnist.trn_Y, mnist.val_Y), axis=0)
    X_trans, Y_trans = data_trans(_X, _Y)
    X = np.concatenate((X_trans, _X), axis=0)
    Y = np.concatenate((Y_trans, _Y), axis=0)
    graph = Graph([
        # Dropout(),
        Linear(784, 256),
        BatchNorm(256),
        relu(),

        Dropout(),
        Linear(256, 64),
        BatchNorm(64),
        relu(),

        Linear(64, 10),
        LogSoftmax(),
        NLLLoss(Y)
    ])
    # 训练
    best_train_acc = 0
    dataloader = PermIterator(X.shape[0], batchsize)
    for i in range(1, 35+1):
        hatys = []
        ys = []
        losss = []
        graph.train()
        for perm in dataloader:
            tX = X[perm]
            tY = Y[perm]
            graph[-1].y = tY
            graph.flush()
            pred, loss = graph.forward(tX)[-2:]
            hatys.append(np.argmax(pred, axis=1))
            ys.append(tY)
            graph.backward()
            graph.optimstep(lr, wd1, wd2)
            losss.append(loss)
        loss = np.average(losss)
        acc = np.average(np.concatenate(hatys)==np.concatenate(ys))
        print(f"epoch {i} loss {loss:.3e} acc {acc:.4f}")
        if acc > best_train_acc:
            best_train_acc = acc
            with open(save_path, "wb") as f:
                pickle.dump(graph, f)

