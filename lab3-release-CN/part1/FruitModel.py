import math
from SST_2.dataset import traindataset, minitraindataset
from fruit import get_document, tokenize
import pickle
import numpy as np
from importlib.machinery import SourcelessFileLoader
from autograd.BaseGraph import Graph
from autograd.BaseNode import *

class NullModel:
    def __init__(self):
        pass

    def __call__(self, text):
        return 0


class NaiveBayesModel:
    def __init__(self):
        self.dataset = traindataset() # 完整训练集，需较长加载时间
        # self.dataset = minitraindataset() # 用来调试的小训练集，仅用于检查代码语法正确性

        # 以下内容可根据需要自行修改，不修改也可以完成本题
        self.token_num = [{}, {}] # token在正负样本中出现次数
        self.V = 0 #语料库token数量
        self.pos_neg_num = [0, 0] # 正负样本数量
        self.count()

    def count(self):
        # TODO: YOUR CODE HERE
        # 提示：统计token分布不需要返回值
        for data, label in self.dataset:
            self.pos_neg_num[label] += 1
            for token in data:
                if token not in self.token_num[label]:
                    self.token_num[label][token] = 1
                    if token not in self.token_num[1 - label]:
                        self.V += 1
                else:
                    self.token_num[label][token] += 1

    def __call__(self, text):
        # TODO: YOUR CODE HERE
        # 返回1或0代表当前句子分类为正/负样本
        P_pos = self.pos_neg_num[1] / (self.pos_neg_num[0] + self.pos_neg_num[1])
        P_neg = self.pos_neg_num[0] / (self.pos_neg_num[0] + self.pos_neg_num[1])
        nums_pos = 0
        nums_neg = 0
        for num in self.token_num[0].values():
            nums_neg += num
        for num in self.token_num[1].values():
            nums_pos += num
        tokens_pos = []
        tokens_neg = []
        for token in text:
            try:
                prob = (self.token_num[0][token] + 1) / (nums_neg + self.V)
            except:
                prob = 1 / (nums_neg + self.V)
            tokens_neg.append(math.log(prob))
        for token in text:
            try:
                prob = (self.token_num[1][token] + 1) / (nums_pos + self.V)
            except:
                prob = 1 / (nums_pos + self.V)
            tokens_pos.append(math.log(prob))
        p_pos = math.log(P_pos)
        p_neg = math.log(P_neg)
        for prob in tokens_pos:
            p_pos += prob
        for prob in tokens_neg:
            p_neg += prob
        return (p_pos >= p_neg)


def buildGraph(dim, num_classes, L): #dim: 输入一维向量长度, num_classes:分类数
    # 以下类均需要在BaseNode.py中实现
    # 也可自行修改模型结构
    nodes = [Attention(dim), relu(), LayerNorm((L, dim)), ResLinear(dim), relu(), LayerNorm((L, dim)), Mean(1), Linear(dim, num_classes), LogSoftmax(), NLLLoss(num_classes)]
    
    graph = Graph(nodes)
    return graph


save_path = "model/attention.npy"

class Embedding():
    def __init__(self):
        self.emb = dict() 
        with open("words.txt", encoding='utf-8') as f: #word.txt存储了每个token对应的feature向量，self.emb是一个存储了token-feature键值对的Dict()，可直接调用使用
            for i in range(50000):
                row = next(f).split()
                word = row[0]
                vector = np.array([float(x) for x in row[1:]])
                self.emb[word] = vector
        
    def __call__(self, text, max_len=50):
        # TODO: YOUR CODE HERE
        # 利用self.emb将句子映射为一个二维向量（LxD），注意，同时需要修改训练代码中的网络维度部分
        # 默认长度L为50，特征维度D为100
        # 提示: 考虑句子如何对齐长度，且可能存在空句子情况（即所有单词均不在emd表内）
        res = np.zeros((max_len, 100))  
        for i, token in enumerate(text):
            if i >= max_len:
                break
            if token in self.emb:
                res[i] =  self.emb[token]
        # print(embedding[0].shape, "I'm here")
        return res


class AttentionModel():
    def __init__(self):
        self.embedding = Embedding()
        with open(save_path, "rb") as f:
            self.network = pickle.load(f)
        self.network.eval()
        self.network.flush()

    def __call__(self, text, max_len=50):
        X = self.embedding(text, max_len)
        X = np.expand_dims(X, 0)
        pred = self.network.forward(X, removelossnode=1)[-1]
        haty = np.argmax(pred, axis=-1)
        return haty[0]


class QAModel():
    def __init__(self):
        self.document_list = get_document()

    def tf(self, word, document):
        # TODO: YOUR CODE HERE
        # 返回单词在文档中的频度
        # document变量结构请参考fruit.py中get_document()函数
        cnt_total = len(document['document'])
        cnt_word = document['document'].count(word)
        return math.log10(cnt_word / cnt_total + 1)
        raise NotImplementedError  

    def idf(self, word):
        # TODO: YOUR CODE HERE
        # 返回单词IDF值，提示：你需要利用self.document_list来遍历所有文档
        # 注意python整除与整数除法的区别
        cnt_docs = 0
        cnt_docs_contain = 0
        for document in self.document_list:
            cnt_docs += 1
            if word in document['document']:
                cnt_docs_contain += 1
        return math.log10(cnt_docs / (cnt_docs_contain + 1))
        raise NotImplementedError
    
    def tfidf(self, word, document):
        # TODO: YOUR CODE HERE
        # 返回TF-IDF值
        return self.tf(word, document) * self.idf(word)
        raise NotImplementedError

    def __call__(self, query):
        query = tokenize(query) # 将问题token化
        # TODO: YOUR CODE HERE
        # 利用上述函数来实现QA
        # 提示：你需要根据TF-IDF值来选择一个最合适的文档，再根据IDF值选择最合适的句子
        # 返回时请返回原本句子，而不是token化后的句子，可以参考README中数据结构部分以及fruit.py中用于数据处理的get_document()函数
        max_score = 0
        best_document = {}
        for document in self.document_list:
            score = 0
            for word in query:
                score += self.tfidf(word, document)
            if score > max_score:
                max_score = score
                best_document = document
        max_score = 0
        best_sentence = None
        for sentence in best_document['sentences']:
            score = 0
            for word in query:
                if word in sentence[0]:
                    score += self.idf(word)
            if score > max_score:
                max_score = score
                best_sentence = sentence[1]
        return best_sentence
        raise NotImplementedError

modeldict = {
    "Null": NullModel,
    "Naive": NaiveBayesModel,
    "Attn": AttentionModel,
    "QA": QAModel,
}


if __name__ == '__main__':
    embedding = Embedding()
    lr = 3e-3   # 学习率
    wd1 = 0  # L1正则化
    wd2 = 1e-4  # L2正则化
    batchsize = 64
    max_epoch = 10
    
    max_L = 50
    num_classes = 2
    feature_D = 100
    
    graph = buildGraph(feature_D, num_classes, max_L) # 维度可以自行修改

    # 训练
    # 完整训练集训练有点慢
    best_train_acc = 0
    dataloader = traindataset(shuffle=True) # 完整训练集
    # dataloader = minitraindataset(shuffle=True) # 用来调试的小训练集
    for i in range(1, max_epoch+1):
        hatys = []
        ys = []
        losss = []
        graph.train()
        X = []
        Y = []
        cnt = 0
        for text, label in dataloader:
            x = embedding(text, max_L)
            # print(x.shape, "no")
            label = np.zeros((1)).astype(np.int32) + label
            X.append(x)
            Y.append(label)
            cnt += 1
            if cnt == batchsize:
                X = np.stack(X, 0)
                Y = np.concatenate(Y, 0)
                graph[-1].y = Y
                graph.flush()
                pred, loss = graph.forward(X)[-2:]
                hatys.append(np.argmax(pred, axis=-1))
                ys.append(Y)
                graph.backward()
                graph.optimstep(lr, wd1, wd2)
                losss.append(loss)
                cnt = 0
                X = []
                Y = []

        loss = np.average(losss)
        acc = np.average(np.concatenate(hatys)==np.concatenate(ys))
        print(f"epoch {i} loss {loss:.3e} acc {acc:.4f}")
        if acc > best_train_acc:
            best_train_acc = acc
            with open(save_path, "wb") as f:
                pickle.dump(graph, f)