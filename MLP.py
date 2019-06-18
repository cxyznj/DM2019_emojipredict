from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import os
import numpy as np
from sklearn.externals import joblib

class MLP:
    def __init__(self):
        self.clf = MLPClassifier(solver='adam', hidden_layer_sizes=(256, 256), batch_size=256, tol=1e-4,
                                 learning_rate_init=1e-3, learning_rate='adaptive', alpha=1e-3,
                                verbose=True, early_stopping=True, n_iter_no_change=5)

    def train(self, features, train_feature, train_label, feature_len, train_words, fdict):
        # 数据预处理使得一个句子是一个加权平均后的一维向量
        prepath = "./preprocessed/mlptrain.data.npy"
        if os.path.exists(prepath):
            print("载入mlp数据中...")
            train_feature = np.load(prepath)
        else:
            print("处理mlp数据中...")
            #train_feature = self.weight_preprocess(features, train_feature, feature_len, train_words, fdict)
            train_feature = self.preprocess(features, train_feature, feature_len)
            train_feature = np.array(train_feature)
            np.save(prepath, train_feature)
        self.clf.fit(train_feature, train_label)


    def predict(self, features, test_feature, feature_len, train_words, fdict):
        prepath = "./preprocessed/mlptest.data.npy"
        if os.path.exists(prepath):
            print("载入mlp数据中...")
            test_feature = np.load(prepath)
        else:
            print("处理mlp数据中...")
            #test_feature = self.weight_preprocess(features, test_feature, feature_len, train_words, fdict)
            test_feature = self.preprocess(features, test_feature, feature_len)
            test_feature = np.array(test_feature)
            np.save(prepath, test_feature)
        return self.clf.predict(test_feature)


    def predict_proba(self, features, test_feature, feature_len):
        prepath = "./preprocessed/mlptest.data"
        if os.path.exists(prepath):
            print("载入mlp数据中...")
            test_feature = np.load(prepath)
        else:
            print("处理mlp数据中...")
            test_feature = self.preprocess(features, test_feature, feature_len)
            test_feature = np.array(test_feature)
            np.save(prepath, test_feature)
        return self.clf.predict_proba(test_feature)


    def preprocess(self, features, t_feature, feature_len):
        result = []
        # s是一个句子的各词的idx列表
        for s in t_feature:
            ave = [.0] * feature_len
            count = 0
            for wordidx in s:
                #if wordidx == -1: continue
                f = features[wordidx]
                ave += f
                count += 1
            if count > 0:
                ave /= count
            result.append(ave)
        return result


    def weight_preprocess(self, features, t_feature, feature_len, train_words, fdict):
        for i, items in enumerate(train_words):
            s = ""
            for item in items:
                s = s + item + ' '
            train_words[i] = s[:-1]
        tfidf_vec = TfidfVectorizer()
        tfidf_matrix = tfidf_vec.fit_transform(train_words)
        #print(tfidf_matrix.shape)
        #print(type(tfidf_matrix))
        word_tfidx = tfidf_vec.vocabulary_
        #print(len(word_tfidx))

        # fdict是t_feature中idx到word的映射，word_tfidx是word到tfidf_matrix稀疏矩阵下标的映射
        # 先根据fdict找到word，再根据word找到稀疏矩阵下标，将下标对应的值做加权平均即可
        print("开始加权")
        result = []
        errcount = 0
        allcount = 0
        # s是一个句子的各词的idx列表
        for i, s in enumerate(t_feature):
            ave = [.0] * feature_len
            w2v = []
            weight = []
            count = 0 # 总字数
            vcount = 0 # 有效字数
            for wordidx in s:
                allcount += 1
                count += 1
                w2v.append(features[wordidx])
                try:
                    word = fdict[wordidx]
                    tfidx = word_tfidx[word]
                    tfval = tfidf_matrix[i, tfidx]
                    weight.append(tfval)
                    vcount += 1
                except:
                    errcount += 1
                    weight.append(.0)
                    continue
            vpercent = vcount/count # 有效率
            # 开始加权平均
            s = sum(weight)
            for j in range(count):
                if weight[j] != .0:
                    ave += (weight[j]/s) * w2v[j] * vpercent
                else:
                    ave += (1/count) * w2v[j]
            result.append(ave)
        print("err = %.6f" %(errcount/allcount))
        return result


def bagging(clfs, features, test_feature, feature_len, weights):
    result = []
    test_len = len(test_feature) # 应该是200000
    predict_vals = []
    count = 0
    for clf in clfs:
        pv = clf.predict(features, test_feature, feature_len)
        predict_vals.append(pv)
        count += 1
    # 用概率均值的结果
    '''
    for i in range(test_len):
        prob = [.0] * 72
        for j in range(count):
            prob = prob + predict_vals[j][i]
        prob /= count
        result.append(prob.index(prob(max)))
    '''
    # 用投票的结果
    sw = sum(weights)
    for i in range(count):
        weights[i] = weights[i] / sw
    for i in range(test_len):
        candipre = {}
        for j in range(count):
            v = predict_vals[j][i]
            if v in candipre:
                candipre[v] += weights[j]
            else:
                candipre[v] = weights[j]
        # 找出字典中最大的元素
        maxidx = 0
        maxnum = 0
        for item in candipre:
            if candipre[item] > maxnum:
                maxnum = candipre[item]
                maxidx = item
        result.append(maxidx)
    return result

def bagging_data(features, test_feature, feature_len):
    clfs = []
    clfs.append(joblib.load("./mlpmodel/200.200.r5s1e-4.a1e-2.relu.ninc5.tol1e-4(0.180752).mlp"))
    clfs.append(joblib.load("./mlpmodel/300.300.r1e-3.a1e-2.relu.ninc5.tol1e-4(0.180266).mlp"))
    clfs.append(joblib.load("./mlpmodel/128.128.r5s1e-4.a1e-2.relu.ninc5.tol1e-4(0.179524).mlp"))
    clfs.append(joblib.load("./mlpmodel/256.256.r1e-3.a1e-2.relu.ninc5.tol1e-4(0.176790).mlp"))
    clfs.append(joblib.load("./mlpmodel/150.150.r5s1e-4.a1e-3.relu.ninc5.tol1e-4(0.177427).mlp"))
    clfs.append(joblib.load("./mlpmodel/400.300.r5s1e-4.a5s1e-3.relu.ninc5.tol1e-4(0.178505).mlp"))
    clfs.append(joblib.load("./mlpmodel/128.128.128.r5s1e-4.a1e-2.relu.ninc5.tol1e-4(0.177590).mlp"))
    clfs.append(joblib.load("./mlpmodel/512.512.r5s1e-4.a1e-3.relu.ninc5.tol1e-4(0.174079).mlp"))
    clfs.append(joblib.load("./mlpmodel/512.512.r1e-3.a1e-2.relu.ninc5.tol1e-4(0.173488).mlp"))
    clfs.append(joblib.load("./mlpmodel/150.150.r1e-3.a5s1e-2.relu.ninc5.tol1e-4(0.175029).mlp"))
    weights = [0.180752, 0.180266, 0.179524, 0.176790, 0.177427, 0.178505, 0.177590, 0.174079, 0.173488, 0.175029]
    result = bagging(clfs, features, test_feature, feature_len, weights)
    #print("acc =", accuracy_score(result, train_label[:valid_size]))
    #print("f1 =", f1_score(result, train_label[:valid_size], average='micro'))

    return result