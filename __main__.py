import load_datasets
import os
import gensim
import numpy as np
import gc
import Model
import torch
import SVM
from sklearn import svm
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
import time
import MLP
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


trainwords_path = "./preprocessed/trainword.data"
testwords_path = "./preprocessed/testword.data"
ttwords_path = "./preprocessed/ttword.data"

wordmodel_path = './preprocessed/wordmodel_256.model'

trainfeature_path = "./preprocessed/trainfeature.npy"
testfeature_path = "./preprocessed/testfeature.npy"
feature_path = "./preprocessed/feature.npy"
featuredict_path = "./preprocessed/feature.dict"

cnnmodel_path = "./model/cnn.pth"
result_path = "./result/result.csv"

svmtrain_path = "./svm/trainfeature.npy"
svmtest_path = "./svm/testfeature.npy"
svmmodel_path = "./svm/model_0.svm"

mlpmodel_path = "./mlp/model0.mlp"

feature_len = 256
sentence_len = 32

# filter=[2:128, 3:128, 4:128, 5:128], feature_len=512, acc=0.142187
# filter=[3:128, 4:128, 5:128], feature_len=512, acc=0.147016

if __name__ == '__main__':
    # train_text/test_text是原始文本，train_label是数值化后的标签
    print("载入数据中...")
    train_text, train_label = load_datasets.load_traindata()
    test_text = load_datasets.load_testdata()
    # train_words/test_words是分词后的结果，数据类型为list，每一行为一个句子，一行中的每一项为句子中的一个分词
    train_words, test_words = load_datasets.load_words(train_text, test_text, train_label, trainfilename=trainwords_path, testfilename=testwords_path, mergefilename=ttwords_path)
    # 采用word2vec模型
    wordmodel = load_datasets.load_wordmodel(model_path=wordmodel_path, train_path=ttwords_path, size=feature_len)
    # train_feature/test_feature是特征矩阵features中特征的下标，其中每一条特征对应一个字,fdict是字到特征下标的字典映射
    train_feature, test_feature, features, fdict = load_datasets.load_feature(wordmodel, train_words, test_words, trainfeature_path, testfeature_path, feature_path, featuredict_path)
    # embedding之后
    #features = list(features)
    #features.append([.0]*feature_len)
    #features = np.array(features)

    # 回收空间
    del (train_text)
    del (test_text)
    del (train_words)
    del (test_words)
    gc.collect()

    result = MLP.bagging_data(features, test_feature, feature_len)
    # MLP method
    #if os.path.exists(mlpmodel_path):
    #    print("载入MLP模型中...")
    #    clf = joblib.load(mlpmodel_path)  # 载入模型
    #else:
    #    print("训练MLP模型中...")
    #    clf = MLP.MLP()
    #    clf.train(features, train_feature, train_label, feature_len)
    #    joblib.dump(clf, mlpmodel_path)  # 保存模型

    #result = clf.predict(features, test_feature, feature_len)
    load_datasets.save_result(result_path, result)

    # 约10%为验证集，其他为训练集
    #valid_size = 128000
    #valid_feature = train_feature[:valid_size]
    #valid_label = train_label[:valid_size]
    #train_feature = train_feature[valid_size:]
    #train_label = train_label[valid_size:]

    # SVM method(failed)
    '''
    train_feature = SVM.genfeature(features, train_feature, feature_len, svmtrain_path)
    train_label = np.array(train_label)

    valid_feature = SVM.genfeature(features, valid_feature, feature_len, svmtest_path)
    valid_label = np.array(valid_label)

    # SVM method
    if os.path.exists(svmmodel_path):
        print("载入SVM模型中...")
        clf = joblib.load(svmmodel_path)  # 载入模型
    else:
        print("训练SVM模型中...")
        clf = OneVsOneClassifier(LinearSVC(verbose=True, C=1e-3)).fit(train_feature, train_label)
        # clf = svm.SVC(gamma='auto', decision_function_shape='ovo', verbose=True, max_iter=1000)
        clf.fit(train_feature, train_label)
        print("SVM模型训练完毕!")
        joblib.dump(clf, svmmodel_path) # 保存模型

    result = clf.score(valid_feature, valid_label)
    print("验证集上的平均准确率为: %.6f" % result)
    '''
    # CNN method
    '''
    epoch = 1
    CNNmodel = Model.model(features, enum=epoch)
    if os.path.exists(cnnmodel_path):
        print("载入模型中...")
        CNNmodel.load_model(cnnmodel_path)
    else:
        print("建立新模型中...")

    bgtime = time.time()
    Model.train(CNNmodel, features, train_feature, train_label, sentence_len, feature_len)
    Model.valid(CNNmodel, features, valid_feature, valid_label, sentence_len, feature_len)
    edtime = time.time()
    print("共计用时%d s" %(edtime-bgtime))
    #result = Model.predict(CNNmodel, features, test_feature, sentence_len, feature_len)
    #load_datasets.save_result(result_path, result)
    '''