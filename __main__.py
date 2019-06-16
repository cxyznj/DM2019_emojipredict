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

mlpmodel_path = "./mlp/150.150.r1e-3.a5s1e-2.relu.ninc5.tol1e-4.mlp"

feature_len = 256
sentence_len = 32
valid_size = 128000
epoch = 1

# filter=[2:128, 3:128, 4:128, 5:128], feature_len=512, acc=0.142187
# filter=[3:128, 4:128, 5:128], feature_len=512, acc=0.147016
# filter=[3:128, 4:128, 5:128, 6:64], feature_len=512, acc=

if __name__ == '__main__':
    # train_text/test_text是原始文本，train_label是数值化后的标签
    print("载入数据中...")
    train_text, train_label = load_datasets.load_traindata()
    test_text = load_datasets.load_testdata()
    #train_words = test_words = None
    #if not (os.path.exists(trainfeature_path) and os.path.exists(testfeature_path) and os.path.exists(feature_path) and os.path.exists(featuredict_path)):
    # train_words/test_words是list类型的数据，每一行为一个句子，一行中的每一项为句子中的一个分词
    train_words, test_words = load_datasets.load_words(train_text, test_text, train_label, trainfilename=trainwords_path, testfilename=testwords_path, mergefilename=ttwords_path)
    # 采用word2vec模型
    wordmodel = load_datasets.load_wordmodel(model_path=wordmodel_path, train_path=ttwords_path, size=feature_len)

    train_feature, test_feature, features, fdict = load_datasets.load_feature(wordmodel, train_words, test_words, trainfeature_path, testfeature_path, feature_path, featuredict_path)
    print("feature总数为%d" % len(features))
    # embedding之后
    #features = list(features)
    #features.append([.0]*feature_len)
    #features = np.array(features)
    # 回收空间
    del (train_text)
    del (test_text)
    #del (train_words)
    del (test_words)
    gc.collect()
    #trfdict = {}
    #for item in fdict:
    #    trfdict[fdict[item]] = item

    # 约10%为验证集，其他为训练集
    #valid_feature = train_feature[:valid_size]
    #valid_label = train_label[:valid_size]
    #train_feature = train_feature[valid_size:]
    #train_label = train_label[valid_size:]

    # MLP method
    #if os.path.exists(mlpmodel_path):
    #    print("载入MLP模型中...")
    #    clf = joblib.load(mlpmodel_path)  # 载入模型
    #else:
    #    print("训练MLP模型中...")
    #    clf = MLP.MLP()
    #    clf.train(features, train_feature, train_label, feature_len, train_words, fdict)
    #    joblib.dump(clf, mlpmodel_path)  # 保存模型

    clfs = []
    clfs.append(joblib.load("./mlp/200.200.r5s1e-4.a1e-2.relu.ninc5.tol1e-4(0.180752).mlp"))
    clfs.append(joblib.load("./mlp/300.300.r1e-3.a1e-2.relu.ninc5.tol1e-4(0.180266).mlp"))
    clfs.append(joblib.load("./mlp/128.128.r5s1e-4.a1e-2.relu.ninc5.tol1e-4(0.179524).mlp"))
    clfs.append(joblib.load("./mlp/256.256.r1e-3.a1e-2.relu.ninc5.tol1e-4(0.176790).mlp"))
    clfs.append(joblib.load("./mlp/150.150.r5s1e-4.a1e-3.relu.ninc5.tol1e-4(0.177427).mlp"))
    clfs.append(joblib.load("./mlp/400.300.r5s1e-4.a5s1e-3.relu.ninc5.tol1e-4(0.178505).mlp"))
    clfs.append(joblib.load("./mlp/128.128.128.r5s1e-4.a1e-2.relu.ninc5.tol1e-4(0.177590).mlp"))
    clfs.append(joblib.load("./mlp/512.512.r5s1e-4.a1e-3.relu.ninc5.tol1e-4(0.174079).mlp"))
    clfs.append(joblib.load("./mlp/512.512.r1e-3.a1e-2.relu.ninc5.tol1e-4(0.173488).mlp"))
    clfs.append(joblib.load("./mlp/150.150.r1e-3.a5s1e-2.relu.ninc5.tol1e-4(0.175029).mlp"))
    weights = [0.180752, 0.180266, 0.179524, 0.176790, 0.177427, 0.178505, 0.177590, 0.174079, 0.173488, 0.175029]
    result = MLP.bagging(clfs, features, test_feature, feature_len, weights)
    #print("acc =", accuracy_score(result, train_label[:valid_size]))
    #print("f1 =", f1_score(result, train_label[:valid_size], average='micro'))

    #result = clf.predict(features, test_feature, feature_len)
    load_datasets.save_result(result_path, result)

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
    for i in range(3):
        CNNmodel = Model.model(features, enum=epoch)
        if os.path.exists(cnnmodel_path):
            print("载入模型中...")
            CNNmodel.load_model(cnnmodel_path)
        else:
            print("建立新模型中...")

        bgtime = time.time()
        Model.train(CNNmodel, features, train_feature, train_label, sentence_len, feature_len)
        Model.valid(CNNmodel, features, valid_feature, valid_label, sentence_len, feature_len)
        ##Model.valid(CNNmodel, features, train_feature[:valid_size], train_label[:valid_size], sentence_len, feature_len)
        edtime = time.time()
        print("共计用时", edtime-bgtime, 's')
        #result = Model.predict(CNNmodel, features, test_feature, sentence_len, feature_len)
        #load_datasets.save_result(result_path, result)
    '''
    '''
    train_acc = .0
    valid_acc = Model.valid(CNNmodel, features, valid_feature, valid_label, feature_len)[1]
    overfitcount = 0
    for i in range(epoch):
        print("$$$$第%d代训练$$$$" %i)
        tacc = Model.train(CNNmodel, features, train_feature[:half], train_label[:half], feature_len)[1]
        vacc = Model.valid(CNNmodel, features, valid_feature, valid_label, feature_len)[1]
        if (tacc > train_acc) and (vacc < valid_acc):
            overfitcount += 1
            if overfitcount > 1:
                print("Warning: 存在过拟合的风险，训练代数%d/%d in first part" %(i, epoch))
                break
        else:
            overfitcount = 0
            CNNmodel.save_model(cnnmodelpath)
        train_acc = tacc
        valid_acc = vacc

        tacc = Model.train(CNNmodel, features, train_feature[half:], train_label[half:], feature_len)[1]
        vacc = Model.valid(CNNmodel, features, valid_feature, valid_label, feature_len)[1]
        if (tacc > train_acc) and (vacc < valid_acc):
            overfitcount += 1
            if overfitcount > 1:
                print("Warning: 存在过拟合的风险，训练代数%d/%d in first part" %(i, epoch))
                break
        else:
            overfitcount = 0
            CNNmodel.save_model(cnnmodelpath)
        train_acc = tacc
        valid_acc = vacc
    #result = predict(CNNmodel, features, test_feature, feature_len)
    #load_datasets.save_result(resultpath, result)
    # train(CNNmodel, features, train_feature, train_label, feature_len)
    # valid(CNNmodel, features, valid_feature, valid_label, feature_len)
    '''