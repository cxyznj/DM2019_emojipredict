import multiprocessing
from gensim.models import word2vec
import gensim
import os
import numpy as np
import csv
import preprocess

def load_traindata(datafilename = 'datasets/train.data', labelfilename = 'datasets/train.solution',
                   emojifilename = 'datasets/emoji.data', bg = 0, ed = 863164):
    train_data = []
    with open(datafilename, encoding='utf-8') as dataf:
        count = bg
        for line in dataf.readlines():
            if count >= ed:
                break
            if count == 0:
                line = line[1:]
            train_data.append(line[:-1])
            count += 1

    emoji = {}
    with open(emojifilename, encoding='utf-8') as emojif:
        count = 0
        for line in emojif.readlines():
            infos = line.split('\t')
            emoji[infos[1][:-1]] = count
            count += 1

    train_label = []
    with open(labelfilename, encoding='utf-8') as labelf:
        count = bg
        for line in labelf.readlines():
            if count >= ed:
                break
            if count == 0:
                train_label.append(emoji[line[2:-2]])
            else:
                train_label.append(emoji[line[1:-2]])
            count += 1
    return train_data, train_label


def load_testdata(datafilename = 'datasets/test.data', bg = 0, ed = 200000):
    test_data = []
    with open(datafilename, encoding='utf-8') as dataf:
        count = bg
        for line in dataf.readlines():
            if count >= ed:
                break
            infos = line.split('\t')
            test_data.append(infos[1][:-1])
            count += 1
    return test_data


def load_words(train_text, test_text, train_label, trainfilename, testfilename, mergefilename):
    if os.path.exists(trainfilename) and os.path.exists(testfilename):
        print("载入分词数据中...")
        def load_cutwords(filename):
            result = []
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    cur = line.split(' ')
                    cur[-1] = cur[-1][:-1]
                    result.append(cur)
            return result
        train_words = load_cutwords(filename=trainfilename)
        test_words = load_cutwords(filename=testfilename)
    else:
        print("处理分词数据中...")
        train_words = preprocess.cutwords(train_text, sfilename=trainfilename)
        test_words = preprocess.cutwords(test_text, sfilename=testfilename)
        #valword = preprocess.feature_select(train_words, train_label) # 卡方检验后的最重要的特征
        #train_words = preprocess.filter_valword(train_words, valword)
        #test_words = preprocess.filter_valword(test_words, valword)
        def save_data(words, filename):
            with open(filename, 'w', encoding='utf-8') as w:
                for word in words:
                    w.write(' '.join(word))
                    w.write('\n')
        save_data(train_words, trainfilename)
        save_data(test_words, testfilename)

        def merge_cutwords(trainpath, testpath, spath):
            with open(trainpath, 'r', encoding='utf-8') as trainf, open(testpath, 'r', encoding='utf-8') as testf, open(
                    spath, 'w', encoding='utf-8') as sf:
                for line in trainf.readlines():
                    sf.write(line)
                for line in testf.readlines():
                    sf.write(line)
        # 合并分词结果，以供word2vec训练
        merge_cutwords(trainfilename, testfilename, mergefilename)
    return train_words, test_words


def load_wordmodel(model_path, train_path, size, window=5, min_count=1, workers = multiprocessing.cpu_count()):
    model_text = model_path.format(size)
    if os.path.exists(model_path):
        print("载入已训练过的分词模型中...")
        model = gensim.models.Word2Vec.load(model_text)
    else:
        print('分词模型训练中...')
        sentences = word2vec.Text8Corpus(train_path)
        model = word2vec.Word2Vec(sentences=sentences, size=size, window=window, min_count=min_count, workers=workers)
        model.save(model_text)
    return model


def load_feature(model, train_words, test_words, trainfilename, testfilename, featurefilename, dictfilename):
    if os.path.exists(featurefilename) and os.path.exists(dictfilename) and os.path.exists(trainfilename) and os.path.exists(testfilename):
        print("载入特征中...")
        feature = np.load(featurefilename, allow_pickle=True)
        with open(dictfilename, 'r', encoding='utf-8') as f:
            a = f.read()
            fdict = eval(a)
        train_feature = np.load(trainfilename, allow_pickle=True)
        test_feature = np.load(testfilename, allow_pickle=True)
    else:
        print("特征提取中...")
        feature = []
        fdict = {}
        train_feature = []
        test_feature = []
        count = 0 # 生成特征矩阵下标
        for line in train_words:
            cur_feature = []
            for word in line:
                if word in model:
                    if word in fdict:
                        cur_feature.append(fdict[word])
                    else:
                        feature.append(model[word])
                        fdict[word] = count
                        cur_feature.append(count)
                        count += 1
                else:
                    cur_feature.append(-1)
            train_feature.append(cur_feature)
        for line in test_words:
            cur_feature = []
            for word in line:
                if word in model:
                    if word in fdict:
                        cur_feature.append(fdict[word])
                    else:
                        feature.append(model[word])
                        fdict[word] = count
                        cur_feature.append(count)
                        count += 1
                else:
                    cur_feature.append(-1)
            test_feature.append(cur_feature)

        train_feature = np.array(train_feature)
        test_feature = np.array(test_feature)
        np.save(trainfilename, train_feature)
        np.save(testfilename, test_feature)
        feature = np.array(feature)
        np.save(featurefilename, feature)
        with open(dictfilename, 'w', encoding='utf-8') as df:
            df.write(str(fdict))
    return train_feature, test_feature, feature, fdict


def save_result(filename, data):
    with open(filename, 'w', newline='') as f:
        csv_write = csv.writer(f, dialect='excel')
        content = ['ID', 'Expected']
        csv_write.writerow(content)

        for i in range(len(data)):
            content[0] = i
            content[1] = data[i]
            csv_write.writerow(content)