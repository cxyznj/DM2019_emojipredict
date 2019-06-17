import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np

def cutwords(wordlists, sfilename):
    result = []
    meaningless_word = ['的', '了', '你', '我', '他', '这', '那', '个']
    useful_symbol = ['…', '!', '！', '~', '～', '？', '?', '...', '.']

    for wordlist in wordlists:
        words = jieba.cut(wordlist, cut_all=False)
        cur_result = []
        for word in words:
            #if word == ' ':
            #    continue
            if word in meaningless_word:
                continue
            #cflag = False
            #for w in word:
            #    if not (('\u4e00' <= w <= '\u9fff') or ('a' <= w <= 'z')
            #            or ('A' <= w <= 'Z') or ('0' <= w <= '9') or (w in useful_symbol)):
            #        cflag = True
            #        break
            #if cflag: continue
            cur_result.append(word)
        result.append(cur_result)
    return result


def feature_select(sentence, label):
    print("begin feature select")
    key = sentence.copy()
    for i, items in enumerate(key):
        s = ""
        for item in items:
            s = s + item + ' '
        key[i] = s[:-1]

    tfidf_vec = TfidfVectorizer()
    tfidf_matrix = tfidf_vec.fit_transform(key)
    Kmodel = SelectKBest(chi2, k=100000)  # 选择k个最佳特征
    Kmodel.fit_transform(tfidf_matrix, label)  # iris.data是特征数据，iris.target是标签数据，该函数可以选择出k个特征
    feature_names = tfidf_vec.get_feature_names()
    valword = []
    for i in Kmodel.get_support(indices=True):
        valword.append(feature_names[i])
    return set(valword)


def filter_valword(sentence, valword):
    result = []
    for i, words in enumerate(sentence):
        cur = []
        for word in words:
            if word in valword:
                cur.append(word)
        result.append(cur)
    return result

