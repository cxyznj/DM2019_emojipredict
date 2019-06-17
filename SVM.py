from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def getfeature(train_words, test_words, train_label):
    train_len = len(train_words)
    key = train_words + test_words
    for i, items in enumerate(key):
        s = ""
        for item in items:
            s = s + item + ' '
        key[i] = s[:-1]
    tfidf_vec = TfidfVectorizer()
    tfidf_matrix = tfidf_vec.fit_transform(key)
    Kmodel = SelectKBest(chi2, k=150000)  # 选择k个最佳特征
    train_feature = Kmodel.fit_transform(tfidf_matrix[:train_len], train_label)
    test_feature = Kmodel.transform(tfidf_matrix[train_len:])
    return train_feature, test_feature