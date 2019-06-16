import jieba
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np

def cutwords(wordlists, sfilename):
    result = []
    # 严格限制标点符号
    #strict_punctuation = '。，、＇：∶；?‘’“”〝〞ˆˇ﹕︰﹔﹖﹑·¨….¸;！´？！～—ˉ｜‖＂〃｀@﹫¡¿﹏﹋﹌︴々﹟#﹩$﹠&﹪%*﹡﹢﹦﹤‐￣¯―﹨ˆ˜﹍﹎+=<­­＿_-\ˇ~﹉﹊（）〈〉‹›﹛﹜『』〖〗［］《》〔〕{}「」【】︵︷︿︹︽_﹁﹃︻︶︸﹀︺︾ˉ﹂﹄︼'
    # 简单限制标点符号
    #simple_punctuation = '’!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~'
    # 去除标点符号
    #punctuation = simple_punctuation + strict_punctuation

    meaningless_word = ['的', '了']
    useful_symbol = ['…', '!', '！', '~', '～', '？', '?', '...', '.']

    for wordlist in wordlists:
        # seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
        # wordlist = re.sub('[{0}]+'.format(punctuation), '', wordlist.strip())
        words = jieba.cut(wordlist, cut_all=False)
        cur_result = []
        for word in words:
            if word == ' ':
                continue
            #if word in meaningless_word:
            #    continue
            #cflag = False
            #for w in word:
            #    if not (('\u4e00' <= w <= '\u9fff') or ('a' <= w <= 'z')
            #            or ('A' <= w <= 'Z') or ('0' <= w <= '9') or (w in useful_symbol)):
            #        cflag = True
            #        break
            #if cflag:
            #    continue
            cur_result.append(word)
        result.append(cur_result)
    #with open(sfilename, 'w', encoding='utf-8') as w:
    #    for words in result:
    #        w.write(' '.join(words))
    #        w.write('\n')
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
    return valword


def filter_valword(sentence, valword):
    result = []
    for i, words in enumerate(sentence):
        cur = []
        for word in words:
            if word in valword:
                cur.append(word)
        result.append(cur)
        if i%1000 == 0:
            print(i)
    return result