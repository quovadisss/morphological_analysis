import re
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer


# Preprocessing
def preprocess_wordlist(txt):
    # txt = txt.lower()
    txt = txt.replace('\t', ' ')
    txt = txt.replace('\n', '')
    txt = txt.replace('.', '. ')
    # txt = re.sub('[^a-zA-Z0-9]',' ',txt)
    txt = re.sub('[0-9]', ' ', txt)
    txt = txt.replace(u'\xa0', u' ')
    txt = txt.replace('  ', ' ')
    txt = txt.replace('   ', ' ')
    txt = txt.replace(' .', '.')
    txt = re.sub(
        '[-=+,#/\?:“”^"—$€£@*\"※~&%ㆍ!』’\\‘|\(\)\[\]\<\>`\'…》]', '', txt)
    txt = txt.strip()
    # 단어 길이 3개이하 삭제
    # shortword = re.compile(r'\W*\b\w{1,2}\b')
    # txt = shortword.sub('', txt)
    return txt


# extract two words after a specific word
def get_index(doc, cword):
    tokens = word_tokenize(doc)
    words_index = []
    for i, j in enumerate(tokens):
        if j == cword:
            word_index = [i+1, i+2]
            words_index.append(word_index)
        else:
            pass

    words = []
    for two_index in words_index:
        word = []
        for i, j in enumerate(tokens):
            if two_index[0] == i:
                word.append(j)
            elif two_index[1] == i:
                word.append(j)
            else:
                pass
        words.append(word)
    return words


def build_network(text):
    split_txt = [' '.join(i) for i in text]
    vectorizer = CountVectorizer()
    net = vectorizer.fit_transform(split_txt)
    vocab = vectorizer.vocabulary_
    voca = [i for i, j in sorted(vocab.items(), key=lambda x: x[1])]
    net = sp.csr_matrix(net.toarray())
    adj = net.T.dot(net).toarray()
    n = adj.shape[0]
    adj[range(n), range(n)] = 0
    df = pd.DataFrame(adj, index=voca, columns=voca)
    return df


def unify_docs(doc):
    result = []
    for i in doc:
        result.extend(i)
    return np.unique(result)
