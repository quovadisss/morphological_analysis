import argparse
import spacy
import pickle
import time
import gensim
import pandas as pd
import numpy as np
import scipy.sparse as sp

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag

from utils import *


# for system
def get_system_words(x):
    get_all = []
    for document in x:
        doc = document.lower()
        sys_words = ['for', 'to']
        get_all_words = []
        for i in sys_words:
            get_words = get_index(doc, i)
            get_all_words.extend(get_words)
        get_all.append(get_all_words)
    return get_all


# for technology
def get_tech_words(x):
    get_all = []
    for document in x:
        doc = document.lower()
        sys_words = ['by', 'with', 'using']
        get_all_words = []
        for i in sys_words:
            get_words = get_index(doc, i)
            get_all_words.extend(get_words)
        get_all.append(get_all_words)
    return get_all


# extract only verb or noun with stopwords
def stopwords_v_n(x, vorn):
    stop = stopwords.words('english')
    stop.extend(['healthcare', 'say', 'be', 'health',
                 'company', 'patient', 'others', 'help', 'a', '.'])
    lemmatizer = WordNetLemmatizer()

    result = []
    for doc in x:
        stopped = []
        for words in doc:
            if len(words) == 2:
                if words[0] not in stop:
                    stopped.append(words[0])
                else:
                    if words[1] not in stop:
                        stopped.append(words[1])
                    else:
                        pass
            elif len(words) == 1:
                if words[0] not in stop:
                    stopped.append(words[0])

        pos_list = []
        for word in stopped:
            pos_tagging = pos_tag([word])
            if pos_tagging[0][1][:2] == vorn[0]:
                pos_list.append(pos_tagging[0][0])
        lemma = [lemmatizer.lemmatize(i, pos=vorn[1]) for i in pos_list]
        result.append(lemma)

    return result


# extract only noun with stopwords
def people_noun(x):
    stop = stopwords.words('english')
    stop.extend(['healthcare', 'say', 'be', 'health',
                 'company', 'patient', 'others', 'help', 'a', '.'])

    def is_noun(pos): return pos[:2] == 'NN'

    noun_list = []
    for sentence in x:
        tokens = word_tokenize(sentence)
        noun = [word for (word, pos) in pos_tag(tokens) if is_noun(pos)]
        noun_list.extend(noun)

    lemmatizer = WordNetLemmatizer()
    lemma = [lemmatizer.lemmatize(i, pos='n') for i in noun_list]
    result = [i for i in lemma if i not in stop]

    return result


# Extract norp, org, prodcut by NER
def people_ner(x):
    stop = stopwords.words('english')
    stop.extend(['healthcare', 'say', 'be', 'health',
                 'company', 'patient', 'others', 'help', 'a', '.'])
    norps = []
    orgs = []
    products = []
    for e_, d in enumerate(x):
        all_corpus = [j.strip() for j in d.split('.')][:-1]

        # NER dictionary in spacy
        nlp = spacy.load('en_core_web_lg')
        ner_dict = {}
        for i in all_corpus:
            doc = nlp(i)
            for e in doc:
                if e.ent_type_ != "":
                    if len(e) > 2:
                        ner_dict[e] = e.ent_type_

        norp = []
        org = []
        product = []
        for i, j in ner_dict.items():
            if j == 'NORP' and str(i) not in stop:
                norp.append(str(i).lower())
            elif j == 'ORG' and str(i) not in stop:
                org.append(str(i).lower())
            elif j == 'PRODUCT' and str(i) not in stop:
                product.append(str(i).lower())
        print(e_)
        norps.append(norp)
        orgs.append(org)
        products.append(product)

    return norps, orgs, products


# Filtering words using pretrained word2vec dictionary
def wether_word2vec(docs):
    new_docs = []
    for doc in docs:
        new_doc = []
        for i in doc:
            try:
                test = model[i]
                new_doc.append(i)
            except KeyError:
                pass
        new_docs.append(new_doc)
    return new_docs


def remove_unimport(words, thres):
    result = []
    for i in words:
        try:
            if const[i] < thres:
                result.append(i)
        except KeyError:
            pass
    print('Length of filtered words', len(result))
    return result


# -------------- Set parser to choose data set --------------
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='health_news',
                    help='Pick data to do clustering')
args = parser.parse_args()


# -------------- Load data --------------
df = pd.read_csv('data/{}.csv'.format(args.data))


# -------------- Remove nan rows --------------
news_ind = [e for e, i in enumerate(df['news']) if type(i) != float]
df = df.iloc[news_ind, :]


# -------------- Preprocessing --------------
news = df['news'].apply(lambda x: preprocess_wordlist(x))


# -------------- Get words by each attribute --------------
# Attribute: System
system_words = get_system_words(news)
st_sys = stopwords_v_n(system_words, ['VB', 'v'])
# Attribute: Technology
tech_words = get_tech_words(news)
st_tech = stopwords_v_n(tech_words, ['NN', 'n'])
# Attribute: People
st_norp, st_org, st_product = people_ner(news)
# Save results
# with open('data/all_words.pkl', 'wb') as fw:
#     pickle.dump([st_sys, st_tech, st_norp, st_org, st_product], fw)
# with open('data/all_words.pkl', 'rb') as fr:
#     all_words = pickle.load(fr)


# -------------- Remove word if it is not in word2vec dictionary --------------
# Pretrained google news vectors can be abtainable on the Internet
word2vec_loc = 'C:/GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(fname=word2vec_loc,
                                                        binary=True)
new_sys = wether_word2vec(st_sys)
new_tech = wether_word2vec(st_tech)
new_norp = wether_word2vec(st_norp)
new_org = wether_word2vec(st_org)
new_prod = wether_word2vec(st_product)
# new_sys = wether_word2vec(all_words[0])
# new_tech = wether_word2vec(all_words[1])
# new_norp = wether_word2vec(all_words[2])
# new_org = wether_word2vec(all_words[3])
# new_prod = wether_word2vec(all_words[4])


# -------------- Build network to calculate structural holes --------------
stop = stopwords.words('english')
stop.extend(['healthcare', 'say', 'be', 'health',
             'company', 'patient', 'others', 'help', 'a', '.'])
words_net = []
for a, b, c, d, e in zip(new_sys, new_tech,
                         new_norp, new_org, new_prod):
    k = a + b + c + d + e
    words_net.append(k)

net_df = build_network(words_net)
net_df.to_csv('data/fintech_net_df.csv')


# -------------- Remove unimportant words by constraints score --------------
# Constraints were calculated by using igraph in R
"""
Calculating constraints in networkx is too slow.
This is because it calculates on python, not C.
The igraph in R is super faster than networkx.
"""
const = pd.read_csv('data/constraint_fintech.csv', encoding='cp949')
const.columns = ['words', 'const']
const = {k: v for k, v in zip(const['words'], const['const'])}
# Unique words
new_sys = unify_docs(new_sys)
new_tech = unify_docs(new_tech)
new_norp = unify_docs(new_norp)
new_org = unify_docs(new_org)
new_prod = unify_docs(new_prod)

# Filtering
filt_sys = remove_unimport(new_sys, 0.06)
filt_tech = remove_unimport(new_tech, 0.04)
filt_norp = remove_unimport(new_norp, 0.05)
filt_org = remove_unimport(new_org, 0.035)
filt_product = remove_unimport(new_prod, 0.05)

# Save results
with open('data/morph_attr_{}.pkl'.format(args.data), 'wb') as fw:
    pickle.dump([filt_sys, filt_tech, filt_norp, filt_org, filt_product], fw)
