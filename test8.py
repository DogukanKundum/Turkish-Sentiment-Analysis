#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import warnings

import jpype
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
tqdm.pandas(desc="progress-bar")
from nltk.tokenize import TweetTokenizer  # a tweet tokenizer from nltk.

tokenizer = TweetTokenizer()
column = ['TWEETS']
df = pd.read_csv("olumlu/tweets.csv", header=0, delimiter="\t", quoting=1)
df.columns = column
turkish_characters = "a|b|c|ç|d|e|f|g|ğ|h|ı|i|j|k|l|m|n|o|ö|p|r|s|ş|t|u|ü|v|y|z|0-9"


def remove_stopwords(df_fon):
    stopwords = open('data/stop_word', 'r').read().split()
    df_fon['stopwords_removed'] = list(map(lambda doc:
                                           [word for word in doc if word not in stopwords], df_fon['TWEETS']))


remove_stopwords(df)


def startJVM():
    jpype.startJVM(jpype.getDefaultJVMPath(),
                   "-Djava.class.path=zemberek_jar/zemberek-tum-2.0.jar", "-ea")
    Tr = jpype.JClass("net.zemberek.tr.yapi.TurkiyeTurkcesi")
    tr = Tr()
    Zemberek = jpype.JClass("net.zemberek.erisim.Zemberek")
    zemberek = Zemberek(tr)
    return zemberek


zemberek = startJVM()


def tokenize(tweet):
    try:
        tweet = tweet.lower()
        tokens = tokenizer.tokenize(tweet)
        tokens = [x for x in tokens if not x.startswith('@')]
        tokens = [x for x in tokens if not x.startswith('#')]
        tokens = [x for x in tokens if not x.startswith('http')]
        tokens = [x for x in tokens if not x.startswith('.')]
        tokens = [x for x in tokens if not x.startswith('(')]
        tokens = [x for x in tokens if not x.startswith(')')]
        tokens = [x for x in tokens if not x == ',']
        tokens = [x for x in tokens if not x == '?']
        tokens = [x for x in tokens if not x == '!']
        tokens = [x for x in tokens if not x == ':']
        tokens = [x for x in tokens if not x == ';']
        tokens = [x for x in tokens if not x == '"']
        tokens = [x for x in tokens if not x == "'"]
        return tokens
    except:
        print("Unexpected error:", sys.exc_info()[0])
        return 'NC'


def tokenizeWithZemberek(tweet):
    try:
        tweet = tweet.lower()
        tokens = tokenizer.tokenize(tweet)
        tokens = [x for x in tokens if not x.startswith('@')]
        tokens = [x for x in tokens if not x.startswith('#')]
        tokens = [x for x in tokens if not x.startswith('http')]
        tokens = [x for x in tokens if not x.startswith('.')]
        tokens = [x for x in tokens if not x.startswith('(')]
        tokens = [x for x in tokens if not x.startswith(')')]
        tokens = [x for x in tokens if not x == ',']
        tokens = [x for x in tokens if not x == '?']
        tokens = [x for x in tokens if not x == '!']
        tokens = [x for x in tokens if not x == ':']
        tokens = [x for x in tokens if not x == ';']
        tokens = [x for x in tokens if not x == '"']
        tokens = [x for x in tokens if not x == "'"]
        zemberekTokens = []
        for token in tokens:
            if token.strip() > '':
                zemberekToken = zemberek.kelimeCozumle(token)
                if zemberekToken:
                    zemberekTokens.append(zemberekToken[0].kok().icerik())
                else:
                    zemberekTokens.append(token)

        return zemberekTokens
    except:
        print("Unexpected error:", sys.exc_info()[0])
        return 'NC'


def postprocess(data):
    data['tokens'] = data['TWEETS'].progress_map(tokenize)
    data['zemberekTokens'] = data['TWEETS'].progress_map(tokenizeWithZemberek)
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data


df = postprocess(df)

df.head(10)

df['LABEL'] = 1

df.LABEL.iloc[3685:] = 0

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['TWEETS'], df['LABEL'], random_state=0)
print(X_train.head())
print('\n\nX_train shape: ', X_train.shape)

from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(encoding='iso-8859-9').fit(X_train)

X_train_vectorized = vect.transform(X_train)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

from sklearn.metrics import roc_auc_score

predictions = model.predict(vect.transform(X_test))
print('AUC: ', roc_auc_score(y_test, predictions))

feature_names = np.array(vect.get_feature_names())
sorted_coef_index = model.coef_[0].argsort()
print('Negative: \n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Positive: \n{}\n'.format(feature_names[sorted_coef_index[:-11:-1]]))

from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(min_df=5).fit(X_train)

X_train_vectorized = vect.transform(X_train)
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
predictions = model.predict(vect.transform(X_test))
print('AUC: ', roc_auc_score(y_test, predictions))

feature_names = np.array(vect.get_feature_names())
sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()
print('Minimum Tfidf: \n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))
print('Maximum Tfidf: \n{}\n'.format(feature_names[sorted_tfidf_index[:-11:-1]]))

vect = CountVectorizer(min_df=5, ngram_range=(1, 2)).fit(X_train)
X_train_vectorized = vect.transform(X_train)

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
predictions = model.predict(vect.transform(X_test))
print('AUC: ', roc_auc_score(y_test, predictions))

feature_names = np.array(vect.get_feature_names())
sorted_coef_index = model.coef_[0].argsort()
print('Negative: \n{}\n'.format(feature_names[sorted_coef_index][:10]))
print('Positive Coef: \n{}\n'.format(feature_names[sorted_coef_index][:-11:-1]))


def remove_tweet_stopwords(df_fon):
    stopwords = open('data/stop_word', 'r').read().split()
    df_fon['stopwords_removed'] = list(map(lambda doc:
                                           [word for word in doc if word not in stopwords], df_fon['TWEETS']))


data = pd.read_csv("tweets_turkcell_end.csv", delimiter="•", encoding='utf-8')
# Preview the first 5 lines of the loaded data
remove_tweet_stopwords(data)
data = postprocess(data)
result = []
for text in data['TWEETS']:
    print(result.append((model.predict(vect.transform([text])))))

# print(model.predict(vect.transform([Review])))
