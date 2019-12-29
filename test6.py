#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

data = pd.read_csv("data/train.csv", engine='python')

def remove_stopwords(df_fon):
    stopwords = open('data/stop_word', 'r').read().split()
    df_fon['stopwords_removed'] = list(map(lambda doc:
        [word for word in doc if word not in stopwords], df_fon['Review']))

remove_stopwords(data)

classification_training = [doc for doc in data.iloc[:, 0]]
sentences_training = [doc for doc in data.iloc[:, 1]]

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(analyzer='word', lowercase=True)
sen_train_vector = vectorizer.fit_transform(sentences_training)

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
model = clf.fit(X=sen_train_vector.toarray(), y=classification_training)

sen_test_vector = vectorizer.transform(['güzel'])
# print(sen_test_vector.toarray())
y_pred = model.predict(sen_test_vector.toarray())
print(y_pred)
