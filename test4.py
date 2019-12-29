#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


class TwitterData_Initialize():
    data = []
    processed_data = []
    wordlist = []

    data_model = None
    data_labels = None
    is_testing = False

    def initialize(self, csv_file, is_testing_set=False, from_cached=None):
        if from_cached is not None:
            self.data_model = pd.read_csv(from_cached)
            return

        self.is_testing = is_testing_set

        if not is_testing_set:
            self.data = pd.read_csv(csv_file, header=0, names=["Rating", "Review"])
            self.data = self.data[self.data["Rating"].isin(["1", "0"])]
        self.processed_data = self.data
        self.wordlist = []
        self.data_model = None
        self.data_labels = None


data = TwitterData_Initialize()
data.initialize("data/hepsiburada.csv")
data.processed_data.head(5)
df = data.processed_data


def remove_stopwords(df_fon):
    stopwords = open('data/stop_word', 'r').read().split()
    df_fon['stopwords_removed'] = list(map(lambda doc:
                                           [word for word in doc if word not in stopwords], df_fon['Review']))


remove_stopwords(df)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['Review'], df['Rating'], random_state=0)
print(X_train.head())
print('\n\nX_train shape: ', X_train.shape)

# CountVectorizer'ı başlatıyoruz ve eğitim verilerimize uyguluyoruz.
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(encoding='iso-8859-9').fit(X_train)

# X_train'deki belgeleri bir belge terim matrisine dönüştürürüz
X_train_vectorized = vect.transform(X_train)

# Bu özellik matrisi X_ train_ vectorized'e dayanarak Lojistik Regresyon sınıflandırıcısını eğiteceğiz
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Daha sonra, X_test kullanarak tahminler yapacağız ve eğri puanının altındaki alanı hesaplayacağız.
from sklearn.metrics import roc_auc_score

predictions = model.predict(vect.transform(X_test))
print('AUC: ', roc_auc_score(y_test, predictions))

# Modelimizin bu tahminleri nasıl yaptığını daha iyi anlamak için, her bir özellik için katsayıları (bir kelime),
# pozitifliği ve olumsuzluk açısından ağırlığını belirlemek için kullanabiliriz.
feature_names = np.array(vect.get_feature_names())
sorted_coef_index = model.coef_[0].argsort()
print('Negatif: \n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Pozitif: \n{}\n'.format(feature_names[sorted_coef_index[:-11:-1]]))

# tf-idf vectorizer'ı başlatacağız ve eğitim verilerimize uygulayacağız.
# En az beş dokümanda görünen kelime dağarcığımızdaki kelimeleri kaldıracağımız min_df = 5 değerini belirtiyoruz.
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(min_df=5).fit(X_train)

X_train_vectorized = vect.transform(X_train)
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
predictions = model.predict(vect.transform(X_test))
print('AUC: ', roc_auc_score(y_test, predictions))

feature_names = np.array(vect.get_feature_names())
sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()
print('En küçük Tfidf: \n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))
print('En büyük Tfidf: \n{}\n'.format(feature_names[sorted_tfidf_index[:-11:-1]]))

# bigramlar, bitişik kelimelerin çiftlerini sayar ve bize kötü ve kötü olmayan gibi özellikler verebilir.
# Bu nedenle, minimum 5 belge sıklığını belirten ve 1 gram ve 2 gram ekstrakt eden eğitim setimizi yeniden yerleştiriyoruz.
vect = CountVectorizer(min_df=5, ngram_range=(1, 2)).fit(X_train)
X_train_vectorized = vect.transform(X_train)

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
predictions = model.predict(vect.transform(X_test))
print('AUC: ', roc_auc_score(y_test, predictions))

# Her özelliği kontrol etmek için katsayıları kullanarak görebiliriz
feature_names = np.array(vect.get_feature_names())
sorted_coef_index = model.coef_[0].argsort()
print('Negatif: \n{}\n'.format(feature_names[sorted_coef_index][:10]))
print('Pozitif Coef: \n{}\n'.format(feature_names[sorted_coef_index][:-11:-1]))

while (True):
    yorum = input("Yorumunuz Nedir?(Programdan çıkmak için \'F\' yazınız)")
    if (yorum == 'F' or yorum == 'f'):
        break
    else:
        print(model.predict(vect.transform([yorum])))
