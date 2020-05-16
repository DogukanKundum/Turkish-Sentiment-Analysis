import pandas as pd  # provide sql-like data manipulation tools. very handy.

pd.options.mode.chained_assignment = None
import numpy as np  # high dimensional vector computing library.
from copy import deepcopy
from string import punctuation
from random import shuffle
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.models.word2vec import Word2Vec  # the word2vec model gensim class
from sklearn import svm
from sklearn import tree

LabeledSentence = gensim.models.doc2vec.LabeledSentence  # we'll talk about this down below

from tqdm import tqdm

tqdm.pandas(desc="progress-bar")

import os
import datetime
import codecs
import sys

from nltk.tokenize import TweetTokenizer  # a tweet tokenizer from nltk.

tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook

sg_flag = 0
tfidfflag = 1

# zemberek
import jpype

jpype.startJVM(jpype.getDefaultJVMPath(),
               "-Djava.class.path=zemberek_jar/zemberek-tum-2.0.jar", "-ea")
# Türkiye Türkçesine göre çözümlemek için gerekli sınıfı hazırla
Tr = jpype.JClass("net.zemberek.tr.yapi.TurkiyeTurkcesi")
# tr nesnesini oluştur
tr = Tr()
# Zemberek sınıfını yükle
Zemberek = jpype.JClass("net.zemberek.erisim.Zemberek")
# zemberek nesnesini oluştur
zemberek = Zemberek(tr)


def ingest():
    data = pd.read_csv("data/tweets_turktelekom_result.csv", header=0, delimiter="•", quoting=3, encoding='utf8')
    data = data[data.TWEETS.isnull() == False]
    data['RESULT'] = data['RESULT'].map(int)
    data = data[data['TWEETS'].isnull() == False]
    # data.reset_index(inplace=True)
    # data.drop('index', axis=1, inplace=True)
    print('dataset loaded with shape', data.shape)
    return data


data = ingest()


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


def postprocess(data, n=1000000):
    data = data.head(n)
    data['tokens'] = data['TWEETS'].progress_map(
        tokenize)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data


data = postprocess(data)

x_train, x_test, y_train, y_test = train_test_split(np.array(data.head(1578628).tokens),
                                                    np.array(data.head(1578628).RESULT), test_size=0.2)


def labelizeTweets(tweets, label_type):
    labelized = []
    for i, v in tqdm(enumerate(tweets)):
        label = '%s_%s' % (label_type, i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized


x_train = labelizeTweets(x_train, 'TRAIN')
x_test = labelizeTweets(x_test, 'TEST')

print("len", len(x_train))

tweet_w2v = Word2Vec(size=200, min_count=3, hs=1, window=7, sg=sg_flag)
tweet_w2v.build_vocab([x.words for x in tqdm(x_train)])
tweet_w2v.train([x.words for x in x_train], total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter)
print("build vocab done!")
from IPython.display import display, Image

output_notebook()
plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="A map of 10000 word vectors",
                       tools="pan,wheel_zoom,box_zoom,reset,hover,save",
                       x_axis_type=None, y_axis_type=None, min_border=1)

# getting a list of word vectors. limit to 10000. each is of 200 dimensions
word_vectors = [tweet_w2v[w] for w in list(tweet_w2v.wv.vocab.keys())[:5000]]

# dimensionality reduction. converting the vectors to 2d vectors
from sklearn.manifold import TSNE

tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
tsne_w2v = tsne_model.fit_transform(word_vectors)

# putting everything in a dataframe
tsne_df = pd.DataFrame(tsne_w2v, columns=['x', 'y'])
tsne_df['words'] = list(tweet_w2v.wv.vocab.keys())[:5000]

# plotting. the corresponding word appears when you hover on the data point.
plot_tfidf.scatter(x='x', y='y', source=tsne_df)
hover = plot_tfidf.select(dict(type=HoverTool))
hover.tooltips = {"word": "@words"}
show(plot_tfidf)

print('building tf-idf matrix ...')
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform([x.words for x in x_train])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print('vocab size :', len(tfidf))


def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            if (tfidfflag == 1):
                vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            else:
                vec += tweet_w2v[word].reshape((1, size))  # * tfidf[word]
            count += 1.
        except KeyError:  # handling the case where the token is not
            # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec


from sklearn.preprocessing import scale

train_vecs_w2v = np.concatenate([buildWordVector(z, 200) for z in tqdm(map(lambda x: x.words, x_train))])
train_vecs_w2v = scale(train_vecs_w2v)

test_vecs_w2v = np.concatenate([buildWordVector(z, 200) for z in tqdm(map(lambda x: x.words, x_test))])
test_vecs_w2v = scale(test_vecs_w2v)

# MLP SCORE
from keras.models import Sequential
from keras.layers import Activation, Dense
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)
clf.fit(train_vecs_w2v,y_train)
y_pred=clf.predict(test_vecs_w2v)
print("RF:", accuracy_score(y_test, y_pred))
print("RF:", classification_report(y_test, y_pred))
print("RF", confusion_matrix(y_test, y_pred))

cla = MLPClassifier()
cla.fit(train_vecs_w2v, y_train)
mlp_pred = cla.predict(test_vecs_w2v)
print("MLP:", accuracy_score(y_test, mlp_pred))
print("MLP:", classification_report(y_test, mlp_pred))
print("MLP", confusion_matrix(y_test, mlp_pred))

# NAİVE BAYES SCORE
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(train_vecs_w2v, y_train)
GaussianNB(priors=None)
nbpredict = clf.predict(test_vecs_w2v)
print("BAYES:", accuracy_score(y_test, nbpredict))
print("BAYES:", classification_report(y_test, nbpredict))
print("BAYES", confusion_matrix(y_test, nbpredict))

# KNN SCORE
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
# fitting the model
knn.fit(train_vecs_w2v, y_train)
# predict the response
knnpred = knn.predict(test_vecs_w2v)
print("KNN:", accuracy_score(y_test, knnpred))
print("KNN:", classification_report(y_test, knnpred))
print("KNN", confusion_matrix(y_test, knnpred))

# DESICION TREE SCORE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
clf_gini.fit(train_vecs_w2v, y_train)
dtpred = clf_gini.predict(test_vecs_w2v)
print("DESICION:", accuracy_score(y_test, dtpred))
print("DESICION:", classification_report(y_test, dtpred))
print("DESICION", confusion_matrix(y_test, dtpred))

# SVM SCORE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

log_model = LogisticRegression()
log_model = log_model.fit(X=train_vecs_w2v, y=y_train)
y_pred = log_model.predict(test_vecs_w2v)
print("SVM:", accuracy_score(y_test, y_pred))
print("SVM:", classification_report(y_test, y_pred))
print("SVM", confusion_matrix(y_test, y_pred))
