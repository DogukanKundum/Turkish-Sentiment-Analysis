import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

pd.options.mode.chained_assignment = None
import numpy as np
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.models.word2vec import Word2Vec  # the word2vec model gensim class

LabeledSentence = gensim.models.doc2vec.LabeledSentence  # we'll talk about this down below

from tqdm import tqdm

tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer  # a tweet tokenizer from nltk.

tokenizer = TweetTokenizer()

# sklearn
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

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
    data = pd.read_csv("data/tweets_vodafone_result.csv", header=0, delimiter="•", quoting=3, encoding = 'utf8')
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


def postprocess(data, n=1000000):
    data = data.head(n)
    data['tokens'] = data['TWEETS'].progress_map(
        tokenize)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
    data['zemberekTokens'] = data['TWEETS'].progress_map(tokenizeWithZemberek)
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data


data = postprocess(data)

x_train, x_test, y_train, y_test = train_test_split(np.array(data.tokens),
                                                    np.array(data.RESULT), test_size=0.2)

x_train_zemberek, x_test_zemberek, y_train_zemberek, y_test_zemberek = train_test_split(np.array(data.zemberekTokens),
                                                                                        np.array(data.RESULT),
                                                                                        test_size=0.2)


def labelizeTweets(tweets, label_type):
    labelized = []
    for i, v in tqdm(enumerate(tweets)):
        label = '%s_%s' % (label_type, i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized


x_train = labelizeTweets(x_train, 'TRAIN')
x_test = labelizeTweets(x_test, 'TEST')

x_train_zemberek = labelizeTweets(x_train_zemberek, 'TRAIN')
x_test_zemberek = labelizeTweets(x_test_zemberek, 'TEST')

n_dims = [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
n_dim = 75
tweet_w2v = Word2Vec(size=n_dim, min_count=3, hs=1, window=7, iter=75, sg=0)
tweet_w2v.build_vocab([x.words for x in tqdm(x_train)])
tweet_w2v.train([x.words for x in tqdm(x_train)], total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter)

tweet_w2v_zemberek = Word2Vec(size=n_dim, min_count=3, hs=1, window=7, iter=75, sg=0)
tweet_w2v_zemberek.build_vocab([x.words for x in tqdm(x_train_zemberek)])
tweet_w2v_zemberek.train([x.words for x in tqdm(x_train_zemberek)], total_examples=tweet_w2v_zemberek.corpus_count,
                         epochs=tweet_w2v_zemberek.iter)

tweet_w2v_sg = Word2Vec(size=n_dim, min_count=3, hs=1, window=7, iter=75, sg=1)
tweet_w2v_sg.build_vocab([x.words for x in tqdm(x_train)])
tweet_w2v_sg.train([x.words for x in tqdm(x_train)], total_examples=tweet_w2v_sg.corpus_count, epochs=tweet_w2v_sg.iter)

tweet_w2v_zemberek_sg = Word2Vec(size=n_dim, min_count=3, hs=1, window=7, iter=75, sg=1)
tweet_w2v_zemberek_sg.build_vocab([x.words for x in tqdm(x_train_zemberek)])
tweet_w2v_zemberek_sg.train([x.words for x in tqdm(x_train_zemberek)],
                            total_examples=tweet_w2v_zemberek_sg.corpus_count, epochs=tweet_w2v_zemberek_sg.iter)

print('building tf-idf matrix ...')
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform([x.words for x in x_train])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print('vocab size :', len(tfidf))

helmotzDic = {}
helmotzZemberekDic = {}

helmotzDocumentDics = []
helmotzZemberekDocumentDics = []

for index, row in data.iterrows():
    dic = {}
    zemberekDic = {}
    for token in row['tokens']:
        if token in dic:
            dic[token] = dic[token] + 1
        else:
            dic[token] = 1

        if token in helmotzDic:
            helmotzDic[token] = helmotzDic[token] + 1
        else:
            helmotzDic[token] = 1

    for token in row['zemberekTokens']:
        if token in zemberekDic:
            zemberekDic[token] = zemberekDic[token] + 1
        else:
            zemberekDic[token] = 1

        if token in helmotzZemberekDic:
            helmotzZemberekDic[token] = helmotzZemberekDic[token] + 1
        else:
            helmotzZemberekDic[token] = 1

    helmotzDocumentDics.append(dic)
    helmotzZemberekDocumentDics.append(zemberekDic)

# In[11]:


print(helmotzDic)


def buildWordVector(tokens, size, multiplyBy, isZemberek, isSg):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            if multiplyBy == "helmotz":
                if isZemberek:
                    if isSg:
                        vec += tweet_w2v_zemberek_sg[word].reshape((1, size)) * tfidf[word]
                    else:
                        vec += tweet_w2v_zemberek[word].reshape((1, size)) * tfidf[word]
                else:
                    if isSg:
                        vec += tweet_w2v_sg[word].reshape((1, size)) * helmotzDic[word]
                    else:
                        vec += tweet_w2v[word].reshape((1, size)) * helmotzDic[word]
            else:
                if isZemberek:
                    if isSg:
                        vec += tweet_w2v_zemberek_sg[word].reshape((1, size))
                    else:
                        vec += tweet_w2v_zemberek[word].reshape((1, size))
                else:
                    if isSg:
                        vec += tweet_w2v_zemberek_sg[word].reshape((1, size))
                    else:
                        vec += tweet_w2v_sg[word].reshape((1, size))

            count += 1.
        except KeyError:  # handling the case where the token is not
            # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec


# In[28]:


# without helmotz
train_vecs_w2v = np.concatenate(
    [buildWordVector(z, n_dim, "none", False, False) for z in tqdm(map(lambda x: x.words, x_train))])
train_vecs_w2v = scale(train_vecs_w2v)

test_vecs_w2v = np.concatenate(
    [buildWordVector(z, n_dim, "none", False, False) for z in tqdm(map(lambda x: x.words, x_test))])
test_vecs_w2v = scale(test_vecs_w2v)

# without helmotz but with zemberek
train_vecs_w2v_zemberek = np.concatenate(
    [buildWordVector(z, n_dim, "none", True, False) for z in tqdm(map(lambda x: x.words, x_train_zemberek))])
train_vecs_w2v_zemberek = scale(train_vecs_w2v_zemberek)

test_vecs_w2v_zemberek = np.concatenate(
    [buildWordVector(z, n_dim, "none", True, False) for z in tqdm(map(lambda x: x.words, x_test_zemberek))])
test_vecs_w2v_zemberek = scale(test_vecs_w2v_zemberek)

# with helmotz
train_vecs_w2v_helmotz = np.concatenate(
    [buildWordVector(z, n_dim, "helmotz", False, False) for z in tqdm(map(lambda x: x.words, x_train))])
train_vecs_w2v_helmotz = scale(train_vecs_w2v_helmotz)

test_vecs_w2v_helmotz = np.concatenate(
    [buildWordVector(z, n_dim, "helmotz", False, False) for z in tqdm(map(lambda x: x.words, x_test))])
test_vecs_w2v_helmotz = scale(test_vecs_w2v_helmotz)

# with helmotz and zemberek
train_vecs_w2v_zemberek_helmotz = np.concatenate(
    [buildWordVector(z, n_dim, "helmotz", True, False) for z in tqdm(map(lambda x: x.words, x_train_zemberek))])
train_vecs_w2v_zemberek_helmotz = scale(train_vecs_w2v_zemberek_helmotz)

test_vecs_w2v_zemberek_helmotz = np.concatenate(
    [buildWordVector(z, n_dim, "helmotz", True, False) for z in tqdm(map(lambda x: x.words, x_test_zemberek))])
test_vecs_w2v_zemberek_helmotz = scale(test_vecs_w2v_zemberek_helmotz)

# without helmotz sg
train_vecs_w2v_sg = np.concatenate(
    [buildWordVector(z, n_dim, "none", False, True) for z in tqdm(map(lambda x: x.words, x_train))])
train_vecs_w2v_sg = scale(train_vecs_w2v_sg)

test_vecs_w2v_sg = np.concatenate(
    [buildWordVector(z, n_dim, "none", False, True) for z in tqdm(map(lambda x: x.words, x_test))])
test_vecs_w2v_sg = scale(test_vecs_w2v_sg)

# without helmotz but with zemberek and sg
train_vecs_w2v_zemberek_sg = np.concatenate(
    [buildWordVector(z, n_dim, "none", True, True) for z in tqdm(map(lambda x: x.words, x_train_zemberek))])
train_vecs_w2v_zemberek_sg = scale(train_vecs_w2v_zemberek_sg)

test_vecs_w2v_zemberek_sg = np.concatenate(
    [buildWordVector(z, n_dim, "none", True, True) for z in tqdm(map(lambda x: x.words, x_test_zemberek))])
test_vecs_w2v_zemberek_sg = scale(test_vecs_w2v_zemberek_sg)

# with helmotz and sg
train_vecs_w2v_helmotz_sg = np.concatenate(
    [buildWordVector(z, n_dim, "helmotz", False, True) for z in tqdm(map(lambda x: x.words, x_train))])
train_vecs_w2v_helmotz_sg = scale(train_vecs_w2v_helmotz_sg)

test_vecs_w2v_helmotz_sg = np.concatenate(
    [buildWordVector(z, n_dim, "helmotz", False, True) for z in tqdm(map(lambda x: x.words, x_test))])
test_vecs_w2v_helmotz_sg = scale(test_vecs_w2v_helmotz_sg)

# with helmotz,zemberek and sg
train_vecs_w2v_zemberek_helmotz_sg = np.concatenate(
    [buildWordVector(z, n_dim, "helmotz", True, True) for z in tqdm(map(lambda x: x.words, x_train_zemberek))])
train_vecs_w2v_zemberek_helmotz_sg = scale(train_vecs_w2v_zemberek_helmotz_sg)

test_vecs_w2v_zemberek_helmotz_sg = np.concatenate(
    [buildWordVector(z, n_dim, "helmotz", True, True) for z in tqdm(map(lambda x: x.words, x_test_zemberek))])
test_vecs_w2v_zemberek_helmotz_sg = scale(test_vecs_w2v_zemberek_helmotz_sg)

# In[29]:


classifiers = ["SVM", "NaiveBayes", "DecisionTree", "MLP", "KNN"]
helmotz = [0, 1]  # without weighting, helmotz
zembereks = [0, 1]
sgs = [0, 1]

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

for sg in sgs:
    for hm in helmotz:
        for zemberek in zembereks:
            for classifier in classifiers:
                cla = None
                if classifier == "SVM":
                    cla = svm.SVC()
                if classifier == "NaiveBayes":
                    cla = GaussianNB()
                if classifier == "DecisionTree":
                    cla = DecisionTreeClassifier()
                if classifier == "MLP":
                    cla = MLPClassifier()
                if classifier == "KNN":
                    cla = KNeighborsClassifier()
                if sg == 0:
                    if hm == 0 and zemberek == 0:
                        cla.fit(train_vecs_w2v, y_train.astype('int'))
                        y_pred = cla.predict(test_vecs_w2v)
                        score = accuracy_score(y_test.astype('int'), y_pred)
                        reports = classification_report(y_test.astype('int'), y_pred)
                        matrix = confusion_matrix(y_test.astype('int'), y_pred)
                    elif hm == 0 and zemberek == 1:
                        cla.fit(train_vecs_w2v_zemberek, y_train_zemberek.astype('int'))
                        y_pred = cla.predict(test_vecs_w2v_zemberek)
                        score = accuracy_score(y_test.astype('int'), y_pred)
                        reports = classification_report(y_test.astype('int'), y_pred)
                        matrix = confusion_matrix(y_test.astype('int'), y_pred)
                    elif hm == 1 and zemberek == 0:
                        cla.fit(train_vecs_w2v_helmotz, y_train.astype('int'))
                        y_pred = cla.predict(test_vecs_w2v_helmotz)
                        score = accuracy_score(y_test.astype('int'), y_pred)
                        reports = classification_report(y_test.astype('int'), y_pred)
                        matrix = confusion_matrix(y_test.astype('int'), y_pred)
                    elif hm == 1 and zemberek == 1:
                        cla.fit(train_vecs_w2v_zemberek_helmotz, y_train_zemberek.astype('int'))
                        y_pred = cla.predict(test_vecs_w2v_zemberek_helmotz)
                        score = accuracy_score(y_test.astype('int'), y_pred)
                        reports = classification_report(y_test.astype('int'), y_pred)
                        matrix = confusion_matrix(y_test.astype('int'), y_pred)
                if sg == 1:
                    if hm == 0 and zemberek == 0:
                        cla.fit(train_vecs_w2v_sg, y_train.astype('int'))
                        y_pred = cla.predict(test_vecs_w2v_sg)
                        score = accuracy_score(y_test.astype('int'), y_pred)
                        reports = classification_report(y_test.astype('int'), y_pred)
                        matrix = confusion_matrix(y_test.astype('int'), y_pred)
                    elif hm == 0 and zemberek == 1:
                        cla.fit(train_vecs_w2v_zemberek_sg, y_train_zemberek.astype('int'))
                        y_pred = cla.predict(test_vecs_w2v_zemberek_sg)
                        score = accuracy_score(y_test.astype('int'), y_pred)
                        reports = classification_report(y_test.astype('int'), y_pred)
                        matrix = confusion_matrix(y_test.astype('int'), y_pred)
                    elif hm == 1 and zemberek == 0:
                        cla.fit(train_vecs_w2v_helmotz_sg, y_train.astype('int'))
                        y_pred = cla.predict(test_vecs_w2v_helmotz_sg)
                        score = accuracy_score(y_test.astype('int'), y_pred)
                        reports = classification_report(y_test.astype('int'), y_pred)
                        matrix = confusion_matrix(y_test.astype('int'), y_pred)
                    elif hm == 1 and zemberek == 1:
                        cla.fit(train_vecs_w2v_zemberek_helmotz_sg, y_train_zemberek.astype('int'))
                        y_pred = cla.predict(test_vecs_w2v_zemberek_helmotz_sg)
                        score = accuracy_score(y_test.astype('int'), y_pred)
                        reports = classification_report(y_test.astype('int'), y_pred)
                        matrix = confusion_matrix(y_test.astype('int'), y_pred)
                print(classifier, " (helmotz=", hm, ") ( sg =", sg, " ) ( zemberek=", zemberek, "):", score)
                print(classifier, " (helmotz=", hm, ") ( sg =", sg, " ) ( zemberek=", zemberek, "):", reports)
                print(classifier, " (helmotz=", hm, ") ( sg =", sg, " ) ( zemberek=", zemberek, "):", matrix)
                print("--------------------------")
