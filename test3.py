# coding=utf-8
import os

import nltk


def getTrainData():
    positives, negatives, traindata = [], [], []
    for filename in os.listdir("train"):
        if filename == "positives.txt":
            with open('train/' + filename) as f:
                positives = [(tweet, 'pos') for tweet in f.readlines()]
        if filename == "negatives.txt":
            with open('train/' + filename) as f:
                negatives = [(tweet, 'neg') for tweet in f.readlines()]

    for (words, sentiment) in negatives + positives:
        words_filtered = [e for e in words.split() if len(e) > 2]
        traindata.append((words_filtered, sentiment))

    return traindata


data = getTrainData()


def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words


def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features


word_features = get_word_features(get_words_in_tweets(data))


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features[word.decode("utf8")] = (word in document_words)
    return features


allsetlength = len(data)
training_set = nltk.classify.apply_features(extract_features, data[:allsetlength / 10 * 8])
test_set = data[allsetlength / 10 * 8:]
classifier = nltk.NaiveBayesClassifier.train(training_set)


def classify(tweet):
    print(classifier.classify(extract_features(tweet.split())))


classify("Bugün çok güzel bir gün")
