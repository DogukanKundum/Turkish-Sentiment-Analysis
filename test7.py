#!/usr/bin/env python
# -*- coding: utf-8 -*-
from textblob.classifiers import NaiveBayesClassifier

with open('data/train.csv', 'r') as fp:
    cl = NaiveBayesClassifier(fp, format="json")
