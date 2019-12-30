#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re

import tweepy
from textblob import TextBlob
from tweepy import OAuthHandler


class TwitterClient(object):
    def __init__(self):
        consumer_key = 'x'
        consumer_secret = 'x'
        access_token = x
        access_token_secret =x
        try:
            # create OAuthHandler object
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            # set access token and secret
            self.auth.set_access_token(access_token, access_token_secret)
            # create tweepy API object to fetch tweets
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")

    def get_tweets(self, query, count):
        tweets = []
        try:
            fetched_tweets = self.api.search(q=query, count=count)
            for tweet in fetched_tweets:
                parsed_tweet = {}
                parsed_tweet['text'] = tweet.text
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)
            if tweet.retweet_count > 0:
                if parsed_tweet not in tweets:
                    tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)
            return tweets
        except tweepy.TweepError as e:
            print("Error: " + str(e))

    def clean_tweet(self, tweet):

        return ' '.join(re.sub("(@[A-Za-z0–9]+) | (\w +:\S +)", " ", tweet).split())

    def get_tweet_sentiment(self, tweet):

        analysis = TextBlob(self.clean_tweet(tweet))

        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'


def main():
    # TwitterClient Class'ı yaratma
    api = TwitterClient()

    tweets = api.get_tweets(query='galatasaray', count=200)
    # pozitif tweetleri toplama
    ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
    # pozitif tweetlerin yüzdesi
    print("Positive tweets percentage: {} % ".format(100 * len(ptweets) / len(tweets)))
    # picking negative tweets from tweets
    ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
    # Negatif tweetlerin yüzdesi
    print("Negative tweets percentage: {} % ".format(100 * len(ntweets) / len(tweets)))
    # Nötr tweetlerin yüzdesi
    otweets = [tweet for tweet in tweets if tweet['sentiment'] == 'neutral']
    print("Neutral tweets percentage: {} % ".format(100 * (len(tweets) - len(ntweets) - len(ptweets)) / len(tweets)))
    # İlk 5 pozitif tweet
    print("\n\nPositive tweets: ")
    for tweet in ptweets[:1]:
        print(tweet['text'])
    # İlk 5 negatif tweet
    print("\n\nNegative tweets: ")
    for tweet in ntweets[:1]:
        print(tweet['text'])
    for tweet in otweets[:1]:
        print(tweet['text'])


if __name__ == "__main__":
    main()
