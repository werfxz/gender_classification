#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 18:01:20 2018

@author: furkan cetin
"""
import os
from xml.etree import ElementTree
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import numpy as np 

#full path of the tweet files
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path += "/en/"

#reading all xml files and saving user tweets to a dictionary
#dictionary key: document name     value: user tweets each one is seperated by new line
def read_user_tweets(dir_path):
    """
    This function reads all xml files in given full directory path
    Tokenize the tweets and also proprocess the tweets
    Returns all user tweets by a dictionary as key:username value:tweets
    """
    tweet_dict = {}
    words = []
    tokenize_dict = {}
    user_tweets = ""
    i = 0
    cachedStopWords = stopwords.words("english")
#    print(cachedStopWords) #print stop words
#    loop over the user files
    for filename in os.listdir(dir_path):
        #skip files if it's not xml               
        if filename.endswith(".xml"):                            
            dom = ElementTree.parse(dir_path+filename)         
            tweets = dom.find('documents')
            #loop over tweet of one user                      
            for tweet in tweets:
                #concantanate tweets of one user by new line                                 
                user_tweets = user_tweets + "\n" + (tweet.text).lower()
            #remove punctiation and numbers
            user_tweets = re.sub(r'[^\w\s]','', user_tweets)
            user_tweets = re.sub(r'[0-9]','', user_tweets)
            #cut '.xml' from file name to get user value as the same as in txt file
            filename = filename[:-4]
            #lowercase the text
            tweet_dict[filename] = user_tweets.lower()
            #tokenize user tweets
            tokenize = word_tokenize(user_tweets)
            tokenize = [word for word in tokenize if not (word.startswith('http') or word.startswith('amp') or word.startswith('xx')) ]
            tokenize_dict[filename] = tokenize
            i += 1
            if i % 100 == 0:
                print(i)
            words += [word for word in tokenize_dict[filename] if word not in cachedStopWords]
            user_tweets = ""
    
    return tweet_dict, words

def read_truth_gender(dir_path):
    """
    This function open the txt file in the dir_path directory and
    extrats real gender of users. Returns all gender of username by dictionary
    key:username   value:gender
    """
    
    file = open(dir_path+"truth.txt", "r")
    lines = file.readlines()
    file.close
    
    gender_dict = {}
    for line in lines:
        user, gender, country = line.split(":::")
        gender_dict[user] = gender
    return gender_dict

gender_dict = read_truth_gender(dir_path)
tweet_dict, words= read_user_tweets(dir_path)

#convert dictionaries to pandas dataframe
df_user_tweets = pd.DataFrame.from_dict(tweet_dict, orient="index")
df_user_genders = pd.DataFrame.from_dict(gender_dict, orient="index")

#merge two dictionary to hava features and labels together
df_features = pd.merge(df_user_tweets, df_user_genders, left_index = True, right_index = True)
df_features.rename(columns={'0_x': 'tweets', '0_y': 'gender'}, inplace=True)

#convert male:1 female:0
df_features.loc[df_features['gender']=='male', 'gender'] = 1
df_features.loc[df_features['gender']=='female', 'gender'] = 0

#creating feature vector for tweets
vect = CountVectorizer(max_df=0.7, binary=True, stop_words="english")
X = vect.fit_transform(df_features['tweets'])

#getting gender column from dataframe
Y = df_features.values
Y = Y[:,1]
Y = Y.astype('int')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=2)

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
y_pred_class = logreg.predict(X_test)

print("Accuracy for train/test:", metrics.accuracy_score(Y_test, y_pred_class))

scores = cross_val_score(logreg, X, Y, cv=10, scoring='accuracy')
print("Accuracy for 10fold:", scores.mean())

true_positives = np.count_nonzero(np.multiply(Y_test, y_pred_class ))
predicted_positives = np.count_nonzero(y_pred_class)
actual_positives = np.count_nonzero(Y_test)

precision = true_positives / predicted_positives
recall = true_positives / actual_positives
F1 = 2 *((precision*recall) / (precision+recall))

print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score:", F1)

#freq = nltk.FreqDist(words)
#print(freq.most_common(200))
