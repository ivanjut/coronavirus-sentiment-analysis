import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None

from copy import deepcopy
from string import punctuation
from random import shuffle

import gensim
from gensim.models.word2vec import Word2Vec
from heapq import nlargest, nsmallest

TaggedDocument = gensim.models.doc2vec.TaggedDocument # we'll talk about this down below

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer, HashingVectorizer

n_dim = 300


#### Load data
df = pd.read_csv('../all_tweets_with_labels.csv')
# labeled = labeled.loc[~(labeled['Sentiment'] == (-1))]
# labeled.dropna(subset=['Sentiment'], inplace=True)
print('Initial data shape', df.shape)


#### Tokenize all data
def tokenize(tweet, return_string, remove_emojis):
    tweet = str(tweet).encode('ascii', 'ignore').decode('ascii').lower() # remove emojis
    tokens = tokenizer.tokenize(tweet)
    if remove_emojis:
        tokens = list(filter(lambda t: not t.startswith('@'), tokens))
        tokens = list(filter(lambda t: not t.startswith('http'), tokens))
    if return_string:
        return ' '.join(tokens)
    else:
        return tokens

def calc_auc(model, X_test, Y_test):
    try:
        return roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1].T)
    except:
        return -1

def pre_process_df(data, return_string=False, remove_emojis=True, test=False):
    data['tokens'] = data['text'].progress_map(lambda tweet: tokenize(tweet, return_string, remove_emojis))
    data['Sentiment'].fillna(-1, inplace=True)
    data['Sentiment'] = data['Sentiment'].map(float)
    # data = data[(data.tokens != 'N/A') & (data['Sentiment'] != -1)]
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    data.dropna(subset=['tokens','Sentiment'], inplace=True)
    # data.drop(data.columns.difference(['tokens','Sentiment']), 1, inplace=True)
    
    X = np.array(data['tokens'])
    countVectorizer = None
    tfidf = None
    if not test:
        countVectorizer = CountVectorizer()
        # countVectorizer = CountVectorizer(ngram_range=(1, 2))
        X = countVectorizer.fit_transform(np.array(data['tokens']))
        tfidf = TfidfTransformer()
        X = tfidf.fit_transform(X)
        # svd = TruncatedSVD(n_components=100, random_state=42)
        # X = svd.fit(X)
    return data, X, countVectorizer, tfidf

def fit_predict(model, train_indices, training_df, X_test, Y_test, index_to_x, model_name):

    X_train, Y_train = index_to_x[train_indices, :], training_df.loc[train_indices, 'Sentiment'].values

    print('X_train shape: {}'.format(X_train.shape))
    print('Y_train shape: {}'.format(Y_train.shape))
    print('X_test shape: {}'.format(X_test.shape))
    print('Y_test shape: {}'.format(Y_test.shape))

    model.fit(X_train, Y_train)
    score = model.score(X_test, Y_test)
    print('Train AUC: ',  calc_auc(model, X_train, Y_train))
    print('Train accuracy: ',  model.score(X_train, Y_train))
    print("Evaluation AUC: ", calc_auc(model, X_test, Y_test))
    print('Evaluation accuracy: ',  model.score(X_test, Y_test))

    if model_name == 'RC':
        predictions = model.decision_function(index_to_x[training_df.index.values, :])
        print(predictions)
        print(0.5**(1+predictions))
    else:
        predictions = model.predict(index_to_x[training_df.index.values, :])

    pred = predictions.flatten()       # should be a 0d numpy vector of predictions
    return training_df.index.values, pred, model

def most_polarizing_words(model, countVectorizer, n):

    if model.coef_.shape[0] == 1:
        coefficients = [(i,c) for i,c in enumerate(model.coef_[0])]
    else:
        return None
    id2word = countVectorizer.get_feature_names()
    strongest_positive = [(id2word[i],coef) for i,coef in nlargest(n, coefficients, key=lambda x: x[1])]
    strongest_negative = [(id2word[i],coef) for i,coef in nsmallest(n, coefficients, key=lambda x: x[1])]
    print('\n Most positive words were:') 
    for (word,coef) in strongest_positive:
        print('{}: {}'.format(word, coef))
    print('\n Most negative words were:') 
    for (word,coef) in strongest_negative:
        print('{}: {}'.format(word, coef))
    return strongest_positive, strongest_negative

df, index_to_x, countVectorizer, tfidf = pre_process_df(df, return_string=True, remove_emojis=False)

labeled_indices = df[~(df['Sentiment'] == -1)].index.values
eval_indices = np.random.choice(labeled_indices, size=labeled_indices.shape[0]//2, replace=False)


results = {}
model = MultinomialNB()
model_name = "Multinomial Naive Bayes"


print('\n \n ------------------ {} --------------------------- \n \n'.format(model_name))
print(model)
eval_df, training_df = df.loc[eval_indices, :], df.drop(index=eval_indices)
X_test, Y_test = index_to_x[eval_df.index.values, :], eval_df['Sentiment']
print("Bootstrapping...")

predicted_tweets = training_df[training_df['Sentiment'] != -1]
train_indices = predicted_tweets.index.values

done = False
i = 0
bad_indices = training_df[training_df['Sentiment'] == -1].index.values
jokes_indices_list = np.array([])
serious_indices_list = np.array([])

# print('LR: {}'.format(cross_validate(lr_clf, X_train, Y_train, cv=5, return_train_score=True)))
# print('SVC: {}'.format(cross_validate(svc_clf, X_train, Y_train, cv=5, return_train_score=True)))
# print('NB: {}'.format(cross_validate(clf, X_train, Y_train, cv=5, return_train_score=True)))
# print('RC: {}'.format(cross_validate(ridge_clf, X_train, Y_train, cv=5, return_train_score=True)))

while not done:

    ################ CLASSIFICATION METHOD ##########################
    bad_indices, pred, model = fit_predict(model, train_indices, training_df, X_test, Y_test, index_to_x, model_name)


    ######################################################################
    assert (bad_indices.shape[0] == pred.shape[0])

    bad_indices_dict = {}
    for j,index in enumerate(bad_indices):
        bad_indices_dict[index] = pred[j]

    print('{} bad indices'.format(pred.shape[0]))

    thresh = (i+4)**4
    if len(bad_indices) > thresh:
        joke_indices = np.array(nsmallest(int(np.floor(thresh/2)), bad_indices_dict, key=bad_indices_dict.get))
        serious_indices = np.array(nlargest(int(np.floor(thresh/2)), bad_indices_dict, key=bad_indices_dict.get))
    else:
        joke_indices = np.array([index for index in bad_indices if bad_indices_dict[index] <= 0.5])
        serious_indices = np.array([index for index in bad_indices if bad_indices_dict[index] > 0.5])

    jokes_indices_list = np.concatenate((jokes_indices_list, joke_indices))
    serious_indices_list = np.concatenate((serious_indices_list, serious_indices))
    bad_indices = np.delete(bad_indices, np.where((np.isin(bad_indices, joke_indices)) | (np.isin(bad_indices, serious_indices))))
    
    if len(bad_indices) == 0:
        done = True

    # pred[jokes_indices_list.astype(int)] = 0
    # pred[serious_indices_list.astype(int)] = 1
    # pred[bad_indices] = -1

    training_df.loc[jokes_indices_list.astype(int), 'Sentiment'] = 0
    training_df.loc[serious_indices_list.astype(int), 'Sentiment'] = 1
    # training_df['Sentiment'] = pred
    # print(training_data[['text', 'Sentiment']])

    # Update training data
    print("Updating new training data...")
    predicted_tweets = training_df[training_df['Sentiment'] != -1]

    train_indices = predicted_tweets.index.values

    print("Processed {} tweets.".format(thresh))
    print(len(bad_indices), " unlabeled tweets left.")
    print("Completed epoch {}.".format(i))
    i += 1
    print("***********")

bad_indices, pred, model = fit_predict(model, training_df.index.values, training_df, X_test, Y_test, index_to_x, model_name)

auc = calc_auc(model, X_test, Y_test)
mean_accuracy = model.score(X_test, Y_test)
results = (auc, mean_accuracy)

print("**************************************************")
print("********** MODEL EVALUATION RESULTS **************")
print("**************************************************")
print("ROC-AUC: ", auc)
print("Mean accuracy: ", mean_accuracy)


# Make mew predictions
new_df = pd.read_csv('../filtered_tweets_final.csv')
print("Length data: ", len(new_df))


new_df['Sentiment'] = -1

new_df, _, _, _ = pre_process_df(new_df, return_string=True, remove_emojis=True, test=True)
print("Finished pre-processing.")

X = tfidf.transform(countVectorizer.transform(new_df['tokens'].values))

predictions = model.predict(X)
probs = model.predict_proba(X)
print(probs.shape)
print(probs[:5, :])

print("Length: ", len(predictions))
print(len(predictions) == len(new_df))

new_df['Sentiment'] = predictions
new_df['sentiment_probs'] = probs[:, 1] 

print(new_df[['text', 'Sentiment', 'sentiment_probs']].head(20))
print(new_df['Sentiment'].value_counts())

new_df.to_csv('../predictions.csv')