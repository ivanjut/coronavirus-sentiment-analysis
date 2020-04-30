import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

from copy import deepcopy
from string import punctuation
from random import shuffle

import gensim
from gensim.models.word2vec import Word2Vec
TaggedDocument = gensim.models.doc2vec.TaggedDocument # we'll talk about this down below

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

n_dim = 300


#### Load data
labeled = pd.read_csv('../labeled_tweets.csv')
labeled = labeled.head(1000)


#### Tokenize all data
def tokenize(tweet):
    try:
        tweet = str(tweet.encode('ascii', 'ignore').decode('ascii').lower()) # remove emojis
        tokens = tokenizer.tokenize(tweet)
        tokens = list(filter(lambda t: not t.startswith('@'), tokens))
        tokens = list(filter(lambda t: not t.startswith('http'), tokens))
        return tokens
    except:
        return 'N/A'

def postprocess(data):
    data['tokens'] = data['text'].progress_map(tokenize)
    data['Sentiment'] = data['Sentiment (1 serious, 0 joking, -1 unsure)'].map(float)
    data = data[(data.tokens != 'N/A') & (data['Sentiment'] != -1)]
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data

processed = postprocess(labeled)


#### Train word embeddings
X_train, X_test, y_train, y_test = train_test_split(np.array(processed['tokens']), np.array(processed['Sentiment']), test_size=0.3)

def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in tqdm(enumerate(tweets)):
        label = '%s_%s'%(label_type,i)
        labelized.append(TaggedDocument(v, [label]))
    return labelized

X_train = labelizeTweets(X_train, 'TRAIN')
X_test = labelizeTweets(X_test, 'TEST')

tweet_w2v = Word2Vec(size=n_dim, min_count=1)
tweet_w2v.build_vocab([x.words for x in tqdm(X_train)])
tweet_w2v.train([x.words for x in tqdm(X_train)], total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.epochs)

# print(tweet_w2v.wv.vocab)
# print("VIRUS EMBEDDING: ", tweet_w2v['virus'])
# print("Virus most similar words: ", tweet_w2v.wv.most_similar('virus'))


#### Tweet embeddings
print('building tf-idf matrix ...')
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform([x.words for x in X_train])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print('vocab size :', len(tfidf))

def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec

from sklearn.preprocessing import scale
train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, X_train))])
train_vecs_w2v = scale(train_vecs_w2v)

test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, X_test))])
test_vecs_w2v = scale(test_vecs_w2v)

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=n_dim))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_vecs_w2v, y_train, epochs=10, batch_size=32, verbose=2)

score = model.evaluate(test_vecs_w2v, y_test, batch_size=128, verbose=2)
print(score[1])