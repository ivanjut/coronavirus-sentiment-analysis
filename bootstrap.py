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


#### Create training and testing tweet vectors
from sklearn.preprocessing import scale
train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, X_train))])
train_vecs_w2v = scale(train_vecs_w2v)

test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, X_test))])
test_vecs_w2v = scale(test_vecs_w2v)


#### Bootstrap rest of data
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.metrics import AUC

print("Loading in full training data...")
training_data = pd.read_csv('../filtered_tweets.csv')
training_data = training_data.head(1000)
print("Tokenizing...")
training_data['tokens'] = training_data['text'].progress_map(tokenize)
training_data = training_data[training_data.tokens != 'N/A']
training_data.reset_index(inplace=True)
training_data.drop('index', inplace=True, axis=1)

X_train_full = labelizeTweets(np.array(training_data['tokens']), 'TRAIN')

train_vecs = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, X_train_full))])
train_vecs = scale(train_vecs)

print("Bootstrapping...")
model = None
num_bootstrapping_epochs = 3
done = False
i = 0
bad_indices = np.arange(len(training_data))
jokes_indices_list = np.array([])
serious_indices_list = np.array([])
# for i in range(num_bootstrapping_epochs):
while not done:

    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=n_dim))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=[AUC()])

    model.fit(train_vecs_w2v, y_train, epochs=10, batch_size=32, verbose=2)

    score = model.evaluate(test_vecs_w2v, y_test, batch_size=128, verbose=2)
    print("Evaluation accuracy: ", score[1])

    predictions = model.predict(train_vecs)
    pred = predictions.flatten()

    # print(pred)
    # joke_indices = np.where(pred <= 0.35)
    # serious_indices = np.where(pred >= 0.65)
    # bad_indices = np.where((0.35 < pred) & (pred < 0.65))

    pred_df = pd.DataFrame(data=pred, columns=['pred'])
    pred_df = pred_df.iloc[bad_indices]
    print(pred_df)
    serious_indices = pred_df.nlargest(n=100, columns=['pred']).index.values
    print(serious_indices)
    joke_indices = pred_df.nsmallest(n=100, columns=['pred']).index.values


    """

    new_preds = pred[bad_indices]
    # joke_indices = np.argpartition(pred, 100)[:100]
    jokes_indices_list = np.concatenate((jokes_indices_list, joke_indices))
    # serious_indices = np.argpartition(pred, -100)[-100:]
    serious_indices_list = np.concatenate((serious_indices_list, serious_indices))
    bad_indices = np.delete(bad_indices, np.concatenate((joke_indices, serious_indices)))
    print("NUMBER OF BAD INDICES: ", len(bad_indices))
    
    if len(bad_indices) == 0:
        done = True

    # print(joke_indices)
    # print(serious_indices)
    # print(bad_indices)
    pred[jokes_indices_list.astype(int)] = 0
    pred[serious_indices_list.astype(int)] = 1
    pred[bad_indices] = -1
    print("TOTAL INDICES SET: ", len(np.concatenate((jokes_indices_list.astype(int), serious_indices_list.astype(int)))))
    print("TOTAL TRAINING DATA LENGTH: ", len(training_data))

    if i!=0:
        print(training_data['Sentiment'].value_counts())
    # training_data['Sentiment'] = pred
    """
    training_data.loc[joke_indices, "Sentiment"] = 0
    training_data.loc[serious_indices, "Sentiment"] = 1
    print(training_data['Sentiment'].value_counts())
    print(training_data[['text', 'Sentiment']])

    # Update training data
    print("Updating new training data...")
    predicted_tweets = training_data[training_data['Sentiment'] != -1]
    X_train = labelizeTweets(np.array(predicted_tweets['tokens']), 'TRAIN')
    y_train = np.array(predicted_tweets['Sentiment'])
    train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, X_train))])
    train_vecs_w2v = scale(train_vecs_w2v)

    print("Completed epoch {}.".format(i))
    i += 1
    print("***********")

    break