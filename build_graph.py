import os
import random
import numpy as np
import pandas as pd
import pickle as pkl
import scipy.sparse as sp
from math import log
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from scipy.spatial.distance import cosine


data = pd.read_csv('../nn_bootstrapped_training_data_tweets.csv')
print(data.head(10))
data = data.head(100)

num_docs = len(data)
print("Number of tweets: ", num_docs)

# Build vocabulary
word_freq = {}
word_set = set()

for _, tokens in data['tokens'].items():
    for word in tokens.split(','):
        word = word.split("'")[1]
        if word in [".", ",", "'", '"', "!", "?", "..", "..."]:
            continue
        word_set.add(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

vocab = list(word_set)
vocab_size = len(vocab)
word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i


term_freq = {}
pmi_dict = {}
for word in word_freq:
    pmi_dict[word] = word_freq[word]/num_docs

for _, row in data.iterrows():
    doc_id = row['id']
    tokens = row['tokens']
    for word in tokens.split(','):
        word = word.split("'")[1]
        if word in [".", ",", "'", '"', "!", "?", "..", "..."]:
            continue
        word_id = word_id_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in term_freq:
            term_freq[doc_word_str] += 1/len(tokens)
        else:
            term_freq[doc_word_str] = 1/len(tokens)

        for next_word in tokens.split(','):
            next_word = next_word.split("'")[1]
            if next_word in [".", ",", "'", '"', "!", "?", "..", "..."]:
                continue
            if next_word != word:
                if (word, next_word) in pmi_dict:
                    pmi_dict[(word, next_word)] += 1/num_docs
                else:
                    pmi_dict[(word, next_word)] = 1/num_docs

for key in pmi_dict:
    if type(key) == tuple:
        pmi_dict[key] = np.log(pmi_dict[key]/(pmi_dict[key[0]] * pmi_dict[key[1]]))


idf_dict = {}
for word in word_freq:
    idf_dict[word] = np.log(num_docs/word_freq[word])


# calculate tf-idf for word-doc pairs
tfidf_dict = {}
for word in vocab:
    word_id = word_id_map[word]
    for _, doc_id in data['id'].items():
        doc_word_str = str(doc_id) + ',' + str(word_id)
        try:
            tfidf_dict[doc_word_str] = term_freq[doc_word_str] * idf_dict[word]
        except:
            tfidf_dict[doc_word_str] = 0

count = 0
for i in tfidf_dict:
    print(i, tfidf_dict[i])
    if tfidf_dict[i] != 0:
        count += 1
    if count > 10:
        break


#         row_tx.append(i)
#         col_tx.append(j)
#         # np.random.uniform(-0.25, 0.25)
#         data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

# # tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
# tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
#                    shape=(test_size, word_embeddings_dim))

# ty = []
# for i in range(test_size):
#     doc_meta = shuffle_doc_name_list[i + train_size]
#     temp = doc_meta.split('\t')
#     label = temp[2]
#     one_hot = [0 for l in range(len(label_list))]
#     label_index = label_list.index(label)
#     one_hot[label_index] = 1
#     ty.append(one_hot)
# ty = np.array(ty)
# print(ty)

# # allx: the the feature vectors of both labeled and unlabeled training instances
# # (a superset of x)
# # unlabeled training instances -> words

# word_vectors = np.random.uniform(-0.01, 0.01,
#                                  (vocab_size, word_embeddings_dim))

# for i in range(len(vocab)):
#     word = vocab[i]
#     if word in word_vector_map:
#         vector = word_vector_map[word]
#         word_vectors[i] = vector

# row_allx = []
# col_allx = []
# data_allx = []

# for i in range(train_size):
#     doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
#     doc_words = shuffle_doc_words_list[i]
#     words = doc_words.split()
#     doc_len = len(words)
#     for word in words:
#         if word in word_vector_map:
#             word_vector = word_vector_map[word]
#             doc_vec = doc_vec + np.array(word_vector)

#     for j in range(word_embeddings_dim):
#         row_allx.append(int(i))
#         col_allx.append(j)
#         # np.random.uniform(-0.25, 0.25)
#         data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len
# for i in range(vocab_size):
#     for j in range(word_embeddings_dim):
#         row_allx.append(int(i + train_size))
#         col_allx.append(j)
#         data_allx.append(word_vectors.item((i, j)))


# row_allx = np.array(row_allx)
# col_allx = np.array(col_allx)
# data_allx = np.array(data_allx)

# allx = sp.csr_matrix(
#     (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

# ally = []
# for i in range(train_size):
#     doc_meta = shuffle_doc_name_list[i]
#     temp = doc_meta.split('\t')
#     label = temp[2]
#     one_hot = [0 for l in range(len(label_list))]
#     label_index = label_list.index(label)
#     one_hot[label_index] = 1
#     ally.append(one_hot)

# for i in range(vocab_size):
#     one_hot = [0 for l in range(len(label_list))]
#     ally.append(one_hot)

# ally = np.array(ally)

# print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

# '''
# Doc word heterogeneous graph
# '''

# # word co-occurence with context windows
# window_size = 20
# windows = []

# for doc_words in shuffle_doc_words_list:
#     words = doc_words.split()
#     length = len(words)
#     if length <= window_size:
#         windows.append(words)
#     else:
#         # print(length, length - window_size + 1)
#         for j in range(length - window_size + 1):
#             window = words[j: j + window_size]
#             windows.append(window)
#             # print(window)


# word_window_freq = {}
# for window in windows:
#     appeared = set()
#     for i in range(len(window)):
#         if window[i] in appeared:
#             continue
#         if window[i] in word_window_freq:
#             word_window_freq[window[i]] += 1
#         else:
#             word_window_freq[window[i]] = 1
#         appeared.add(window[i])

# word_pair_count = {}
# for window in windows:
#     for i in range(1, len(window)):
#         for j in range(0, i):
#             word_i = window[i]
#             word_i_id = word_id_map[word_i]
#             word_j = window[j]
#             word_j_id = word_id_map[word_j]
#             if word_i_id == word_j_id:
#                 continue
#             word_pair_str = str(word_i_id) + ',' + str(word_j_id)
#             if word_pair_str in word_pair_count:
#                 word_pair_count[word_pair_str] += 1
#             else:
#                 word_pair_count[word_pair_str] = 1
#             # two orders
#             word_pair_str = str(word_j_id) + ',' + str(word_i_id)
#             if word_pair_str in word_pair_count:
#                 word_pair_count[word_pair_str] += 1
#             else:
#                 word_pair_count[word_pair_str] = 1

# row = []
# col = []
# weight = []

# # pmi as weights

# num_window = len(windows)

# for key in word_pair_count:
#     temp = key.split(',')
#     i = int(temp[0])
#     j = int(temp[1])
#     count = word_pair_count[key]
#     word_freq_i = word_window_freq[vocab[i]]
#     word_freq_j = word_window_freq[vocab[j]]
#     pmi = log((1.0 * count / num_window) /
#               (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
#     if pmi <= 0:
#         continue
#     row.append(train_size + i)
#     col.append(train_size + j)
#     weight.append(pmi)

# # word vector cosine similarity as weights

# '''
# for i in range(vocab_size):
#     for j in range(vocab_size):
#         if vocab[i] in word_vector_map and vocab[j] in word_vector_map:
#             vector_i = np.array(word_vector_map[vocab[i]])
#             vector_j = np.array(word_vector_map[vocab[j]])
#             similarity = 1.0 - cosine(vector_i, vector_j)
#             if similarity > 0.9:
#                 print(vocab[i], vocab[j], similarity)
#                 row.append(train_size + i)
#                 col.append(train_size + j)
#                 weight.append(similarity)
# '''
# # doc word frequency
# doc_word_freq = {}

# for doc_id in range(len(shuffle_doc_words_list)):
#     doc_words = shuffle_doc_words_list[doc_id]
#     words = doc_words.split()
#     for word in words:
#         word_id = word_id_map[word]
#         doc_word_str = str(doc_id) + ',' + str(word_id)
#         if doc_word_str in doc_word_freq:
#             doc_word_freq[doc_word_str] += 1
#         else:
#             doc_word_freq[doc_word_str] = 1

# for i in range(len(shuffle_doc_words_list)):
#     doc_words = shuffle_doc_words_list[i]
#     words = doc_words.split()
#     doc_word_set = set()
#     for word in words:
#         if word in doc_word_set:
#             continue
#         j = word_id_map[word]
#         key = str(i) + ',' + str(j)
#         freq = doc_word_freq[key]
#         if i < train_size:
#             row.append(i)
#         else:
#             row.append(i + vocab_size)
#         col.append(train_size + j)
#         idf = log(1.0 * len(shuffle_doc_words_list) /
#                   word_doc_freq[vocab[j]])
#         weight.append(freq * idf)
#         doc_word_set.add(word)

# node_size = train_size + vocab_size + test_size
# adj = sp.csr_matrix(
#     (weight, (row, col)), shape=(node_size, node_size))

# # dump objects
# f = open("data/ind.{}.x".format(dataset), 'wb')
# pkl.dump(x, f)
# f.close()

# f = open("data/ind.{}.y".format(dataset), 'wb')
# pkl.dump(y, f)
# f.close()

# f = open("data/ind.{}.tx".format(dataset), 'wb')
# pkl.dump(tx, f)
# f.close()

# f = open("data/ind.{}.ty".format(dataset), 'wb')
# pkl.dump(ty, f)
# f.close()

# f = open("data/ind.{}.allx".format(dataset), 'wb')
# pkl.dump(allx, f)
# f.close()

# f = open("data/ind.{}.ally".format(dataset), 'wb')
# pkl.dump(ally, f)
# f.close()

# f = open("data/ind.{}.adj".format(dataset), 'wb')
# pkl.dump(adj, f)
# f.close()