from nltk.corpus import reuters
import pprint
import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

START_TOKEN = '<START>'
END_TOKEN = '<END>'

def read_corpus(category="crude"):
    files = reuters.fileids(category)
    return [[START_TOKEN] + [w.lower() for w in list(reuters.words(f))] + [END_TOKEN] for f in files]
    '''
    list_return = []
    for f in files:
        list_ = [START_TOKEN]
        for w in list(reuters.words(f)):
            list_.append(w.lower())
        list_.append(END_TOKEN)
        list_return.append(list_)
    return list_return
    '''

#reuters_corpus = read_corpus()
#pprint.pprint(reuters_corpus[:3], compact=True, width=100)
#print(len(reuters_corpus))

def  distinct_words(corpus):
    corpus_words = []
    num_corpus_words = -1
    for list_ in corpus:
        for words in list_:
            corpus_words.append(words)
    corpus_words = list(set(corpus_words))
    if (len(corpus_words) != 0):
        num_corpus_words = len(corpus_words)
    return sorted(corpus_words), num_corpus_words

def compute_co_occurrence_matrix(corpus, window_size=4):
    words, num_words = distinct_words(corpus)
    M = None
    word2Ind = {}
    '''
    id_ = 0
    for w in words:
        word2Ind[w] = id_
        id_ += 1
    '''
    word2Ind = dict(zip(words, range(num_words)))
    M = np.zeros((num_words, num_words))
    for w in words:
        for doc in corpus:
            for i in range(len(doc)):
                if (doc[i] == w):
                    for j in range(max(0, i - window_size), i):
                        M[word2Ind[w]][word2Ind[doc[j]]] += 1
                    for j in range(i + 1, min(i + window_size + 1, len(doc))):
                        M[word2Ind[w]][word2Ind[doc[j]]] += 1
    return M, word2Ind

def reduce_to_k_dim(M, k=2):
    n_iters = 10
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))
    SVD = TruncatedSVD(n_components=k, n_iter=n_iters)
    M_reduced = SVD.fit_transform(M)
    print("Done.")
    return M_reduced

def plot_embeddings(M_reduced, word2Ind, words):
    x = []
    y = []
    for pair in M_reduced:
        x.append(pair[0])
        y.append(pair[1])
    plt.plot(x, y, color='w', marker='+', markeredgecolor='r', markersize=5)
    for i in range(len(words)):
        plt.text(M_reduced[word2Ind[words[i]]][0], M_reduced[word2Ind[words[i]]][1], words[i], fontsize=4)
    plt.show()

reuters_corpus = read_corpus()
M_co_occurrence, word2Ind_co_occurrence = compute_co_occurrence_matrix(reuters_corpus)
M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)

M_lengths = np.linalg.norm(M_reduced_co_occurrence, axis=1)
M_normalized = M_reduced_co_occurrence / M_lengths[:, np.newaxis]

words = ['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']

plot_embeddings(M_normalized, word2Ind_co_occurrence, words)