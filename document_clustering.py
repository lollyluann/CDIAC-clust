from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans

import sys, logging, os
from time import time
import numpy as np
from nltk.corpus.reader.plaintext import PlaintextCorpusReader

#=========1=========2=========3=========4=========5=========6=========7=

#f = open("doc_cluster_output.txt","w")
corpusdir = "/home/ljung/extension_sorted_data/converted_pdfs/"
newcorpus = PlaintextCorpusReader(corpusdir, '.*')
dataset = newcorpus.words()

n_ft = int(sys.argv[1]) #default 10000?
idf = True
minibatch = False
hashing = False
n_comp = int(sys.argv[2])
vbs = False
new_k = int(sys.argv[3])

# Extract features from the training dataset using a sparse vectorizer
if hashing:
    # USE HASHING
    if idf:
        # WITH IDF
        # Perform IDF normalization on the output of HashingVectorizer
        hasher = HashingVectorizer(n_features=n_ft, stop_words='english', 
            alternate_sign=False, norm=None, binary=False)
        vectorizer = make_pipeline(hasher, TfidfTransformer())
    else:
        # WITHOUT IDF
        vectorizer = HashingVectorizer(n_features=n_ft, stop_words='english', 
                alternate_sign=False, norm='l2', binary=False)
else:
    # NOT USING HASHING
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_ft,
                 min_df=2, stop_words='english', use_idf=idf)

X = vectorizer.fit_transform(dataset)
print("n_samples: %d, n_features: %d" % X.shape)

#=========1=========2=========3=========4=========5=========6=========7=

#Perform dimensionality reduction using LSA")
if n_comp>0:
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(n_comp)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)

    print("done in %fs" % (time() - t0))
    explained_variance = svd.explained_variance_ratio_.sum()

if not hashing:
    print("Top terms per cluster:")

    if n_comp:
        original_space_centroids = svd.inverse_transform(km.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
    else:
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    for i in range(new_k):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()
        print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

#=========1=========2=========3=========4=========5=========6=========7=

if minibatch:
    km = MiniBatchKMeans(n_clusters=new_k, init='k-means++', n_init=1,
                  init_size=1000, batch_size=1000, verbose=vbs)
else:
    km = KMeans(n_clusters=new_k, init='k-means++', max_iter=100, n_init=1,
                verbose=vbs)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
print()

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))

#=========1=========2=========3=========4=========5=========6=========7=

if not hashing:
    print("Top terms per cluster:")
    if n_comp>0:
        original_space_centroids = svd.inverse_transform(km.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
    else:
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    for i in range(new_k):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()
