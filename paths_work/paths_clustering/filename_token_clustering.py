from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import metrics
import codecs
import numpy as np
import numpy
import sys
import os

# ARGUMENTS
vector_file = sys.argv[1]
min_k = int(sys.argv[2])
max_k = int(sys.argv[3])


# runs through all k values and finds optimal k value for silhouette score. 

#=========1=========2=========3=========4=========5=========6=========7=

class autovivify_list(dict):
  '''A pickleable version of collections.defaultdict'''
  def __missing__(self, key):
    '''Given a missing key, set initial value to an empty list'''
    value = self[key] = []
    return value

  def __add__(self, x):
    '''Override addition for numeric types when self is empty'''
    if not self and isinstance(x, Number):
      return x
    raise ValueError

  def __sub__(self, x):
    '''Also provide subtraction method'''
    if not self and isinstance(x, Number):
      return -1 * x
    raise ValueError

def build_word_vector_matrix(vector_file, n_words):
  '''Return the vectors and labels for the first n_words in vector file'''
  numpy_arrays = []
  labels_array = []
  with codecs.open(vector_file, 'r', 'utf-8') as f:
    for c, r in enumerate(f):
      sr = r.split()
      labels_array.append(sr[0])
      numpy_arrays.append( numpy.array([float(i) for i in sr[1:]]) )
      if c == n_words:
        return numpy.array( numpy_arrays ), labels_array
  return numpy.array( numpy_arrays ), labels_array

def find_word_clusters(labels_array, cluster_labels):
  '''Return the set of words in each cluster'''
  cluster_to_words = autovivify_list()
  for c, i in enumerate(cluster_labels):
    cluster_to_words[ i ].append( labels_array[c] )
  return cluster_to_words

def kmeans(k, df):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=8, max_iter=300, 
          tol=0.0001, precompute_distances='auto', verbose=0, 
          random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
    kmeans.fit(df)

    cluster_labels = kmeans.labels_
    return cluster_labels

def dbscan(df):
    db = DBSCAN(eps=0.5, min_samples=300).fit(df)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(n_clusters)
    return labels

#=========1=========2=========3=========4=========5=========6=========7=

def optimize_k(df, labels_array, min_k, max_k):
    best_sil_score = -1
    best_k = 0
    for k in range(min_k, max_k + 1):
        cluster_labels = kmeans(k,df) 
        cluster_to_words = find_word_clusters(labels_array, cluster_labels)

        # we evaluate our clusters using several measures of well-definedness
        print("Silhouette score: ")
        sil_score = metrics.silhouette_score(df, cluster_labels, metric='euclidean')
        print(sil_score)
        if (sil_score > best_sil_score):
            best_sil_score = sil_score
            best_k = k
        print("Calinksi-harabaz score: ")
        ch_score = metrics.calinski_harabaz_score(df, cluster_labels)
        print(ch_score)
    

    cluster_labels = kmeans(best_k, df)
    cluster_to_words = find_word_clusters(labels_array, cluster_labels)
    # we evaluate our clusters using several measures of well-definedness
    print("Silhouette score: ")
    sil_score = metrics.silhouette_score(df, cluster_labels, metric='euclidean')
    print(sil_score)
    print("Calinksi-harabaz score: ")
    ch_score = metrics.calinski_harabaz_score(df, cluster_labels)
    print(ch_score)
    
    output = open("clusters.txt", "w")
    os.chdir("/home/ljung/CDIAC-clust")
    for c in cluster_to_words:
        #print(cluster_to_words[c], "\n")
        output.write(str(cluster_to_words[c]) + "\n")

    output.close()

    print("The optimal k value is: ", best_k)



def main():
    # MAIN FUNCTION
    df, labels_array = build_word_vector_matrix(vector_file, 14869)
    optimize_k(df, labels_array, min_k, max_k)
    #dbscan(df)
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main() 







