from sklearn.cluster import KMeans
import numpy, sys, os, codecs

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

# MAIN FUNCTION

vector_file = sys.argv[1]
num_clusters = int(sys.argv[2])

df, labels_array = build_word_vector_matrix(vector_file, 14869)
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init=8, max_iter=300, 
      tol=0.0001, precompute_distances='auto', verbose=0, 
      random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
kmeans.fit(df)

cluster_labels = kmeans.labels_
cluster_inertia = kmeans.inertia_
cluster_to_words = find_word_clusters(labels_array, cluster_labels)
#print(kmeans.cluster_centers_)

output = open("clusters.txt", "w")
os.chdir("/home/ljung/CDIAC-clust")
for c in cluster_to_words:
    print(cluster_to_words[c], "\n")
    output.write(str(cluster_to_words[c]) + "\n")

output.close()


