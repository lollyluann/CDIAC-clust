from sklearn.feature_selection import VarianceThreshold
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import manifold
import matplotlib
matplotlib.use('Agg')
from numpy import loadtxt
import matplotlib.pyplot as plt
import codecs
import numpy
import sys
import ast
import os

#=========1=========2=========3=========4=========5=========6=========7=

# First argument is the vector file as a .txt

# JUST A FUNCTION TO READ IN THE EMBEDDING TXT FILE
def build_word_vector_matrix(vector_file, n_words):
  # Return the vectors and labels for the first n_words in vector file
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

# MAIN PROGRAM
vector_file = sys.argv[1]
clusters = sys.argv[2]
path_vectors, paths = build_word_vector_matrix(vector_file, 14869)


# project the vectors down into R^3 using PCA 
pca = PCA(n_components=3)
reduced_path_vectors = pca.fit_transform(path_vectors)


'''
# project the vectors down into R^3 using feature subset selection
sel = VarianceThreshold(threshold=(.00001 * (1 - .00001)))
reduced_path_vectors = sel.fit_transform(path_vectors)
print(reduced_path_vectors.shape)
'''

'''
mds = manifold.MDS(n_components=3, n_jobs=4, dissimilarity="euclidean", random_state=1, verbose=2)
print("Fitting to the path_vectors matrix. ")
reduced_path_vectors = mds.fit_transform(path_vectors)  # shape (n_components, n_samples)
print("Done fitting. ")
'''

# create a dict that maps paths to 3D vectors
reduced_path_dict = {}
for i in range(len(paths)):
    reduced_path_dict.update({paths[i]:reduced_path_vectors[i]})

#print(reduced_path_dict)

# plot in 3D
fig = plt.figure(figsize=(20,10))
ax = Axes3D(fig)

#=========1=========2=========3=========4=========5=========6=========7=

with open(clusters) as f:
    cluster_list = f.readlines()
# list of clusters, each cluster is a list of paths as strings
for cluster in cluster_list:
    color = numpy.random.rand(3,)
    cluster_path_list = ast.literal_eval(cluster)
    for path_name in cluster_path_list:
        vector = reduced_path_dict.get(path_name)    
        ax.scatter(vector[0], vector[1], vector[2],
        s=fig.dpi/72., marker='.',c=color)
       
plt.savefig("embedding_3D", dpi=1000)
'''
# get list of x,y,z components
x = []
y = []
z = []
for vector in reduced_path_vectors:
    x.append(vector[0])
    y.append(vector[1])
    z.append(vector[2])
#print(x)

# plot in 3D
fig = plt.figure(figsize=(20,10))
ax = Axes3D(fig)
ax.scatter(x, y, z, s=fig.dpi/72., marker='.')
plt.savefig("embedding_3D", dpi=1000)
'''
