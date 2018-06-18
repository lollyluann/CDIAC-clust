from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import codecs
import numpy
import sys
import os

# First argument is the vector file as a .txt

# JUST A FUNCTION TO READ IN THE EMBEDDING TXT FILE
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

# MAIN PROGRAM
vector_file = sys.argv[1]
path_vectors, paths = build_word_vector_matrix(vector_file, 14869)

# project the vectors down into R^3. 
pca = PCA(n_components=3)
reduced_path_vectors = pca.fit_transform(path_vectors)

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
fig = plt.figure()
ax = Axes3D(fig)
#ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, s=(72./fig.dpi)**2)
plt.savefig("embedding_3D", dpi=1000)

