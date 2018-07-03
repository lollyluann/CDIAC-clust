from pyxdameraulevenshtein import damerau_levenshtein_distance
from collections import defaultdict
import scipy.cluster.hierarchy
import generate_shortened_token_dict
import numpy as np
import sys
import os

def cluster_ngrams(n, ngrams, compute_distance, max_dist, method):
    """
    Cluster ngrams.
    Params:
        ngrams: [list] List of tuple of words in each ngram to cluster.
        compute_distance: [func] Function that computes distance between two
            pairs of ngrams.
        max_dist: [float] Maximum distance allowed for two clusters to merge.
        method: [string] Method to use for clustering.  'single',
            'complete', 'average', 'centroid', 'median', 'ward', or 'weighted'.
            See http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html for details.
    Returns:
        clusters: [list] List of ngrams in each cluster.
    """
    indices = np.triu_indices(len(ngrams), 1)
    pairwise_dists = np.apply_along_axis(
        lambda col: compute_distance(ngrams[col[0]], ngrams[col[1]]),
        0, indices)
    hierarchy = scipy.cluster.hierarchy.linkage(pairwise_dists, method=method)
    clusters = dict((i, [i]) for i in range(len(ngrams)))
    for (i, iteration) in enumerate(hierarchy):
        cl1, cl2, dist, num_items = iteration
        if dist >  max_dist:
            break
        items1 = clusters[cl1]
        items2 = clusters[cl2]
        del clusters[cl1]
        del clusters[cl2]
        clusters[len(ngrams) + i] = items1 + items2
    ngram_clusters = []
    for cluster in clusters.values():
        ngram_clusters.append([ngrams[i] for i in cluster])
    return ngram_clusters


def dl_ngram_dist(ngram1, ngram2):
    """
    Compute distance between ngrams by summing the Damerau-Levenshtein distance
    for consecutive words in ngrams.
    Params:
        ngram1: [tuple] Tuple of words.
        ngram2: [tuple] Tuple of words.
    Returns:
        distance [int] Measure of distance between two ngrams.
    """
    return sum(damerau_levenshtein_distance(w1, w2) for w1, w2 in zip(ngram1,
        ngram2))

def get_tokens(root_path, n):

    allpaths = generate_shortened_token_dict.DFS(root_path, n)
    # print(allpaths)

    #os.chdir(root_path)

    file_pathtokens_dict = allpaths[0]
    file_path_dict = allpaths[1]

    # Some stuff we were using earlier to write to a file
    #f1 = open(root_path + "file_pathtokens_dict.txt","w")
    #f2 = open(root_path + "file_path_dict.txt","w")
    #f1.write( str(file_pathtokens_dict) )
    #f2.write( str(file_path_dict) )
    #f1.close()
    #f2.close()

    return file_pathtokens_dict

# INPUT: A dictionary with filenames as keys and tuples of ngrams
# RETURNS: A list with repeats of all the god damned ngrams
def dict_vals_list(dictionary):
    ngram_list = []
    for key, value in dictionary.items():
        for x in value:
            ngram_list.append(x)
    return ngram_list

# We read in the path of the root directory and n
root_path = sys.argv[1]
token_length = int(sys.argv[2])

# We generate the filename to tokens dict
file_pathtokens_dict = get_tokens(root_path, token_length)
ngram_list = dict_vals_list(file_pathtokens_dict)

# We cluster.
ngram_clusters = cluster_ngrams(token_length, ngram_list, dl_ngram_dist, 4, "average")
print(ngram_clusters)

os.chdir("/home/ljung/CDIAC-clust")
f = open("ngram_clusters_output.txt", "w")
for piece in ngram_clusters:
    for x in piece:
        f.write(x)
f.close()

