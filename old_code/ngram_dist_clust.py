from collections import defaultdict
import scipy.cluster.hierarchy
import generate_token_dict
import numpy as np
import ngram_dist
import sys

# We read in the path of the root directory and n
root_path = sys.argv[1]
token_length = int(sys.argv[2])

#=========1=========2=========3=========4=========5=========6=========7=

def get_paths(root_path, n):

    allpaths = generate_token_dict.DFS(root_path, n)
    file_pathtokens_dict = allpaths[0]
    file_path_dict = allpaths[1]

    return file_path_dict

#=========1=========2=========3=========4=========5=========6=========7=

# INPUT: A dictionary with filenames as keys and tuples of ngrams
# RETURNS: A list of all the paths
def dict_vals_list(dictionary):
    path_list = []
    for key, value in dictionary.items():
        path_list.append(value)
    return path_list

#=========1=========2=========3=========4=========5=========6=========7=

# We generate the filename to tokens dict
file_pathtokens_dict = get_tokens(root_path, token_length)
path_list = dict_vals_list(file_pathtokens_dict)
print("length of path_list: ", len(path_list))

# DISTANCE MATRIX COMPUTATION
# NOTE it might be nice to have the filenames as well, but this is not 
# necessary because the paths contain the filenames. 
def compute_dist_matrix(path_list):
    # initialize the distance matrix, it will be two dimensional, with
    # number of rows and columns equal to the length of path_list.
    dist_matrix = []
    for path_a in path_list:
        row = []
        for path_b in path_list:
            # We use ngram word_dist implemented in another script
            dist = ngram_dist.word_dist(token_length,path_a,path_b)
            row.append(dist)
        dist_matrix.append(row)


