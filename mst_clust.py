from collections import defaultdict
import scipy.cluster.hierarchy
import generate_token_dict
import numpy as np
import ngram_dist
import sys

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

# We read in the path of the root directory and n
root_path = sys.argv[1]
token_length = int(sys.argv[2])

# We generate the filename to tokens dict
file_pathtokens_dict = get_tokens(root_path, token_length)
path_list = dict_vals_list(file_pathtokens_dict)



