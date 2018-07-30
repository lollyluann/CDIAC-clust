from tqdm import tqdm
import path_utilities

import numpy as np

from tqdm import tqdm
import sys
import os

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETER: two DIRECTORY paths (not file paths)
    RETURNS: distance between the files in the specified directories '''
def path_dist(path1, path2):
    path1_folders = path1.split("/")
    path2_folders = path2.split("/")
    # remove empty strings
    path1_folders = list(filter(None, path1_folders))
    path2_folders = list(filter(None, path2_folders))
    
    # "shared" contains the shared directory closest to the bottom
    # "common_index" contains the tree depth of "shared", e.g.
    # the common_index of the top level directory is 0. Its 
    # grandchildren have common index 2. 
    shared = ""
    common_index = 0
    min_folderlist_len = min(len(path1_folders), len(path2_folders))
    # iterate over the smaller of the lengths of the folderlists
    for i in range(min_folderlist_len):
        # if the directories match
        if path1_folders[i] == path2_folders[i]:
            # save the directory name in shared, save common index
            shared = path1_folders[i]
            common_index = i
        # once they are no longer equal, stop iterating. 
        else:
            break
    
    # compute distances to closest common ancestor
    p1_dist_to_shared = len(path1_folders)-1-common_index
    p2_dist_to_shared = len(path2_folders)-1-common_index

    total_dist = p1_dist_to_shared + p2_dist_to_shared
    return total_dist

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETER: a path to a directory
    DOES: checks if the directory alone (not children) has any FILES '''
def has_files(directory):
    for item in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, item)):
            return True
    return False

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETER: a path to a directory
    RETURNS: a list of the folders with files in them in "path" '''
def DFS(path):
    stack = []
    all_dirs = []
    stack.append(path);
    while len(stack) > 0:
        # pop a path off the stack 
        tmp = stack.pop(len(stack) - 1)
        # if this is a directory and has at least one file in it
        if os.path.isdir(tmp):
            if has_files(tmp)==True:
                all_dirs.append(tmp)
            # for every item in the "tmp" directory
            for item in os.listdir(tmp):
                # throws the path given by "tmp" + "item" onto the stack
                stack.append(os.path.join(tmp, item))
    return all_dirs

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETER: a directory path
    RETURNS: Naive max distance according to above distance metric. '''
def naive_max_dist(root):
    max_dist = 0
    all_paths = DFS(root)
    for path_a in tqdm(all_paths):
        for path_b in all_paths:
            dist = path_dist(path_a, path_b) 
            if (max_dist < dist):
                max_dist = dist
    return max_dist

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETER: all the paths in a cluster
    RETURNS: the average intracluster distance '''
def intracluster_dist(cluster_paths, cluster_directory_dict):
    
    distances = []
    for directory_a, freq_a in tqdm(cluster_directory_dict.items()):
        for directory_b, freq_b in cluster_directory_dict.items():
            dist = path_dist(directory_a, directory_b)
            for i in range(freq_a * freq_b):
                distances.append(dist)
    dists = np.array(distances)
    
    # compute the max distance over all directories in this cluster
    max_dist = 0
    for directory_a, freq_a in tqdm(cluster_directory_dict.items()):
        for directory_b, freq_b in cluster_directory_dict.items():
            dist = path_dist(directory_a, directory_b)
            if (max_dist < dist):
                max_dist = dist
    if max_dist == 0:
        max_dist = 1
 
    return np.mean(dists) / max_dist

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETER: all the paths in each cluster, and a dict mapping unique
               directories to counts. 
    RETURNS: tuple of a list of the naive intracluster distances
             for each cluster and the total normalized version of these
             scores.  '''
def compute_naive_score(list_cluster_lists, cluster_directories):
    
    # number of files in each cluster 
    cluster_file_counts = []
    num_clusters = len(list_cluster_lists)

    for i in tqdm(range(num_clusters)):

        # list of the paths in the ith cluster
        cluster_paths = list_cluster_lists[i]
        cluster_file_count = len(cluster_paths)
        cluster_file_counts.append(cluster_file_count)

    total_num_files = sum(cluster_file_counts)

    # naive tree dist scores for each cluster    
    cluster_scores = []
    
    # scores scaled by cluster size, sum is normalized
    scaled_scores = []
    for i in tqdm(range(num_clusters)):
        cluster_paths = list_cluster_lists[i]
        cluster_directory_dict = cluster_directories[i]
        naive_score = 1 - intracluster_dist(cluster_paths, 
                                            cluster_directory_dict)
        cluster_weight = cluster_file_counts[i] / total_num_files
        scaled_scores.append(naive_score * cluster_weight)
        cluster_scores.append(naive_score)
        print("Naive score for cluster ", i, "is: ", naive_score)
    
    total_naive_score = sum(scaled_scores)
    print("Total naive score: ", total_naive_score)
    return cluster_scores, total_naive_score
        
#=========1=========2=========3=========4=========5=========6=========7=

def main():
    root_path = sys.argv[1]
    max_dist = naive_max_dist(root_path)
    print("The max_dist is: ", max_dist)

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main() 










