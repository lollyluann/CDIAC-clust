import matplotlib
matplotlib.use('Agg')

from sklearn.cluster import AgglomerativeClustering
from worse_schema_finder import get_better_dict
from sklearn.neighbors import kneighbors_graph
from plot_dendrogram import plot_dendrogram 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.backends.backend_pdf
import matplotlib.gridspec as gridspec
import get_cluster_stats as get_stats
from matplotlib import pyplot as plt
import calculate_file_distances
from collections import Counter
from sklearn import manifold
from tqdm import tqdm
import path_utilities
import pylatex as pl
import pandas as pd
import numpy as np
import sklearn
import pickle
import sys
import csv
import os
import re

np.set_printoptions(threshold=np.nan)

# Converts all the .xls or .xlsx files in a directory to .csv files. 
# Then it clusters the schemas of these .csv files using agglomerative
# clustering. 

#=========1=========2=========3=========4=========5=========6=========7=

# ARGUMENTS
# source directory and output directory
directory = sys.argv[1]
out_dir = sys.argv[2]           # location of converted files. 
ext_dict_dir = sys.argv[3]      # location of ext ".npy" file. 
overwrite = sys.argv[4]         # overwrite the distance matrix file.
overwrite_plot = sys.argv[5]    # overwrite plot cache. 
# overwrite is a string, should be "0" for don't overwrite, and "1"
# for do

def check_valid_dir(some_dir):
    if not os.path.isdir(directory):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("")
        print("DIES IST EIN UNGÜLTIGES VERZEICHNIS!!!!")
        print("")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        exit()

check_valid_dir(directory)
check_valid_dir(out_dir)
check_valid_dir(ext_dict_dir)

directory = os.path.abspath(directory)
out_dir = os.path.abspath(out_dir)
ext_dict_dir = os.path.abspath(ext_dict_dir)

print(directory)
print(out_dir)
print(ext_dict_dir)

'''
script_dir = os.path.dirname(os.path.abspath(__file__))
paths_work_dir = os.path.join(script_dir, "../paths_work/")
sys.path.append(paths_work_dir)
import calculate_file_distances
'''

xls_path = os.path.join(directory, "xls/")
xlsx_path = os.path.join(directory, "xlsx/")
csv_path = os.path.join(directory, "csv/")
tsv_path = os.path.join(directory, "tsv/")

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

#RETURNS: a dictionary which maps filenames to csvs header lists. 
def get_header_dict(csv_dir, csv_path_list, 
                    fill_threshold, converted_status):
    header_dict = {}
    # number of files with no valid header
    bad_files = 0
    decode_probs = 0

    # This code is rather confusing because I wanted the function to 
    # be able to handle both types of inputs (lists of paths in names)
    # and just directory locations. 

    # CASE 1:
    # If we're reading in converted files, we only need the csv_dir
    # argument, so we get a list of the filenames from that directory. 
    # These filenames are in the form:
    # "@home@ljung@pub8@oceans@some_file.csv"
    if (converted_status):
        dir_list = os.listdir(csv_dir)

    # CASE 2:
    # Otherwise, we are reading in a list of the true, original 
    # locations of files that were csvs to begin with in the dataset.
    else:
        dir_list = csv_path_list

#=========1=========2=========3=========4=========5=========6=========7=
    
    # CASE 1: "path" looks like:"@home@ljung@pub8@oceans@some_file.csv" 
    # CASE 2: "path" is literally the path of that file in the original
    # dataset as a string. 
    for path in tqdm(dir_list):
        if (converted_status): 
            # get the new location of the current file in "csv_dir", 
            # i.e. not in original dataset. 
            filename = path
            path = os.path.join(csv_dir, path) 
        else:
            # convert to "@home@ljung@pub8@oceans@some_file.csv" form. 
            filename = path_utilities.str_encode(path)

        # So now in both cases, filename has the "@"s, and path is
        # the location of some copy of the file. 

        with open(path, "r") as f:
            # read csv and get the header as a list
            reader = csv.reader(f)
            try:
                header_list = next(reader)
                
                # if the header is empty, try the next line
                if (len(header_list) == 0):
                    header_list = next(reader)
                 
                # number of nonempty attribute strings
                num_nonempty = 0
                for attribute in header_list:
                    if not (attribute == ""):
                        num_nonempty = num_nonempty + 1
                fill_ratio = num_nonempty / len(header_list)                

                # keep checking lines until you get one where there
                # are enough nonempty attributes
                while (fill_ratio <= fill_threshold):
                    # if there's only one nonempty attribute, it's
                    # probably just a descriptor of the table, so try
                    # the next line. 
                    header_list = next(reader)
                    num_nonempty = 0
                    for attribute in header_list:
                        if not (attribute == ""):
                            num_nonempty = num_nonempty + 1
                    if (len(header_list) == 0):
                        fill_ratio = -1
                    else:
                        fill_ratio = num_nonempty / len(header_list)
                    
                    #===================================================
                    # Here we've hardcoded some information about 
                    # scientific data to work better with CDIAC. 
                    # feel free to remove it. 
                    
                    # people seem to denote pre-header stuff with a *
                    for attribute in header_list:
                        if (attribute != "" and attribute[-1] == "*"):
                            fill_ratio = -1
                    if (len(header_list) > 3):
                        if (header_list[0] == "Year" 
                            and header_list[2] != ""):
                            break
                        if (header_list[0] == "Citation"):
                            fill_ratio = -1
                    #===================================================
    
            except UnicodeDecodeError:
                decode_probs = decode_probs + 1                    
            except StopIteration:
                bad_files = bad_files + 1
                #os.system("cp " + path + " ~/bad_csvs/")
                continue
            # throw a key value pair in the dict, with filename as key
            header_dict.update({filename:header_list})
    print("Throwing out this number of files, all have less than ", 
          fill_threshold*100, 
          "% nonempty cells in every row: ", bad_files)    
    print("Number of UnicodeDecodeErrors: ", decode_probs)
    return header_dict

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

#RETURNS: Jaccard distance between two lists of strings. 
def jaccard_distance(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    #print(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    if (union == 0):
        union = 1
    return 1 - float(intersection / union)
    
#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

# DOES: computes the jaccard distance matrix of the headers in 
# header_dict. 
# RETURNS: a tuple with the first element being an array of all the 
# headers in numpy array form, the second being the jaccard dist
# matrix, and the third being a list of 2-tuples (filename, header_list)
def dist_mat_generator(header_dict, overwrite_path, overwrite):

    # Define the names for the files we write distance matrix and the
    # filename_header_pairs list to. 
    dist_mat_path = overwrite_path + "-dist.npy"
    headpairs_path = overwrite_path + "-headpairs.pkl"

    schema_matrix = []
    # list of tuples, first element is filename, second is header_list
    filename_header_pairs = []

    for filename, header_list in header_dict.items():
        schema_matrix.append(header_list)
        filename_header_pairs.append([filename, header_list])
   
    # we just need an empty numpy array 
    jacc_matrix = np.zeros((2,1))

#=========1=========2=========3=========4=========5=========6=========7=

    if (not os.path.isfile(dist_mat_path) or 
        not os.path.isfile(headpairs_path) or overwrite == "1"):
        
        print("No existing cached files for this directory. ")
        print("Generating distance matrix using jaccard similarity. ")
        print("This could take a while... ")
        # we generate the distance matrix as a list
        dist_mat_list = []
        # iterating over the header array once...
        for header_a in tqdm(schema_matrix):
            #===========================
            print(header_a)
            #===========================
            # storing distances for a single header
            single_row = []
            # iterating again...
            for header_b in schema_matrix:
                jacc = jaccard_distance(header_a, header_b)
                single_row.append(jacc)
            # add one row to the list
            dist_mat_list.append(np.array(single_row))
        # convert list to numpy array
        jacc_matrix = np.array(dist_mat_list)
        jacc_matrix = np.stack(jacc_matrix, axis=0)
        print(jacc_matrix.shape)
        # save on disk, because computation is expensive
        print("Saving file to: ", dist_mat_path)
        np.save(dist_mat_path, jacc_matrix)
        with open(headpairs_path, 'wb') as f:
            pickle.dump(filename_header_pairs, f)
    else:
        print("Loading file from: ", dist_mat_path)
        jacc_matrix = np.load(dist_mat_path)
        with open(headpairs_path, 'rb') as f:
            filename_header_pairs = pickle.load(f)
                
    return schema_matrix, jacc_matrix, filename_header_pairs

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

def agglomerative(jacc_matrix, 
                  num_clusters, 
                  filename_header_pairs, 
                  overwrite):
    clustering = AgglomerativeClustering(n_clusters=num_clusters, 
                                         affinity='precomputed', 
                                         linkage='complete')
    clustering.fit(jacc_matrix)
    labels = clustering.labels_
    #print(labels)

    if (overwrite == 1):
        plt.figure(figsize=(17,9))
        plot_dendrogram(clustering, labels = clustering.labels_)
        
        plt.savefig("dendrogram", dpi=300)
        print("dendrogram written to \"dendrogram.png\"")
    
    return labels

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

# DOES: plots the schema_clusters for the csv files. 
def plot_clusters(jacc_matrix, labels, plot_mat_path, overwrite_plot):

    if not os.path.isfile(plot_mat_path) or overwrite_plot == "1":
        # multidimensional scaling to convert distance matrix into 3D
        mds = manifold.MDS(n_components=3, n_jobs=4, 
                           dissimilarity="precomputed", 
                           random_state=1, verbose=2)
        print("Fitting to the distance matrix. ")
        # shape (n_components, n_samples)
        pos = mds.fit_transform(jacc_matrix)
        np.save(plot_mat_path,pos)
    else:
        pos = np.load(plot_mat_path)
    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]

    # set up plot
    print("Setting up plot. ")
    fig = plt.figure(figsize=(17,9))
    ax = Axes3D(fig)

    # create data frame with MDS results, cluster numbers, filenames
    df = pd.DataFrame(dict(x=xs, y=ys, z=zs, label=labels)) 
    # group by cluster
    groups = df.groupby('label')

    # for each cluster, plot the files in that cluster
    for name, group in tqdm(groups):
        # color = ('#%06X' % random.randint(0,256**3-1))
        color = np.random.rand(3,)
        for t in range(group.shape[0]):
            ax.scatter(group.x.iloc[t], 
                       group.y.iloc[t], 
                       group.z.iloc[t], 
                       c=color, marker='o')
            ax.set_aspect('auto')

    plt.savefig("3D_schema_cluster", dpi=300)
    print("scatter plot written to \"3D_schema_cluster.png\"")
    return

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

# NOT NORMALIZED YET, SHOULD NORMALIZE IT
def tree_dist(path1, path2, max_dist):
    return calculate_file_distances.path_dist(path1, path2) / max_dist

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

# RETURNS:  Silhouette coefficient of clustering, and list of the
#           silhouette coeffs for each of the clusters.
def compute_silhouette(cluster_directories, root_path):

    if (len(cluster_directories) < 2):
        print("Must be at least 2 clusters. ")
        return 0

    max_dist = calculate_file_distances.naive_max_dist(root_path)
    silhouette_list = []
    total_num_files = 0
    total_silhouette_sum = 0
    for cluster_i in tqdm(cluster_directories):
        num_files_i = 0
        silhouette_sum_i = 0
        for dir_x, count_x in tqdm(cluster_i.items()):
            # COMPUTE INTERCLUSTER DISTANCE
            # for each point
            set_of_means_x = []
            for cluster_j in cluster_directories:
                if (cluster_j == cluster_i):
                    # print("Cluster match identified.")
                    pass
                else:
                    total_count_j = 0
                    total_dist_j = 0
                    for dir_y, count_y in cluster_j.items():
                        # Make sure this function normalizes
                        # get the distance between the two dirs
                        xy_dist = tree_dist(dir_x, dir_y, max_dist)
                        # multiply by the y count to add to the sum
                        # since we have "count_y" files in this dir
                        # to take dist with. 
                        xy_dist *= count_y
                        # add this quantity to the running sum so we
                        # can compute the mean for x,j.
                        total_dist_j += xy_dist
                        # keep track of total count so we can divide 
                        # by it when computing mean.  
                        total_count_j += count_y
                    if (total_count_j == 0):
                        total_count_j = 1
                    mean_x_j = total_dist_j / total_count_j
                    set_of_means_x.append(mean_x_j)
            min_mean_x = min(set_of_means_x)
            
            # COMPUTE INTRACLUSTER DISTANCE
            total_count_i = 0
            total_dist_i = 0
            for dir_y, count_y in cluster_i.items():
                if (dir_y == dir_x):
                    # print("Directory match identified. ")
                    pass
                else:
                    xy_dist = tree_dist(dir_x, dir_y, max_dist)
                    # multiply by the y count to add to the sum
                    # since we have "count_y" files in this dir
                    # to take dist with. 
                    xy_dist *= count_y
                    # add this quantity to the running sum so we
                    # can compute the mean for x,j.
                    total_dist_i += xy_dist
                    # keep track of total count so we can divide 
                    # by it when computing mean.  
                    total_count_i += count_y
            # get the mean of the distances between x and the other 
            # files in cluster i
            if (total_count_i == 0):
                total_count_i = 1
            mean_x_i = total_dist_i / total_count_i
            # compute the denominator of silhouette for x
            silhouette_x_denom = max([min_mean_x, mean_x_i])
            # compute silhouette of x
            if (silhouette_x_denom == 0):
                silhouette_x_denom = 1
            silhouette_x = (min_mean_x - mean_x_i) / silhouette_x_denom
            # add the silhouette for x to our sum of all silhouettes in
            # cluster i 
            silhouette_sum_i += silhouette_x
            # count number of files in cluster i
            num_files_i += count_x
        # compute average silhouette for cluster i
        if (num_files_i == 0):
            num_files_i = 1
        silhouette_i = silhouette_sum_i / num_files_i
        # keep track of the total number of files in the dataset
        total_num_files += num_files_i
        # keep track of the total silhouette sum for all clusters
        total_silhouette_sum += silhouette_sum_i
        # add the silhouette coeff for cluster i to our list of coeffs
        silhouette_list.append(silhouette_i)
    # compute the total silhouette coeff for whole clustering
    if (total_num_files == 0):
        total_num_files = 1
    total_silhouette = total_silhouette_sum / total_num_files
    
    return total_silhouette, silhouette_list

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

# DOES: generates barcharts which show the distribution of unique
#       filepaths in a cluster. 
def generate_barcharts(filename_header_pairs, 
                       labels, num_clusters, root_path):
    
    # create a dict mapping cluster indices to lists of filepaths
    cluster_filepath_dict = {}
    # list of lists, each list is full of the filepaths for one cluster.
    list_cluster_lists = []
    # list of dicts, keys are unique directories, values are counts
    # each list corresponds to a cluster
    cluster_directories = []
    # initialize each child list. 
    for k in range(num_clusters):
        list_cluster_lists.append([])
        # add k empty dicts
        cluster_directories.append({})    

    # for each label in labels
    for i in tqdm(range(len(labels))):
        # get the corresponding filename
        filename_header_pair = filename_header_pairs[i]
        filename = filename_header_pair[0]
        # transform "@" delimiters to "/"
        filename = path_utilities.str_decode(filename)
        # remove the actual filename to get its directory
        decoded_filepath = path_utilities.remove_path_end(filename)
        # get common prefix of top level dataset directory
        common_prefix = path_utilities.remove_path_end(root_path)
        # remove the common prefix for display on barchart. The " - 1"
        # is so that we include the leading "/". 
        len_pre = len(common_prefix)
        len_decod = len(decoded_filepath)
        decoded_filepath_trunc = decoded_filepath[len_pre - 1:len_decod]
        #print("filename is: ", decoded_filepath)
        # add it to the appropriate list based on the label
        list_cluster_lists[labels[i]].append(decoded_filepath_trunc) 
  
#=========1=========2=========3=========4=========5=========6=========7=

    # create a list of dicts, one for each cluster, which map dirs to 
    # counts. 
    for k in range(num_clusters):
        for directory in list_cluster_lists[k]:
            if directory in cluster_directories[k]:
                old_count = cluster_directories[k].get(directory)
                new_count = old_count + 1
                cluster_directories[k].update({directory:new_count})
            else:
                cluster_directories[k].update({directory:1})
    
    # get a list of the cluster statistic for printing to pdf
    cluster_stats = get_stats.get_cluster_stats(cluster_directories)

    # compute silhouette coefficients for each cluster (sil_list)
    # and for the entire clustering (sil)
    sil, sil_list = compute_silhouette(cluster_directories, root_path)
    for coeff in sil_list:
        print(coeff)
    print("total: ", sil)

    # just make font a bit smaller
    matplotlib.rcParams.update({'font.size': 6})

    # for each cluster
    for k in range(num_clusters):
        single_cluster_stats = cluster_stats[k]
        #fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(10, 20))
        plt.figure(k)
        # get frequencies of the paths
        path_counts = Counter(list_cluster_lists[k])
        
        # Create a datafram from path_counts        
        df = pd.DataFrame.from_dict(path_counts, orient='index')
        # rename the frequency axis
        df = df.rename(columns={ df.columns[0]: "freqs" })
        # sort it with highest freqs on top
        sorted_df = df.sort_values("freqs",ascending=False)
        # take only the top 10
        top_10_slice = sorted_df.head(10)
        # plot with corresponding axes
        top_10_slice.plot(kind='bar')
        # leave enough space for x-axis labels
        # fig.subplots_adjust(hspace=7)

#=========1=========2=========3=========4=========5=========6=========7=

        fig_title = ("Directory distribution for cluster "+str(k)+"\n"
                 +"Number of unique directories: " 
                 +str(single_cluster_stats[0])+"\n"
                 +"Mean frequency: "+str(single_cluster_stats[1])+"\n"
                 +"Median frequency: "+str(single_cluster_stats[3])+"\n"
                 +"Standard deviation of frequencies: " 
                 +str(single_cluster_stats[2])+"\n"
                 +"Closest common ancestor of all directories: " 
                 +single_cluster_stats[4])
        plt.title(fig_title, y=1.08)
        plt.xlabel('Directory')
        plt.ylabel('Quantity of files in directory')
        plt.tight_layout()
    pdf = matplotlib.backends.backend_pdf.PdfPages("tabular_stats.pdf")
    for fig in range(1, plt.gcf().number + 1): # opens empty fig
        pdf.savefig( fig )
    pdf.close()
 
#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

# RETURNS: a list of lists, one for each cluster, which contain
#          attribute, count pairs.  
def get_cluster_attributes(filename_header_pairs, 
                       labels, num_clusters):
    
    # list of dicts, keys are unique attributes, values are counts
    # each list corresponds to a cluster
    attr_dicts = []
    # initialize each child list. 
    for k in range(num_clusters):
        # add k empty dicts
        attr_dicts.append({})    

    # for each label in labels
    for i in tqdm(range(len(labels))):
        # get the corresponding header
        filename_header_pair = filename_header_pairs[i]
        header = filename_header_pair[1]
        # for each attribute in this header
        for attribute in header:
            # if it's already in this cluster's dict
            if attribute in attr_dicts[labels[i]]:
                old_count = attr_dicts[labels[i]].get(attribute)
                new_count = old_count + 1
                # increment the frequency count
                attr_dicts[labels[i]].update({attribute:new_count})
            # otherwise, add it to the dict with a count of 1
            else:
                attr_dicts[labels[i]].update({attribute:1})
    
#=========1=========2=========3=========4=========5=========6=========7=

    clust_attr_lists = []
    array_list = []
    max_length = 0
    for attr_dict in attr_dicts:
        clust_attr_list = []
        for attribute, count in attr_dict.items():
            clust_attr_list.append([attribute,count])
        clust_attr_list = sorted(clust_attr_list, key=lambda x: x[1])
        if (max_length < len(clust_attr_list)):
            max_length = len(clust_attr_list)
        clust_attr_lists.append(clust_attr_list)
        attr_df = pd.DataFrame(clust_attr_list)
        sorted_attr_df = attr_df.iloc[::-1]
        sorted_array = sorted_attr_df.values 
        array_list.append(sorted_array)

    new_array_list = []
    print("max", max_length)
    print("length", len(array_list))
    for array in array_list:
        print("iterating")
        diff = max_length - array.shape[0]
        if (diff > 0):
            print("diff: ",diff)
            arr = np.zeros(shape=(diff, 2))
            #print(arr)
            array = np.append(array, arr, axis=0)
            
        #print(array.shape)
        new_array_list.append(array)

    concat = np.concatenate(new_array_list, axis=1)
    concat = concat[0:50]
    concat_df = pd.DataFrame(concat)
    concat_df.to_csv("top_50_attributes.csv")
    print(concat_df)

    for attr_count_pair in clust_attr_lists[0]:
        attr = attr_count_pair[0]
        count = attr_count_pair[1]
        #print(attr, count)

    
        
    return clust_attr_lists 

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

def main():

    # MAIN PROGRAM:
    num_clusters = 15
    ext_dict_file_loc = os.path.join(ext_dict_dir,"extension_index.npy")
    ext_to_paths_dict = np.load(ext_dict_file_loc).item()
    csv_path_list = []
    if "csv" in ext_to_paths_dict:
        csv_path_list = ext_to_paths_dict["csv"]
    
    # if csvs have less than fill_threshold*100% nonempty cells in 
    # every row then we throw them out of our clustering. 
    fill_threshold = 0.4
    
    # we have two dicts, one made up of files which were converted to
    # csv format, and the other made up of files that were in csv
    # format originally. we concatenate both dicts into "header_dict".
    header_dict_converted = get_better_dict(out_dir, [],  
                                            fill_threshold, True)
    #header_dict_csv = get_better_dict("", csv_path_list,
    #                                  fill_threshold, True) 
    header_dict = dict(header_dict_converted)
    #header_dict.update(header_dict_csv)

    # note here that "directory" does NOT have a trailing "/"
    dist_mat_path = directory
    plot_mat_path = directory + "_plot.npy"
    print("We are storing the distance matrix in: ", dist_mat_path)
    
    dist_tuple = dist_mat_generator(header_dict, 
                                        dist_mat_path, overwrite)

    schema_matrix, jacc_matrix, filename_header_pairs = dist_tuple
    length = jacc_matrix.shape[0]
    
    # cluster, generate labels, plot, and generate a pdf of barcharts
    labels = agglomerative(jacc_matrix, num_clusters, 
                           filename_header_pairs, overwrite_plot)
    #plot_clusters(jacc_matrix, labels, plot_mat_path, overwrite_plot)
    #generate_barcharts(filename_header_pairs, labels, 
    #                   num_clusters, directory)

    clust_attr_lists = get_cluster_attributes(filename_header_pairs, 
                                              labels, num_clusters) 

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main() 

