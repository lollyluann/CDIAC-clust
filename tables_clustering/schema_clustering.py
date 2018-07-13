import matplotlib
matplotlib.use('Agg')

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from plot_dendrogram import plot_dendrogram 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.backends.backend_pdf
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
import calculate_file_distances
from collections import Counter
from sklearn import manifold
import get_cluster_stats
from tqdm import tqdm
import path_utilities
import pandas as pd
import numpy as np
import silhouette
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

#RETURNS: a dictionary which maps filenames to csvs header lists. 
def get_header_dict(csv_dir, csv_path_list, fill_threshold, converted_status):
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
                    # probably just a descriptor of the table, so try the
                    # next line. 
                    header_list = next(reader)
                    num_nonempty = 0
                    for attribute in header_list:
                        if not (attribute == ""):
                            num_nonempty = num_nonempty + 1
                    if (len(header_list) == 0):
                        fill_ratio = -1
                    else:
                        fill_ratio = num_nonempty / len(header_list)

                    #================================================
                    # Here we've hardcoded some information about 
                    # scientific data to work better with CDIAC. 
                    # feel free to remove it. 
                    
                    # people seem to denote pre-header stuff with a *
                    for attribute in header_list:
                        if (attribute != "" and attribute[-1] == "*"):
                            fill_ratio = -1
                    if (len(header_list) > 3):
                        if (header_list[0] == "Year" and header_list[2] != ""):
                            break
                        if (header_list[0] == "Citation"):
                            fill_ratio = -1
                        
                    #================================================
            except UnicodeDecodeError:
                decode_probs = decode_probs + 1                    
            except StopIteration:
                bad_files = bad_files + 1
                #os.system("cp " + path + " ~/bad_csvs/")
                continue
            # throw a key value pair in the dict, with filename as key
            header_dict.update({filename:header_list})
    print("Throwing out this number of files, all have less than ", fill_threshold*100, "% nonempty cells in every row: ", bad_files)    
    print("Number of UnicodeDecodeErrors: ", decode_probs)
    return header_dict

#=========1=========2=========3=========4=========5=========6=========7=

def jaccard_distance(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    #print(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    if (union == 0):
        union = 1
    return 1 - float(intersection / union)
    
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

    if not os.path.isfile(dist_mat_path) or not os.path.isfile(headpairs_path) or overwrite == "1":
        print("No existing cached files for this directory. ")
        print("Generating distance matrix using jaccard similarity. ")
        print("This could take a while... ")
        # we generate the distance matrix as a list
        dist_mat_list = []
        # iterating over the header array once...
        for header_a in tqdm(schema_matrix):
            #===========================
            #print(header_a)
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

def agglomerative(jacc_matrix, num_clusters, filename_header_pairs, overwrite):
    clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity='precomputed', linkage='complete')
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

# DOES: plots the schema_clusters for the csv files. 
def plot_clusters(jacc_matrix, labels, plot_mat_path, overwrite_plot):

    if not os.path.isfile(plot_mat_path) or overwrite_plot == "1":
        # multidimensional scaling to convert distance matrix into 3 dimensions
        mds = manifold.MDS(n_components=3, n_jobs=4, dissimilarity="precomputed", random_state=1, verbose=2)
        print("Fitting to the distance matrix. ")
        pos = mds.fit_transform(jacc_matrix)  # shape (n_components, n_samples)
        np.save(plot_mat_path,pos)
    else:
        pos = np.load(plot_mat_path)
    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]

    # set up plot
    print("Setting up plot. ")
    fig = plt.figure(figsize=(17,9))
    ax = Axes3D(fig)

    # create data frame with MDS results, cluster numbers, and filenames
    df = pd.DataFrame(dict(x=xs, y=ys, z=zs, label=labels)) 
    # group by cluster
    groups = df.groupby('label')

    # for each cluster, plot the files in that cluster
    for name, group in tqdm(groups):
        # color = ('#%06X' % random.randint(0,256**3-1))
        color = np.random.rand(3,)
        for t in range(group.shape[0]):
            ax.scatter(group.x.iloc[t], group.y.iloc[t], group.z.iloc[t], 
                c=color, marker='o')
            ax.set_aspect('auto')

    plt.savefig("3D_schema_cluster", dpi=300)
    print("scatter plot written to \"3D_schema_cluster.png\"")
    return

#=========1=========2=========3=========4=========5=========6=========7=

def tree_dist(path1, path2):
    return calculate_file_distances.path_dist(path1, path2)

#=========1=========2=========3=========4=========5=========6=========7=

#RETURNS: Silhouette coefficient of clustering. 
def compute_silhouette(filename_header_pairs, labels):
    path_list = []
    for pair in filename_header_pairs:
        filename = pair[0]
        path_list.append(filename)
    paths_array = np.array(path_list)
    return silhouette.silhouette_score_block(paths_array, labels, metric=tree_dist)

#=========1=========2=========3=========4=========5=========6=========7=

# DOES: generates barcharts which show the distribution of unique
#       filepaths in a cluster. 
def generate_barcharts(filename_header_pairs, labels, num_clusters, root_path):
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
        decoded_filepath_trunc = decoded_filepath[len(common_prefix) - 1:len(decoded_filepath)]
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
    cluster_stats = get_cluster_stats.get_cluster_stats(cluster_directories)
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
        figure_title = "Directory distribution for cluster " + str(k) + "\n"
        figure_title += "Number of unique directories: " + str(single_cluster_stats[0]) + "\n"
        figure_title += "Mean frequency: " + str(single_cluster_stats[1]) + "\n"
        figure_title += "Median frequency: " + str(single_cluster_stats[3]) + "\n"
        figure_title += "Standard deviation of frequencies: " + str(single_cluster_stats[2]) + "\n"
        figure_title += "Closest common ancestor of all directories: " + single_cluster_stats[4]
        plt.title(figure_title, y=1.08)
        plt.xlabel('Directory')
        plt.ylabel('Quantity of files in directory')
        plt.tight_layout()
    pdf = matplotlib.backends.backend_pdf.PdfPages("tabular_barcharts.pdf")
    for fig in range(1, plt.gcf().number + 1): ## will open an empty extra figure :(
        pdf.savefig( fig )
    pdf.close()
 
#=========1=========2=========3=========4=========5=========6=========7=

def main():

    # MAIN PROGRAM:
    num_clusters = 15
    ext_dict_file_loc = os.path.join(ext_dict_dir, "extension_index.npy")
    ext_to_paths_dict = np.load(ext_dict_file_loc).item()
    csv_path_list = []
    if "csv" in ext_to_paths_dict:
        csv_path_list = ext_to_paths_dict["csv"]
    
    # if csvs have less than fill_threshold*100% nonempty cells in every row
    # then we throw them out of our clustering. 
    fill_threshold = 0.4
    
    # we have two dicts, one made up of files which were converted to
    # csv format, and the other made up of files that were in csv
    # format originally. we concatenate both dicts into "header_dict".
    header_dict_converted = get_header_dict(out_dir, [],  fill_threshold, True)
    header_dict_csv = get_header_dict("", csv_path_list, fill_threshold, False)
    header_dict = dict(header_dict_converted)
    header_dict.update(header_dict_csv)

    # note here that "directory" does NOT have a trailing "/"
    dist_mat_path = directory
    plot_mat_path = directory + "_plot.npy"
    print("We are storing the distance matrix in the following file: ", dist_mat_path)
    schema_matrix, jacc_matrix, filename_header_pairs = dist_mat_generator(header_dict, dist_mat_path, overwrite)
    length = jacc_matrix.shape[0]
    
    # cluster, generate labels, plot, and generate a pdf of barcharts
    labels = agglomerative(jacc_matrix, num_clusters, filename_header_pairs, overwrite_plot)
    #plot_clusters(jacc_matrix, labels, plot_mat_path, overwrite_plot)
    generate_barcharts(filename_header_pairs, labels, num_clusters, directory)
    sil_coeff = compute_silhouette(filename_header_pairs, labels)
    print("Silhouette coefficient: ", sil_coeff)

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main() 

