import matplotlib
matplotlib.use('Agg')

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from plot_dendrogram import plot_dendrogram 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.backends.backend_pdf
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from collections import Counter
from sklearn import manifold
from tqdm import tqdm
import pandas as pd
import numpy as np
import sklearn
import sys
import csv
import os

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
#check_valid_dir(dataset_dir)
xls_path = os.path.join(directory, "xls/")
xlsx_path = os.path.join(directory, "xlsx/")
csv_path = os.path.join(directory, "csv/")
tsv_path = os.path.join(directory, "tsv/")

#=========1=========2=========3=========4=========5=========6=========7=

def str_encode(string):
    return string.replace("/","@")
   
#=========1=========2=========3=========4=========5=========6=========7=

def str_decode(string):
    return string.replace("@","/")

#=========1=========2=========3=========4=========5=========6=========7=

#RETURNS: a dictionary which maps filenames to csvs header lists. 
def get_header_dict_csv(csv_path_list, fill_threshold):
    header_dict = {}
    # number of files with no valid header
    bad_files = 0
    decode_probs = 0
    for path in tqdm(csv_path_list):
        # filename is the path of the file in the orig dataset encoded
        # in the form "|home|ljung|pub8|oceans|some_file.csv"
        filename = str_encode(path)        
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
                    #print(len(header_list))
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

#RETURNS: a dictionary which maps filenames to csvs header lists. 
def get_header_dict_converted(csv_dir, fill_threshold):
    header_dict = {}
    # get a list of all files in the directory
    dir_list = os.listdir(csv_dir)
    # number of files with no valid header
    bad_files = 0
    decode_probs = 0
    for filename in tqdm(dir_list):
        # get the path of the current file
        path = os.path.join(csv_dir, filename) 
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
                    #print(len(header_list))
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

def jaccard_similarity(list1, list2):
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
def dist_mat_generator(header_dict, dist_mat_path, overwrite):
    schema_matrix = []
    # list of tuples, first element is filename, second is header_list
    filename_header_pairs = []

    for filename, header_list in header_dict.items():
        schema_matrix.append(header_list)
        filename_header_pairs.append([filename, header_list])
   
    # we just need an empty numpy array 
    jacc_matrix = np.zeros((2,1))

    if not os.path.isfile(dist_mat_path) or overwrite == "1":
        print("No existing distance matrix for this directory. ")
        print("Generating distance matrix using jaccard similarity. ")
        print("This could take a while... ")

        # we generate the distance matrix as a list
        dist_mat_list = []
        # iterating over the header array once...
        for header_a in tqdm(schema_matrix):
            # storing distances for a single header
            single_row = []
            # iterating again...
            for header_b in schema_matrix:
                jacc = jaccard_similarity(header_a, header_b)
                single_row.append(jacc)
            # add one row to the list
            dist_mat_list.append(np.array(single_row))
        # convert list to numpy array
        jacc_matrix = np.array(dist_mat_list)
        jacc_matrix = np.stack(jacc_matrix, axis=0)
        print(jacc_matrix.shape)
        # save on disk, because computation is expensive
        np.save(dist_mat_path, jacc_matrix)

    else:
        jacc_matrix = np.load(dist_mat_path)

    return schema_matrix, jacc_matrix, filename_header_pairs

#=========1=========2=========3=========4=========5=========6=========7=

# RETURNS: index of "char" in "string". Returns the length of the string
# if there are no instances of that character. 
def find_char_from_end(string, char):
    
        length = len(string)
        # we iterate on the characters starting from end of the string
        pos = length - 1
        # dot pos will be position of the first period from the end,
        # i.e. the file extension dot. If it is still "length" at end,
        # we know that there are no dots in the filename. 
        dot_pos = length
        while (pos >= 0):
            if (filename[pos] == char):
                dot_pos = pos
                break
            pos = pos - 1
        return dot_pos

#=========1=========2=========3=========4=========5=========6=========7=

def agglomerative(jacc_matrix, num_clusters, filename_header_pairs):
    clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity='precomputed', linkage='complete')
    clustering.fit(jacc_matrix)
    labels = clustering.labels_
    print(labels)

    #plt.figure(figsize=(17,9))
    #plot_dendrogram(clustering, labels = clustering.labels_)
    
    #plt.savefig("dendrogram", dpi=300)
    #print("dendrogram written to \"dendrogram.png\"")
    
    '''
    # we have a trash cluster that we use for anything that the header
    # reader didn't like, so we'll actually have one more cluster that
    # will be passed in as a directory. 
    num_clusters = num_clusters + 1

    cluster_path_bins = []
    for i in range(num_clusters):
        cluster_path_bins.append([])

    clust_label_dict = {}
    print("length of labels is: ", len(labels))
    print("length of filename_header_pairs is: ", len(filename_header_pairs))
    for i in range(len(labels)):
        label = labels[i]
        pair = filename_header_pairs[i]
        filename = pair[0]
        
        dot_pos = find_char_from_end(filename, ".")
        
        # recall that these filenames are the converted filenames. 
        # we want to be able to map back to the original filenames so
        # we can find the directory using the filename_path_dict from
        # generate_token_dict to get the path.
        # converted files will be of the form
        # <filename>.<orig_ext>.csv.<i> 
        extension = filename[dot_pos:length]
        # if the file was converted from some other type...
        if (extension != ".csv"):
            # <filename>.<orig_ext>.csv
            filename_no_num = filename[0:dot_pos]
            second_dot_pos = find_char_from_end(filename_no_num, ".")
            # <filename>.<orig_ext>
            orig_filename = filename_no_num[0:second_dot_pos]
            # here we'll convert the filename to its original path
            # in the dataset. 
            orig_path = "<INSERT_PATH_PARSE_FUNC_HERE>"
            # add the path to the appropriate list, one list per
            # cluster, then later on we'll count frequencies in 
            # these lists to create the histograms. 
            cluster_path_bins[label].append(orig_path)

        clust_label_dict.update({filename:label})

    return clust_label_dict 
    '''

    return labels

#=========1=========2=========3=========4=========5=========6=========7=

def create_unconvert_dict(directory, out_dir, xls_valid_list, xlsx_valid_list, tsv_valid_list, file_path_dict):
    converted_list = os.listdir(out_dir)
    for filename in converted_list:  
        print("")
    return

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

# DOES: generates barcharts which show the distribution of unique
#       filepaths in a cluster. 
def generate_barcharts(filename_header_pairs, labels, num_clusters):
    # create a dict mapping cluster indices to lists of filepaths
    cluster_filepath_dict = {}
    # list of lists, each list is full of the filepaths for one cluster.
    list_cluster_lists = []
    # initialize each child list. 
    for k in range(num_clusters):
        list_cluster_lists.append([])
    # for each label in labels
    for i in tqdm(range(len(labels))):
        # get the corresponding filename
        filename_header_pair = filename_header_pairs[i]
        filename = filename_header_pair[0]
        at_loc = 0
        j = len(filename) - 1
        while (j >= 0):
            if (filename[j] == "@"):
                at_loc = j
                break
            j = j - 1
        filepath = filename[0:at_loc]
        decoded_filepath = str_decode(filepath)
        #print("filename is: ", filepath)
        # add it to the appropriate list based on the label
        list_cluster_lists[labels[i]].append(decoded_filepath)
     
    matplotlib.rcParams.update({'font.size': 6})
    # for each cluster
    for k in range(num_clusters):

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
    # print(ext_to_paths_dict)
    csv_path_list = []
    if "csv" in ext_to_paths_dict:
        csv_path_list = ext_to_paths_dict["csv"]
    
    # if csvs have less than fill_threshold*100% nonempty cells in every row
    # then we throw them out of our clustering. 
    fill_threshold = 0.4
    
    # we have two dicts, one made up of files which were converted to
    # csv format, and the other made up of files that were in csv
    # format originally. we concatenate both dicts into "header_dict".
    header_dict_converted = get_header_dict_converted(out_dir, fill_threshold)
    header_dict_csv = get_header_dict_csv(csv_path_list, fill_threshold)
    header_dict = dict(header_dict_converted)
    header_dict.update(header_dict_csv)

    dist_mat_path = directory[0:len(directory) - 1] + ".npy"
    plot_mat_path = directory[0:len(directory) - 1] + "_plot.npy"
    print("We are storing the distance matrix is the following file: ", dist_mat_path)
    schema_matrix, jacc_matrix, filename_header_pairs = dist_mat_generator(header_dict, dist_mat_path, overwrite)
    length = jacc_matrix.shape[0]

    labels = agglomerative(jacc_matrix, num_clusters, filename_header_pairs)
    #plot_clusters(jacc_matrix, labels, plot_mat_path, overwrite_plot)
    generate_barcharts(filename_header_pairs, labels, num_clusters)

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main() 

