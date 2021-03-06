from calculate_file_distances import compute_naive_score
from frequencydrop import compute_freqdrop_score
from silhouette import compute_silhouette
from collections import Counter
from tqdm import tqdm

from matplotlib.backends.backend_pdf import PdfPages

from matplotlib import pyplot as plt

import get_cluster_stats as get_stats
import pandas as pd

import path_utilities
import matplotlib
import pickle
import os

# DOES: generates barcharts which show the distribution of unique
#       filepaths in a cluster, prints these as well as stats to a pdf,
#       and also prints the info to a text file as well.  
def generate_results(filename_header_pairs, labels, num_clusters, 
                       dataset_path, write_path, dataset_name):

    #===================================================================
    #=#BLOCK#=#: Generates two data structures: 
    #            "list_cluster_lists": list of lists, each list contains 
    #            the filepaths for one cluster.
    #            "cluster_directories": list of dicts, one per cluster, 
    #            keys are unique directories, values are counts
    #===================================================================
    print("Creating list of filepaths for each cluster. ")
    print("Creating list of dicts which "
          + "map directories to frequencies. ") 
 
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
        common_prefix = path_utilities.remove_path_end(dataset_path)
        
        # remove the common prefix for display on barchart. The " - 1"
        # is so that we include the leading "/". 
        len_pre = len(common_prefix)
        len_decod = len(decoded_filepath)
        decoded_filepath_trunc = decoded_filepath[len_pre - 1:len_decod]
        
        # add it to the appropriate list based on the label
        list_cluster_lists[labels[i]].append(decoded_filepath_trunc)   

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
    
    #===================================================================
    #=#BLOCK#=#: Prints cluster information to .pdf and .txt files.  
    #===================================================================
    print("Printing cluster info to .txt and .pdf files. ")   
 
    # get a list of the cluster statistic for printing to pdf
    cluster_stats = get_stats.get_cluster_stats(cluster_directories)
    
    # compute silhouette coefficients for each cluster (sil_list)
    # and for the entire clustering (sil)
    sil, sil_list = compute_silhouette(cluster_directories,dataset_path)
    l = 0
    for coeff in sil_list:
        # print("Silhouette score for cluster " + str(l)+": "+str(coeff))
        l += 1
    print("Total silhouette for entire clustering: ", sil)

    # get the frequency drop score of the clusters
    fd_scores, fd_total = compute_freqdrop_score(cluster_directories) 
    freqdrop_total = fd_total 
    freqdrop_scores = fd_scores

    # get the naive tree dist score of the clusters
    td_scores, td_total = compute_naive_score(list_cluster_lists,
                                              cluster_directories)
 
    # just make font a bit smaller
    matplotlib.rcParams.update({'font.size': 4})
    print("\n\nGenerating barcharts...")
    
    # open the pdf and text files for writing 
    pdf_path = os.path.join(write_path, "structured_stats_" + dataset_name 
                            + "_k=" + str(num_clusters) + ".pdf")
    txt_path = os.path.join(write_path, "structured_stats_" + dataset_name 
                            + "_k=" + str(num_clusters) + ".txt")
    pkl_path = os.path.join(write_path, "histogram_data_" + dataset_name 
                            + "_k=" + str(num_clusters) + ".pkl")
    pdf = PdfPages(pdf_path)
    f = open(txt_path,'w')

    # save list_cluster_lists to a pkl file
    with open(pkl_path, 'wb') as filehandle:      
        pickle.dump(list_cluster_lists, filehandle)
 
    # for each cluster
    for k in range(num_clusters):
        single_cluster_stats = cluster_stats[k]
        
        #fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(10, 20))
        #plt.figure(k) 
        plt.clf()

        # get frequencies of the paths
        path_counts = Counter(list_cluster_lists[k])
        
        # Create a dataframe from path_counts        
        df = pd.DataFrame.from_dict(path_counts, orient='index')
        
        # rename the frequency axis
        df = df.rename(columns={ df.columns[0]: "freqs" })
        
        # sort it with highest freqs on top
        sorted_df = df.sort_values("freqs",ascending=False)
        top_10_slice = sorted_df.head(10)
        top_10_slice.plot(kind='bar')
        
        # leave enough space for x-axis labels
        # fig.subplots_adjust(hspace=7)

        fig_title = ("Directory distribution for cluster "+str(k)+"\n"
        +"Number of unique directories: " 
        +str(single_cluster_stats[0])+"\n"
        +"Mean frequency: "+str(single_cluster_stats[1])+"\n"
        +"Median frequency: "+str(single_cluster_stats[3])+"\n"
        +"Standard deviation of frequencies: " 
        +str(single_cluster_stats[2])+"\n"
        +"Closest common ancestor of all directories: " 
        +single_cluster_stats[4] + "\n"
        +"Silhouette score: " + str(sil_list[k]) + "\n"
        + "Frequency drop score: " + str(freqdrop_scores[k]))
        plt.title(fig_title)
        plt.xlabel('Directory')
        plt.ylabel('Quantity of files in directory')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.38, top=0.87)
        pdf.savefig(plt.gcf())

        # print to .txt file as well
        f.write(fig_title)
        f.write("\n\n")
  
    # setting ensemble to just freqdrop 
    ensemble_score = ((sil+1)/2 + freqdrop_total)/2
    scores = []
    sil = (sil + 1) / 2
    scores.append(freqdrop_total)
    scores.append(sil)
    scores.append(td_total)
    f.write("Total_silhouette: " + str(sil))
    f.write("Total_frequency drop: " + str(freqdrop_total))
    f.write("Total ensemble score: " + str(ensemble_score))
    f.write("Total naive score: " + str(td_total))
    f.close()
    pdf.close()
    return list_cluster_lists, scores
