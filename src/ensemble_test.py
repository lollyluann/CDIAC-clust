from create_n_ary_dataset import generate_dataset_txt
from create_n_ary_dataset import generate_dataset_csv
from create_n_ary_dataset import read_seed

from path_utilities import get_last_dir_from_path 
from converting_utilities import convert
from test_options import load_arguments
from shuffle_directory import shuffle

import schema_clustering
import DFS

import multiprocessing
import sys
import os

#=========1=========2=========3=========4=========5=========6=========7=

def check_valid_dir(some_dir):
    if not os.path.isdir(some_dir):
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

#=========1=========2=========3=========4=========5=========6=========7=

def main():

    print("ARGUMENTS: ")
    args = load_arguments()
    print("Arguments loaded. ")

    dataset_path = args.dataset_path
    dest = os.path.join(dataset_path, "../")
    num_clusters = args.num_clusters
    num_top_exts = args.num_extensions
    num_processes = args.num_processes
    overwrite_dist = 'y'
    overwrite_plot = 'y'
    fill_threshold = 0.4
    
    # check if destination is valid, get its absolute path
    check_valid_dir(dest)
    dest = os.path.abspath(dest)

    # check if dataset is valid, get its absolute path
    check_valid_dir(dataset_path)
    dataset_path = os.path.abspath(dataset_path)
    
    # the name of the top-level directory of the dataset
    dataset_name = get_last_dir_from_path(dataset_path)

    # define the write path for the entire program
    write_path = "../../cluster-datalake-outputs/" + dataset_name + "--output/"
    if not os.path.isdir(write_path):
        os.system("mkdir " + write_path)
    print("All results printing to " + write_path)
    
    # get absolute path 
    write_path = os.path.abspath(write_path)

    # write results to a text file
    f = open(os.path.join(write_path, 'shuffle_test_' + dataset_name + '.txt'), 'w')
    f.write("shuffle_ratio" + "," + "freqdrop_score" + ","
            + "silhouette_score" + "," + "naive_tree_dist_score" 
            + "," + "\n")

    #===================================================================
    #=#: Shuffle and cluster, recording the ensemble score. 
    #===================================================================

    shuffle_tracker = []

    # get a list of the paths to every file in the dataset
    # rooted at "dataset_path"
    filepaths = DFS.DFS(dataset_path)

    # generate path to the new root of our test dataset
    shuffled_dataset_name = "shuffled_" + dataset_name
    shuffled_dataset_path = os.path.join(dest, shuffled_dataset_name)
    print("clustering: ", shuffled_dataset_path)
   
    # copy dataset to this new location
    os.system("cp -r " + dataset_path + " " + shuffled_dataset_path) 
    
    # we gradually increase the proportion of the test dataset
    # which is shuffled
    shuffle_ratio = 0.0
    while shuffle_ratio <= 1.0:

        # define the write path for the entire program
        write_path = "../../cluster-datalake-outputs/" + shuffled_dataset_name + "--output/"
       
        # get converted file location and output location
        out_dir = os.path.join(shuffled_dataset_path, 
                               "../" + "converted-" + shuffled_dataset_name)


        if not os.path.isdir(write_path):
            os.system("mkdir " + write_path)
        
        if not os.path.isdir(out_dir):
            os.system("mkdir " + out_dir)
        csv_path = os.path.join(out_dir, "csv/")
        if not os.path.isdir(csv_path):
            os.system("mkdir " + csv_path)
        txt_path = os.path.join(out_dir, "txt/")
        if not os.path.isdir(txt_path):
            os.system("mkdir " + txt_path)


        # shuffle and convert the test dataset
        shuffle_tracker = shuffle(shuffled_dataset_path, shuffle_ratio, False, shuffle_tracker, filepaths)
        DFS.extension_indexer(shuffled_dataset_path, num_top_exts, write_path)

        # cluster the shuffled test dataset
        scores = schema_clustering.runflow(shuffled_dataset_path, 
                                           num_clusters, 
                                           overwrite_dist, 
                                           overwrite_plot,
                                           fill_threshold)

        # print results
        print("Shuffle ratio: ", shuffle_ratio, "Freqdrop score: ", scores[0], "Silhouette score: ", scores[1], "Naive score: ", scores[2])
        f.write(format(shuffle_ratio, '.3f') + ","
                + format(scores[0], '.3f') + ","
                + format(scores[1], '.3f') + ","
                + format(scores[2], '.3f') + ","
                + '\n')

        
        
        # delete the shuffled dataset, outputs, and converted files
        os.system("rm -r " + write_path) 
        os.system("rm -r " + out_dir) 

        shuffle_ratio += args.step

    f.close()
    return

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()
