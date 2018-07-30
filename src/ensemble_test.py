from create_n_ary_dataset import generate_dataset_txt
from create_n_ary_dataset import generate_dataset_csv
from create_n_ary_dataset import read_seed
from converting_utilities import convert
from test_options import load_arguments
from shuffle_directory import shuffle

import schema_clustering

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

    dest = args.dest
    n = args.num_children
    height = args.height
    num_clusters = n ** height
    overwrite_dist = '1'
    overwrite_plot = '1'
    fill_threshold = 0.4

    # define the write path for the entire program
    write_path = "../../cluster-datalake-outputs/" + dataset_name + "--output/"
    if not os.path.isdir(write_path):
        os.system("mkdir " + write_path)
    print("All results printing to " + write_path)
    
    # get absolute path 
    write_path = os.path.abspath(write_path)

    # write results to a text file
    f = open(os.path.join(write_path, 'shuffle_test_' + dataset_name + '.txt'), 'w')

    #===================================================================
    #=#: Create n-ary dataset. 
    #===================================================================

    # check if destination is valid, get its absolute path
    check_valid_dir(dest)
    dest = os.path.abspath(dest)

    # generate path to the new root of our toy dataset
    dataset_name = str(n) + "-ary_toy_dataset"
    dataset_path = os.path.join(dest, dataset_name)

    # make sure a directory with the same name doesn't already exist
    if os.path.isdir(dataset_path):
        print("This directory already exists, change the existing "
              + "directory's name, or try a different destination. ")
        exit()

    # create the directory
    if not os.path.isdir(dataset_path):
        os.system("mkdir " + dataset_path)

    word_list = read_seed()
    
    # read documentation for these functions for explanation of "+ 1"
    generate_dataset_txt(dataset_path, n, height + 1, word_list)
    generate_dataset_csv(dataset_path, n, height + 1, word_list)
    print("Should have worked. If you see a bunch of copy errors, "
          + "make sure there are no spaces between words in your "
          + "seed_words.csv file. ")
 
    #===================================================================
    #=#: Shuffle and cluster, recording the ensemble score. 
    #===================================================================
    
    # we gradually increase the proportion of the test dataset
    # which is shuffled
    shuffle_ratio = 0.0
    while shuffle_ratio <= 1.0:

        '''               
        # generate path to the new root of our toy dataset
        shuffled_dataset_name = str(n) + "-ary_test"
        shuffled_dataset_path = os.path.join(dest, shuffled_dataset_name)

        # make sure a directory with the same name doesn't already exist
        if os.path.isdir(shuffled_dataset_path):
            print("This directory already exists, change the existing "
                  + "directory's name, or try a different destination. ")
            exit()

        # create the directory
        if not os.path.isdir(shuffled_dataset_path): 
            os.system("mkdir " + shuffled_dataset_path)

        # get seed words for the dummy text and csv files
        word_list = read_seed()
        
        # read documentation for these functions for explanation of "+ 1"
        generate_dataset_txt(shuffled_dataset_path, n, height + 1, word_list)
        generate_dataset_csv(shuffled_dataset_path, n, height + 1, word_list)
        print("Should have worked. If you see a bunch of copy errors, "
              + "make sure there are no spaces between words in your "
              + "seed_words.csv file. ")
        '''

        # shuffle and convert the test dataset
        shuffle(shuffled_dataset_path, shuffle_ratio, False)
        convert(shuffled_dataset_path, num_top_exts, num_processes)

        # cluster the shuffled test dataset
        scores = schema_clustering.runflow(shuffled_dataset_path, 
                                           num_clusters, 
                                           overwrite_dist, 
                                           overwrite_plot,
                                           fill_threshold)

        # print results
        print("Shuffle ratio: ", shuffle_ratio, "Freqdrop score: ", scores[0], "Silhouette score: ", scores[1], "Naive score: ", scores[2])
        f.write("Shuffle ratio: " + format(shuffle_ratio, '.3f') 
                + " Freqdrop score: " + format(scores[0], '.3f') 
                + " Silhouette score: " + format(scores[1], '.3f')
                + " Naive score: " + format(scores[2], '.3f')
                + '\n')

        # get converted file location and output location
        out_dir = os.path.join(shuffled_dataset_path, 
                               "../" + "converted-" + shuffled_dataset_name)
        
        # define the write path for the entire program
        write_path = "../../cluster-datalake-outputs/" + shuffled_dataset_name + "--output/"
        
        # delete the shuffled dataset, outputs, and converted files
        os.system("rm -r " + write_path) 
        os.system("rm -r " + out_dir)
        os.system("rm -r " + shuffled_dataset_path)

        shuffle_ratio += args.step

    f.close()
    return

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()
