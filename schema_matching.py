from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm
import numpy as np
import sklearn
import sys
import csv
import os

# Converts all the .xls or .xlsx files in a directory to .csv files. 

#=========1=========2=========3=========4=========5=========6=========7=

# ARGUMENTS
# source directory and output directory
directory = sys.argv[1]
out_dir = sys.argv[2]

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

# grab the contents of the directory
dir_list = os.listdir(directory)

#=========1=========2=========3=========4=========5=========6=========7=

# RETURNS: list of filenames which are candidates for conversion.

def get_valid_filenames(directory):
    dir_list = os.listdir(directory)
    list_valid_exts = [".xls", ".xlsx"]
    list_caps_exts = {".XLS":".xls", ".XLSX":".xlsx"}
    valid_list = []
    # for each filename in the directory...
    for filename in tqdm(dir_list):
        length = len(filename)
        valid = False
        # we iterate on the characters starting from end of the string
        pos = length - 1
        # dot pos will be position of the first period from the end,
        # i.e. the file extension dot. If it is still "length" at end,
        # we know that there are no dots in the filename. 
        dot_pos = length
        while (pos >= 0):
            if (filename[pos] == "."):
                dot_pos = pos
                break
            pos = pos - 1
        # if there is a dot somewhere in filename, i.e. if there is an
        # extension...
        if (dot_pos < length):
            extension = filename[dot_pos:length]
            if extension in list_valid_exts:
                valid_list.append(filename)
                valid = True
            # if the extension is an ALLCAPS version...
            elif extension in list_caps_exts.keys():
                new_filename = filename[0:dot_pos] + list_caps_exts[extension]
                # change it to lowercase and add it to valid_list
                os.system("mv " + os.path.join(directory, filename) + " " + os.path.join(directory, new_filename))
                valid_list.append(new_filename)
                valid = True
        if (valid == False):
            print(extension)
            print("This filename is invalid: ", filename)
        
    print("There are ", len(valid_list), " candidates for conversion. ")
    return valid_list

#=========1=========2=========3=========4=========5=========6=========7=

# DOES: converts all the files in valid list to csv, and puts the
# resultant files in out_dir. 
def convert_those_files(valid_list, directory, out_dir):
    for filename in tqdm(valid_list):
        # getting the filename without file extension
        length = len(filename)
        pos = length - 1
        dot_pos = length
        while (pos >= 0):
            if (filename[pos] == "."):
                dot_pos = pos
                break
            pos = pos - 1
        fn_no_ext = filename[0:dot_pos]

        # converting
        in_path = os.path.join(directory, filename)
        out_path = os.path.join(out_dir, fn_no_ext)
        if not os.path.isfile(out_path + ".csv.0"):
            print("converting")
            os.system("ssconvert " + in_path + " " + out_path + ".csv > /dev/null 2>&1 -S")

#=========1=========2=========3=========4=========5=========6=========7=

#RETURNS: a dictionary which maps filenames to csvs header lists. 
def get_header_dict(csv_dir):
    header_dict = {}
    # get a list of all files in the directory
    dir_list = os.listdir(csv_dir)
    # number of files with no valid header
    bad_files = 0
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
                while (fill_ratio <= 0.4):
                    # if there's only one nonempty attribute, it's
                    # probably just a descriptor of the table, so try the
                    # next line. 
                    header_list = next(reader)
                    num_nonempty = 0
                    for attribute in header_list:
                        if not (attribute == ""):
                            num_nonempty = num_nonempty + 1
                    fill_ratio = num_nonempty / len(header_list)
                    # people seem to denote pre-header stuff with a *
                    for attribute in header_list:
                        if (attribute != "" and attribute[-1] == "*"):
                            fill_ratio = -1
                    if (header_list[0] == ""):
                        fill_ratio = -1
            except StopIteration:
                bad_files = bad_files + 1
                continue
            # throw a key value pair in the dict, with filename as key
            header_dict.update({filename:header_list})
    print("Throwing out this number of files, all have less than 10% nonempty cells in every row: ", bad_files)    
    return header_dict

#=========1=========2=========3=========4=========5=========6=========7=

def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    #print(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection / union)
    
#=========1=========2=========3=========4=========5=========6=========7=

# DOES: computes the jaccard distance matrix of the headers in 
# header_dict. 
# RETURNS: a tuple with the first element being an array of all the 
# headers in numpy array form, and the secnond being the jaccard dist
# matrix. 
def dist_mat_generator(header_dict, dist_mat_path):
    list_of_headers = []
    # we're keeping track of the max number of attributes in any header
    # so we know how big to make the second axis of the numpy array
    max_attributes = 0
    # for each key value pair, maps filenames to headers as lists of 
    # strings...
    for filename, header_list in header_dict.items():
        # just find the max_attributes value
        if (len(header_list) > max_attributes):
            max_attributes = len(header_list)
    # again for each key value pair...
    for filename, header_list in header_dict.items():
        length = len(header_list)
        print(header_list)
        diff = max_attributes - length
        # add in empty strings until all headers have max length
        for x in range(diff):
            header_list.append("")
        # add each header to a list as a numpy array
        list_of_headers.append(np.array(header_list))
    # convert list of numpy arrays to numpy array 
    schema_matrix = np.array(list_of_headers)
    schema_matrix = np.stack(schema_matrix, axis=0)
    print(schema_matrix.shape)
    num_headers = schema_matrix.shape[0]
    
    jacc_matrix = np.zeros((2,1))

    if not os.path.isfile(dist_mat_path):
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


    return schema_matrix, jacc_matrix

#=========1=========2=========3=========4=========5=========6=========7=

def agglomerative(jacc_matrix, num_clusters):
    clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity='precomputed', linkage='average')
    clustering.fit(jacc_matrix)
    labels = clustering.labels_
    print(labels)

#=========1=========2=========3=========4=========5=========6=========7=

# MAIN PROGRAM: 
valid_list = get_valid_filenames(directory)
convert_those_files(valid_list, directory, out_dir)
header_dict = get_header_dict(out_dir)
dist_mat_path = directory[0:len(directory) - 1] + ".npy"
print(dist_mat_path)
schema_matrix, jacc_matrix = dist_mat_generator(header_dict, dist_mat_path)
print(jacc_matrix)
agglomerative(jacc_matrix, 10)

#labels = spectral_clustering(NN_matrix, n_clusters=num_clusters, eigen_solver='arpack')
#print(labels)


