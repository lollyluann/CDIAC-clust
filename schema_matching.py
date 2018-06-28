from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import generate_token_dict
from tqdm import tqdm
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
out_dir = sys.argv[2]
dataset_dir = sys.argv[3]
overwrite = sys.argv[4]
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
check_valid_dir(dataset_dir)
xls_path = os.path.join(directory, "xls/")
xlsx_path = os.path.join(directory, "xlsx/")
csv_path = os.path.join(directory, "csv/")
tsv_path = os.path.join(directory, "tsv/")

#=========1=========2=========3=========4=========5=========6=========7=

# RETURNS: list of filenames which are candidates for conversion.

def get_valid_filenames(directory):
    if not os.path.isdir(directory):
        return []
    dir_list = os.listdir(directory)
    print("size of directory: ", len(dir_list))
    list_valid_exts = [".xls", ".xlsx", ".tsv"]
    list_caps_exts = {".XLS":".xls", ".XLSX":".xlsx", ".TSV":".tsv"}
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
            #print("out_path: ", out_path)
            print("converting")
            os.system("ssconvert " + in_path + " " + out_path + ".csv > /dev/null 2>&1 -S")

#=========1=========2=========3=========4=========5=========6=========7=

#RETURNS: a dictionary which maps filenames to csvs header lists. 
def get_header_dict(csv_dir, fill_threshold):
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
# headers in numpy array form, and the secnond being the jaccard dist
# matrix. 
def dist_mat_generator(header_dict, dist_mat_path, overwrite):
    schema_matrix = []
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

def agglomerative(jacc_matrix, num_clusters, filename_header_pairs):
    clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity='precomputed', linkage='complete')
    clustering.fit(jacc_matrix)
    labels = clustering.labels_
    print(labels)

    clust_label_dict = {}
    print("length of labels is: ", len(labels))
    print("length of filename_header_pairs is: ", len(filename_header_pairs))
    for i in range(len(labels)):
        label = labels[i]
        pair = filename_header_pairs[i]
        filename = pair[0]
        
        length = len(filename)
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
        extension = filename[dot_pos:length]
        
        








        clust_label_dict.update({filename:label})

    return clust_label_dict

#=========1=========2=========3=========4=========5=========6=========7=

def create_unconvert_dict(directory, out_dir, xls_valid_list, xlsx_valid_list, tsv_valid_list, file_path_dict):
    converted_list = os.listdir(out_dir)
    for filename in converted_list:  

                   

# MAIN PROGRAM: 
xls_valid_list = get_valid_filenames(xls_path)
xlsx_valid_list = get_valid_filenames(xlsx_path)
tsv_valid_list = get_valid_filenames(tsv_path)
convert_those_files(xls_valid_list, xls_path, out_dir)
convert_those_files(xlsx_valid_list, xlsx_path, out_dir)
convert_those_files(tsv_valid_list, tsv_path, out_dir)
csv_files_path = os.path.join(csv_path, "*.csv")
os.system("cp -R -u -p " + csv_files_path + " " + out_dir)

# gets a dictionary of the files to their paths
file_pathtokens_dict, file_path_dict = generate_token_dict.DFS(dataset_dir,1)

# if csvs have less than fill_threshold*100% nonempty cells in every row
# then we throw them out of our clustering. 
fill_threshold = 0.4
header_dict = get_header_dict(out_dir, fill_threshold)
dist_mat_path = directory[0:len(directory) - 1] + ".npy"
print("We are storing the distance matrix is the following file: ", dist_mat_path)
schema_matrix, jacc_matrix, filename_header_pairs = dist_mat_generator(header_dict, dist_mat_path, overwrite)
length = jacc_matrix.shape[0]
clust_label_dict = agglomerative(jacc_matrix, 15, filename_header_pairs)



