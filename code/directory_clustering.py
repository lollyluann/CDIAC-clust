%%time 

import locale
import glob
import os.path
import requests
import tarfile
import sys
import codecs
from smart_open import smart_open
import re

def doc2vec(dataset_path, write_path, csv_path_list):

    locale.setlocale(locale.LC_ALL, 'C')
    all_lines = []

    if sys.version > '3':
        control_chars = [chr(0x85)]
    else:
        control_chars = [unichr(0x85)]

    # Convert text to lower-case and strip punctuation/symbols from words
    def normalize_text(text):
        norm_text = text.lower()
        # Replace breaks with spaces
        norm_text = norm_text.replace('<br />', ' ')
        # Pad punctuation with spaces on both sides
        norm_text = re.sub(r"([\.\",\(\)!\?;:])", " \\1 ", norm_text)
        return norm_text

    if not os.path.isfile(os.path.join(write_path,'alldata-id.txt')):
       # DFS goes here 


        else:
            print("IMDB archive directory already available without download.")

        # Collect & normalize test/train data
        print("Cleaning up dataset...")
        folders = ['train/pos', 'train/neg', 'test/pos', 'test/neg', 'train/unsup']
        for fol in folders:
            temp = u''
            newline = "\n".encode("utf-8")
            output = fol.replace('/', '-') + '.txt'
            # Is there a better pattern to use?
            txt_files = glob.glob(os.path.join(dirname, fol, '*.txt'))
            print(" %s: %i files" % (fol, len(txt_files)))
            with smart_open(os.path.join(dirname, output), "wb") as n:
                for i, txt in enumerate(txt_files):
                    with smart_open(txt, "rb") as t:
                        one_text = t.read().decode("utf-8")
                        for c in control_chars:
                            one_text = one_text.replace(c, ' ')
                        one_text = normalize_text(one_text)
                        all_lines.append(one_text)
                        n.write(one_text.encode("utf-8"))
                        n.write(newline)

        # Save to disk for instant re-use on any future runs
        with smart_open(os.path.join(dirname, 'alldata-id.txt'), 'wb') as f:
            for idx, line in enumerate(all_lines):
                num_line = u"_*{0} {1}\n".format(idx, line)
                f.write(num_line.encode("utf-8"))

    assert os.path.isfile("aclImdb/alldata-id.txt"), "alldata-id.txt unavailable"
    print("Success, alldata-id.txt is available for next steps.")



# MAIN PROGRAM:
def runflow(dataset_path, num_clusters, 
            overwrite, overwrite_plot):
   
    #===================================================================
    #=#BLOCK#=#: Get read and write paths for cluster functions 
    #===================================================================
    print("Getting read and write paths for cluster functions. ")  
 
    if overwrite == 'y' or overwrite == 'Y':
        overwrite = "1"
    if overwrite_plot == 'y' or overwrite_plot == 'Y':
        overwrite_plot = "1"
 
    # check if the dataset location is a valid directory 
    check_valid_dir(dataset_path)
   
    # get its absolute path
    dataset_path = os.path.abspath(dataset_path)
    
    # the name of the top-level directory of the dataset
    dataset_name = get_last_dir_from_path(dataset_path)
     
    # define the write path for the entire program
    write_path = "../../cluster-datalake-outputs/" + dataset_name + "--output/"
    if not os.path.isdir(write_path):
        os.system("mkdir " + write_path)
    print("All results printing to " + write_path)
    
    # get absolute paths 
    write_path = os.path.abspath(write_path)
    
    # get the location of the extension index file
    print("Finding extension index file. ")
    ext_dict_file_loc = os.path.join(write_path, "extension_index_"
                                     + dataset_name + ".npy")
    
    # check if the above paths are valid
    check_valid_file(ext_dict_file_loc)
    
    # load the extension to path dict
    ext_to_paths_dict = np.load(ext_dict_file_loc).item()
    csv_path_list = []
    if "csv" in ext_to_paths_dict:
        csv_path_list = ext_to_paths_dict["csv"]
 
    # location of files converted to csv format
    csv_dir = os.path.join(out_dir, "csv/")

    return

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

def main():
    
    arg_list = parse_args()
    dataset_path = arg_list[0]
    num_clusters = arg_list[1]
    overwrite = arg_list[2]
    overwrite_plot = arg_list[3]
    fill_threshold = arg_list[4]

    print("Don't run this file standalone. ")
    runflow(dataset_path, num_clusters, 
                    overwrite, overwrite_plot, fill_threshold)    
    return

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main() 


