from path_utilities import get_last_dir_from_path 
from gensim.models.doc2vec import TaggedDocument
from collections import namedtuple
from smart_open import smart_open
from tqdm import tqdm

import numpy as np

import requests
import os.path
import tarfile
import codecs
import gensim
import locale
import glob
import sys
import re

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

def parse_args():

    print("Parsing arguments. ")
    # ARGUMENTS    
    # source directory and output directory
    dataset_path = sys.argv[1]          # directory of dataset
    num_clusters = 0                    # number of clusters to generate
    fill_threshold = 1                  # ignore rows filled less
    overwrite = "0"                     # overwrite the distance matrix
    overwrite_plot = "0"                # overwrite plot cache 
    # overwrite is a string, should be "0" for don't overwrite, and "1"
    # for do
    arg_list = [
                dataset_path, 
                num_clusters, 
                overwrite, 
                overwrite_plot, 
                fill_threshold,
               ]
    print("Arguments parsed. ")
    return arg_list

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

def check_valid_file(some_file):
    if not os.path.isfile(some_file):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("")
        print("DIES IST KEIN GÜLTIGER SPEICHERORT FÜR DATEIEN!!!!")
        print("")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        exit()

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

def doc2vec(dataset_path, dataset_name, write_path, txt_path_list):

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

    alldata_path = os.path.join(write_path, 'alldata-id_' + dataset_name + '.txt') 

    if not os.path.isfile(alldata_path):

        # Collect & normalize test/train data
        print("Cleaning up dataset...")
            
        # list of the absolute paths of every text file
        print(" %i files" % (len(txt_path_list)))
 
        # for each file "txt"
        for i, txt in tqdm(enumerate(txt_path_list)):
            with smart_open(txt, "rb") as t:

                try: 
                    # "one_text" is the whole document
                    one_text = t.read().decode("utf-8")
                    for c in control_chars:
                        one_text = one_text.replace(c, ' ')
                    one_text = normalize_text(one_text)
                    all_lines.append(one_text)
                except UnicodeDecodeError:
                    
                    # we skip this file, but we need to preserve index pos
                    all_lines.append(" ")
                    continue

        # Save to disk for instant re-use on any future runs
        with smart_open(alldata_path, 'wb') as f:
            for idx, line in enumerate(all_lines):
                num_line = u"_*{0} {1}\n".format(idx, line)
                f.write(num_line.encode("utf-8"))

    assert os.path.isfile(alldata_path), "alldata unavailable"
    print("Success, alldata is available for next steps.")

    #===================================================================
    #=#BLOCK#=#: Read in alldata 
    #===================================================================

    # this data object class suffices as a `TaggedDocument` 
    # (with `words` and `tags`)
    # plus adds other state helpful for our later evaluation/reporting
    SentimentDocument = namedtuple('SentimentDocument', 
                                   'words tags split sentiment')

    alldocs = []
    with smart_open(alldata_path, 'rb', encoding='utf-8') as alldata:
        for line_no, line in tqdm(enumerate(alldata)):
            tokens = gensim.utils.to_unicode(line).split()
            words = tokens[1:]
            
            # 'tags = [tokens[0]]' would also work at extra memory cost
            tags = [line_no] 
            
            # 25k train, 25k test, 25k extra
            split = 'train'
  
            
            # [12.5K pos, 12.5K neg]*2 then unknown
            sentiment = 1.0
            alldocs.append(SentimentDocument(words, tags, split, sentiment))

    train_docs = [doc for doc in alldocs if doc.split == 'train']
    test_docs = [doc for doc in alldocs if doc.split == 'test']

    print('%d docs: %d train-sentiment, %d test-sentiment' % 
          (len(alldocs), len(train_docs), len(test_docs)))



    return

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

# MAIN PROGRAM:
def runflow(dataset_path): 
   
    #===================================================================
    #=#BLOCK#=#: Get read and write paths for cluster functions 
    #===================================================================
    print("Getting read and write paths for cluster functions. ")  
  
    # check if the dataset location is a valid directory 
    check_valid_dir(dataset_path)
   
    # get its absolute path
    dataset_path = os.path.abspath(dataset_path)
    
    # the name of the top-level directory of the dataset
    dataset_name = get_last_dir_from_path(dataset_path)
     
    # get converted file location and output location
    out_dir = os.path.join(dataset_path, 
                           "../" + "converted-" + dataset_name)
    
    # define the write path for the entire program
    write_path = "../../cluster-datalake-outputs/" + dataset_name + "--output/"
    if not os.path.isdir(write_path):
        os.system("mkdir " + write_path)
    print("All results printing to " + write_path)
    
    # get absolute paths 
    out_dir = os.path.abspath(out_dir)
    write_path = os.path.abspath(write_path)
    
    # get the location of the extension index file
    print("Finding extension index file. ")
    ext_dict_file_loc = os.path.join(write_path, "extension_index_"
                                     + dataset_name + ".npy")
    
    # check if the above paths are valid 
    check_valid_dir(out_dir)
    check_valid_file(ext_dict_file_loc)
    
    # load the extension to path dict
    ext_to_paths_dict = np.load(ext_dict_file_loc).item()
    csv_path_list = []
    txt_path_list = []
    if "csv" in ext_to_paths_dict:
        csv_path_list = ext_to_paths_dict["csv"]
    if "txt" in ext_to_paths_dict:
        txt_path_list = ext_to_paths_dict["txt"]
 
    # location of files converted to csv format
    csv_dir = os.path.join(out_dir, "csv/")
    converted_csv_list = os.listdir(csv_dir)

    doc2vec(dataset_path, dataset_name, write_path, csv_path_list)

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
    runflow(dataset_path) 
    return

#=========1=========2=========3=========4=========5=========6=========7=
#=========1=========2=========3=========4=========5=========6=========7=

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main() 


