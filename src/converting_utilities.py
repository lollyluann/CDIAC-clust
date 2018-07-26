from bs4 import BeautifulSoup
from itertools import repeat
from tqdm import tqdm

from multiprocessing import Pool

import pandas as pd
import numpy as np
import subprocess
import sys
import csv
import os
import re

import path_utilities
import textract
import DFS

#=========1=========2=========3=========4=========5=========6=========7=

def parse_args():
    # ARGUMENTS
    dataset_path = sys.argv[1]         # the dataset location
    num_top_exts = int(sys.argv[2])      # use k most frequent extensions 
    num_processes = int(sys.argv[3])

    return dataset_path, num_top_exts, num_processes

def check_valid_dir(some_dir):
    if not os.path.isdir(some_dir):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("")
        print("DIES IST EIN UNGÜLTIGES VERZEICHNIS!!!!")
        print("fix yo directory")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        exit()

def progressBar(value, endvalue, bar_length=40):
        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write("\rFile {0}/{1}: [{2}] {3}%".format(value, endvalue, arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()

#==================2=========3=========4=========5=========6=========7=

def pdf_action(path, output_dir): 
    transformed_path = path_utilities.str_encode(path)
    FNULL = open(os.devnull, 'w')
    if not os.path.isfile(os.path.join(output_dir, transformed_path + ".txt")):
        subprocess.call(["ebook-convert", path, os.path.join(output_dir, transformed_path + ".txt")], 
                         stdout=FNULL, stderr=subprocess.STDOUT, close_fds=True) 

''' DOES: converts all the pdfs whose paths are specified in "pdf_paths"
          and puts the resultant text files in "dest/pdf/" '''
def convert_pdfs(pdf_paths, dest, num_processes):
    num_pdfs = len(pdf_paths)
    print("Converting", num_pdfs, "pdfs... This may take awhile.")
    output_dir = os.path.join(dest, "pdf")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    with Pool(num_processes) as p:
        p.starmap(pdf_action, zip(pdf_paths, [output_dir]*num_pdfs))

def docs_action(path, output_dir):
    transformed_path = path_utilities.str_encode(path)
    try:     
        if not os.path.isfile(os.path.join(output_dir, transformed_path + ".txt")):
            f = open(os.path.join(output_dir, transformed_path+".txt"), "w")
            contents = textract.process(path).decode("UTF-8").replace("\n", " ")
            f.write(contents)
            f.close()
    except textract.exceptions.ShellError:
        print("File skipped due to error") 
    
def convert_doc(doc_paths, dest, num_processes):
    num_docs = len(doc_paths)
    print("Converting", num_docs, "docs... This may take awhile.")
    output_dir = os.path.join(dest, "doc")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    with Pool(num_processes) as p:
        p.starmap(docs_action, zip(doc_paths, [output_dir]*num_docs))

def convert_docx(docx_paths, dest, num_processes):
    num_docxs = len(docx_paths)
    print("Converting", num_docxs, "docxs... This may take awhile.")
    output_dir = os.path.join(dest, "docx")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    with Pool(num_processes) as p:
        p.starmap(docs_action, zip(docx_paths, [output_dir]*num_docxs))

def mls_action(filetype, path, output_dir):
    with open(path, 'r', errors="backslashreplace") as content_file:
        contents = content_file.read()
        if filetype == "html":
            soup = BeautifulSoup(contents, 'html.parser')         
        else:
            soup = BeautifulSoup(contents, 'xml')         
        transformed_path = path_utilities.str_encode(path)
        if not os.path.isfile(os.path.join(output_dir, transformed_path + ".txt")):
            f = open(os.path.join(output_dir,transformed_path+".txt"), "w")
            f.write(soup.get_text())
            f.close()

def convert_html(html_paths, dest, num_processes):
    num_htmls = len(html_paths)
    print("Converting", num_htmls, "htmls... This may take awhile.")
    output_dir = os.path.join(dest, "html")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    with Pool(num_processes) as p:
        p.starmap(mls_action, zip(["html"]*len(html_paths), html_paths, [output_dir]*num_htmls))

def convert_xml(xml_paths, dest, num_processes):
    num_xmls = len(xml_paths)
    print("Converting", num_xmls, "xmls... This may take awhile.")
    output_dir = os.path.join(dest, "xml")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    with Pool(num_processes) as p:
        p.starmap(mls_action, zip(["xml"]*len(xml_paths), xml_paths, [output_dir]*num_xmls))

#=========1=========2=========3=========4=========5=========6=========7=

# RETURNS: list of filepaths which are candidates for conversion.
def get_valid_filenames_struct(dir_list):
    print("size of virtual directory: ", len(dir_list))
    list_valid_exts = [".xls", ".xlsx", ".tsv"]
    valid_list = []
    # for each filename in the directory...
    for path in tqdm(dir_list):
        filename = path_utilities.get_fname_from_path(path)
        length = len(filename)
        valid = False
        extension = path_utilities.get_single_extension(filename).lower()
        if extension in list_valid_exts:
            valid_list.append(path)
            valid = True
        if (valid == False):
            print(extension)
            print("This filename is invalid: ", filename)
        
    print("There are ", len(valid_list), " candidates for conversion. ")
    return valid_list

#=========1=========2=========3=========4=========5=========6=========7=

def tabular_action(path, out_path):
    subprocess.call(["ssconvert", path, out_path, ".csv > /dev/null 2>&1 -s"], 
                     stdout=FNULL, stderr=subprocess.STDOUT, close_fds=True) 

# DOES: converts all the files in valid list to csv, and puts the
#       resultant files in out_dir. 
def convert_tabular(valid_list, out_dir, num_processes):
    encoded_names = []
    out_path = []
    for path in valid_list:
        encoded_names.append(path_utilities.str_encode(path))
        out_path.append(os.path.join(out_dir, path_utilities.str_encode(path)))
    # output will look like <encoded_filepath_w/_extension>.csv.<i>
    print("Converting", len(valid_list), "tabular files... This may take awhile.")
    with Pool(num_processes) as p:
        p.starmap(tabular_action, zip(encoded_names, out_path))

#=========1=========2=========3=========4=========5=========6=========7=

def tsv_action(path, out_path):
    try:
        # use 'with' if the program isn't going to immediately terminate
        # so you don't leave files open
        # the 'b' is necessary on Windows
        # it prevents \x1a, Ctrl-z, from ending the stream prematurely
        # and also stops Python converting to / from different line terminators
        # On other platforms, it has no effect
        in_txt = csv.reader(open(path, "r"), delimiter = '\t')
        out_csv = csv.writer(open(out_path, 'w'))

        out_csv.writerows(in_txt)
        if not os.path.isfile(out_path):
            print("Did not save converted .tsv correctly. ")
    except:
        print("File skipped due to error")
    
def convert_tsv(valid_list, out_dir, num_processes): 
    encoded_names = []
    out_path = []
    for path in valid_list:
        encoded_names.append(path_utilities.str_encode(path))
        out_path.append(os.path.join(out_dir, path_utilities.str_encode(path)))
    # output will look like <encoded_filepath_w/_extension>.csv.<i>
    print("Converting", len(valid_list), "tsvs... This may take awhile.")
    with Pool(num_processes) as p:
        p.starmap(tsv_action, zip(encoded_names, out_path))  

#=========1=========2=========3=========4=========5=========6=========7=

# MAIN FUNCTION
def convert(dataset_path, num_top_exts, num_processes):
    
    check_valid_dir(dataset_path)

    # the name of the top-level directory of the dataset
    dataset_name = path_utilities.get_last_dir_from_path(dataset_path)

    # get its absolute path
    dataset_path = os.path.abspath(dataset_path)
    dest = os.path.join(dataset_path, "../finalconverted-" + dataset_name + "/")
####################################################################################################################################################
    if not os.path.isdir(dest):
        os.system("mkdir " + dest)
    check_valid_dir(dest)

    # get the script output location
    write_path = os.path.join("../../cluster-datalake-outputs/" + dataset_name + "--output/")

    # get its absolute path
    write_path = os.path.abspath(write_path)
    if not os.path.isdir(write_path):
        os.system("mkdir ../../cluster-datalake-outputs/") 
        os.system("mkdir " + write_path)
    check_valid_dir(write_path)

    # create the destination directories for converted files.  
    csv_dest = os.path.join(dest, "csv/")
    print("csv_dest: ", csv_dest)
    if not os.path.isdir(csv_dest):
        os.system("mkdir " + csv_dest)

    # get a dictionary which maps extension names of the form "csv"
    # to lists of the full paths of files with those extensions in the 
    # dataset.
    # CREATES "extension_index_<dataset_name>.npy"
    ext_locations = DFS.extension_indexer(dataset_path, 
                                          num_top_exts, write_path)

    # if we have extensions with the following names, performs 
    # conversion.
     
    if "pdf" in ext_locations:
        pdf_paths = ext_locations.get("pdf")
        convert_pdfs(pdf_paths, dest, num_processes)
    if "doc" in ext_locations:
        doc_paths = ext_locations.get("doc")
        convert_doc(doc_paths, dest, num_processes)
    if "docx" in ext_locations:
        docx_paths = ext_locations.get("docx")
        convert_docx(docx_paths, dest, num_processes)
    if "html" in ext_locations:
        html_paths = ext_locations.get("html")
        convert_html(html_paths, dest, num_processes)
    if "xml" in ext_locations:
        xml_paths = ext_locations.get("xml")
        convert_xml(xml_paths, dest, num_processes)
    
    if "xls" in ext_locations:
        xls_paths = ext_locations.get("xls")
        valid_xls = get_valid_filenames_struct(xls_paths)
        convert_tabular(valid_xls, csv_dest, num_processes)
    if "xlsx" in ext_locations:
        xlsx_paths = ext_locations.get("xlsx")
        valid_xlsx = get_valid_filenames_struct(xlsx_paths)
        convert_tabular(valid_xlsx, csv_dest, num_processes)
    if "tsv" in ext_locations:
        tsv_paths = ext_locations.get("tsv")
        valid_tsv = get_valid_filenames_struct(tsv_paths)
        convert_tsv(valid_tsv, csv_dest, num_processes)
    
def main():
    
    # get the arguments
    dataset_path, num_top_exts = parse_args()
    convert(dataset_path, num_top_exts)
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main() 


