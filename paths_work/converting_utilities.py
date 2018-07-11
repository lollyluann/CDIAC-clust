from tqdm import tqdm
import new_DFS
import sys
import textract
import os

#=========1=========2=========3=========4=========5=========6=========7=

# ARGUMENTS
directory = sys.argv[1]         # the dataset location
dest = sys.argv[2]              # the destination directory
num_top_exts = int(sys.argv[3])      # use k most frequent extensions 

#=========1=========2=========3=========4=========5=========6=========7=

''' DOES: converts all the pdfs whose paths are specified in "pdf_paths"
          and puts the resultant text files in "dest/pdf/" '''
def convert_pdfs(pdf_paths, dest):
    num_pdfs = len(pdf_paths)
    print(num_pdfs, " pdfs for conversion")
    for path in tqdm(pdf_paths):      
        output_dir = os.path.join(dest, "pdf")
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        transformed_path = new_DFS.str_encode(path)
        if not os.path.isfile(os.path.join(output_dir, transformed_path + ".txt")):
            os.system("ebook-convert " + path + " " + os.path.join(output_dir, transformed_path + ".txt")) 

def convert_doc(doc_paths, dest):
    num_docs = len(doc_paths)
    print(num_docs, " docs for conversion")
    for path in tqdm(doc_paths): 
        try:     
            output_dir = os.path.join(dest, "doc")
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            transformed_path = new_DFS.str_encode(path)
            os.chdir(output_dir)
            if not os.path.isfile(os.path.join(output_dir, transformed_path + ".txt")):
                f = open(transformed_path+".txt", "w")
                contents = textract.process(path).decode("UTF-8").replace("\n", " ")
                f.write(contents)
                f.close()
        except textract.exceptions.ShellError:
            continue
 
def convert_docx(docx_paths, dest):
    num_docxs = len(docx_paths)
    print(num_docxs, " docxs for conversion")
    for path in tqdm(docx_paths):     
        try:
            output_dir = os.path.join(dest, "docx")
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            transformed_paath = new_DFS.str_encode(path)
            os.chdir(output_dir)
            if not os.path.isfile(os.path.join(output_dir, transformed_path + ".txt")):
                f = open(transformed_path+".txt", "w")
                contents = textract.process(path)
                f.write(str(contents))
                f.close()
        except textract.exceptions.ShellError:
            continue

#=========1=========2=========3=========4=========5=========6=========7=


# RETURNS: list of filepaths which are candidates for conversion.
def get_valid_filenames_tabular(dir_list):
    print("size of virtual directory: ", len(dir_list))
    list_valid_exts = [".xls", ".xlsx", ".tsv"]
    list_caps_exts = {".XLS":".xls", ".XLSX":".xlsx", ".TSV":".tsv"}
    valid_list = []
    # for each filename in the directory...
    for path in tqdm(dir_list):
        filename = new_DFS.get_fname_from_path(path)
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
                valid_list.append(path)
                valid = True
            # if the extension is an ALLCAPS version...
            elif extension in list_caps_exts.keys():
                #new_filename = filename[0:dot_pos] 
                # + list_caps_exts[extension]
                # change it to lowercase and add it to valid_list
                #os.system("mv " + os.path.join(directory, filename) 
                # + " " + os.path.join(directory, new_filename))
                valid_list.append(path)
                valid = True
        if (valid == False):
            print(extension)
            print("This filename is invalid: ", filename)
        
    print("There are ", len(valid_list), " candidates for conversion. ")
    return valid_list

#=========1=========2=========3=========4=========5=========6=========7=

# DOES: converts all the files in valid list to csv, and puts the
#       resultant files in out_dir. 
def convert_tabular(valid_list, out_dir):
    for path in tqdm(valid_list):
        # filename = new_DFS.get_fname_from_path(path)

        # converting
        # output will look like <encoded_filepath_w/_extension>.csv.<i>
        encoded_filename = new_DFS.str_encode(path)
        out_path = os.path.join(out_dir, encoded_filename)
        if not os.path.isfile(out_path + ".csv.0"):
            print("out_path: ", out_path)
            print("converting")
            os.system("ssconvert " + path + " " + out_path + ".csv > /dev/null 2>&1 -S")

#=========1=========2=========3=========4=========5=========6=========7=

def main():
    # create the destination directories for converted files.  
    csv_dest = os.path.join(dest, "csv/")
    print("csv_dest: ", csv_dest)
    if not os.path.isdir(csv_dest):
        os.system("mkdir " + csv_dest)

    # get a dictionary which maps extension names of the form "csv"
    # to lists of the full paths of files with those extensions in the 
    # dataset.
    ext_locations = new_DFS.extension_indexer(directory, num_top_exts)

    # if we have extensions with the following names, performs 
    # conversion. 
    if "pdf" in ext_locations:
        pdf_paths = ext_locations.get("pdf")
        convert_pdfs(pdf_paths, dest)
    if "doc" in ext_locations:
        doc_paths = ext_locations.get("doc")
        convert_doc(doc_paths, dest)
    if "docx" in ext_locations:
        docx_paths = ext_locations.get("docx")
        convert_docx(docx_paths, dest)
    '''
    if "xls" in ext_locations:
        xls_paths = ext_locations.get("xls")
        valid_xls = get_valid_filenames_tabular(xls_paths)
        convert_tabular(valid_xls, csv_dest)
    if "xlsx" in ext_locations:
        xlsx_paths = ext_locations.get("xlsx")
        valid_xlsx = get_valid_filenames_tabular(xlsx_paths)
        convert_tabular(valid_xlsx, csv_dest)
    if "tsv" in ext_locations:
        tsv_paths = ext_locations.get("tsv")
        valid_tsv = get_valid_filenames_tabular(tsv_paths)
        convert_tabular(valid_tsv, csv_dest)
    '''
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main() 


