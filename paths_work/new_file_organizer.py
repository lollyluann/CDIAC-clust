from shutil import copyfile
import new_DFS
import tokenizer
import sys
import os

#=========1=========2=========3=========4=========5=========6=========7=

def copy_and_sort(input_path, output_path):
    # gets a dictionary of the files to their paths
    transformed_paths = new_DFS.DFS(input_path)

    # reads in the top file extensions
    ext_file = open("top_exts.txt", "r")
    lines  = ext_file.readlines()
    top_n_exts = []
    # removes newline characters
    for e in lines:
        top_n_exts.append(e.rstrip())

    # makes a new directory for the sorted data
    # path = "/home/ljung/extension_sorted_data_renamed/"
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    # makes a new folder for each of the top extensions
    for extension in top_n_exts:
        if not os.path.isdir(output_path+extension+"/"):
            os.mkdir(os.path.join(output_path,extension))

    # checks every file and moves those with the top extensions
        # e.g. /home/bob.pdf moves to /home/files/pdf/|home|bob.pdf
    for fp in transformed_paths:
        fn = new_DFS.get_fname_from_path(fp)
        if fn[:2]!="._":
            ext = tokenizer.get_single_extension(fn)
            if ext in top_n_exts:
                if not os.path.exists(output_path+ext+"/"+fp):
                    copyfile(new_DFS.str_decode(fp), output_path+ext+"/"+fp)


def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    copy_and_sort(input_path, output_path)

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main() 
