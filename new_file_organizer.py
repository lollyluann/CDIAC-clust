from shutil import copyfile
import new_DFS
import tokenizer
import sys
import os

#=========1=========2=========3=========4=========5=========6=========7=

path = sys.argv[1]

# gets a dictionary of the files to their paths
transformed_paths = new_DFS.DFS(path)

# reads in the top file extensions
ext_file = open("top_exts.txt", "r")
lines  = ext_file.readlines()
top_n_exts = []
# removes newline characters
for e in lines:
    top_n_exts.append(e.rstrip())

# makes a new directory for the sorted data
path = "/home/ljung/extension_sorted_data/"
if not os.path.isdir(path):
    os.mkdir(path)
# makes a new folder for each of the top extensions
for p in top_n_exts:
    if not os.path.isdir(path+p+"/"):
        os.mkdir(path+p+"/")

# checks every file and moves those with the top extensions
    # e.g. /home/bob.pdf moves to /home/files/pdf/|home|bob.pdf
for fp in transformed_paths:
    fn = new_DFS.get_fname_from_path(fp)
    ext = tokenizer.get_single_extension(fn)
    if ext in top_n_exts:
        if not os.path.exists(path+ext+"/"+fp):
            copyfile(new_DFS.str_decode(fp), path+ext+"/"+fp)
