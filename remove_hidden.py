import sys
import os

# Removes all the hidden files which start with "._" from the specified
# directory. 

#=========1=========2=========3=========4=========5=========6=========7=

# ARGUMENTS
# the directory from which we're removing hidden files
directory = sys.argv[1]
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

# grab the contents of the directory
dir_list = os.listdir(directory)
old_size = len(dir_list)
print(old_size)

# navigate to that directory
os.chdir(directory)

# delete all files that begin with "._"
os.system("rm ._*")

dir_list = os.listdir(directory)
new_size = len(dir_list)
print(new_size)
print("removed ", old_size - new_size, " files")
