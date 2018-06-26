from tqdm import tqdm
import sys
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

def get_valid_filenames(dir_list, directory):
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
        if not os.path.isfile(out_path + ".0"):
            os.system("ssconvert " + in_path + " " + out_path + ".csv > /dev/null 2>&1 -S")

#=========1=========2=========3=========4=========5=========6=========7=

# MAIN PROGRAM: 
valid_list = get_valid_filenames(dir_list, directory)
convert_those_files(valid_list, directory, out_dir)








