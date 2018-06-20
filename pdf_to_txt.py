import sys
import os

#=========1=========2=========3=========4=========5=========6=========7=

source = sys.argv[1]
dest = sys.argv[2]

''' PARAMETER: a single filename.
    RETURNS: the file without its extension '''
def remove_extension(filename):
    length = len(filename)
    for ch in filename[::-1]:
        if ch==".":
            break
        length = length - 1
    return filename[:length - 1]

# loops through each file in "source" and converts it to a txt file.
pdfs = os.listdir(source)
num_pdfs = len(pdfs)
i = 0
for filename in os.listdir(source):
    filename_no_ext = remove_extension(filename)
    os.system("ebook-convert " + source + filename 
        + " " + dest + filename_no_ext + ".txt")
    i = i + 1
    print("===================================================")
    print("\n\n\n\n") 
    print(i, " out of ", num_pdfs) 
    print("\n\n\n\n") 
    print("===================================================")
