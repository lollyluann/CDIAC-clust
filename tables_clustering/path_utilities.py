import os

#=========1=========2=========3=========4=========5=========6=========7=

def str_encode(string):
    return string.replace("/","@")
   
#=========1=========2=========3=========4=========5=========6=========7=

def str_decode(string):
    return string.replace("@","/")

#=========1=========2=========3=========4=========5=========6=========7=

# RETURNS: index of "char" in "string". Returns the length of the string
# if there are no instances of that character. 
def find_char_from_end(string, char):
    
        length = len(string)
        # we iterate on the characters starting from end of the string
        pos = length - 1
        # dot pos will be position of the first period from the end,
        # i.e. the file extension dot. If it is still "length" at end,
        # we know that there are no dots in the filename. 
        dot_pos = length
        while (pos >= 0):
            if (filename[pos] == char):
                dot_pos = pos
                break
            pos = pos - 1
        return dot_pos

#=========1=========2=========3=========4=========5=========6=========7=

# Works with or without trailing "/". 
# RETURNS: "/hello/no/" from argument "/hello/no/yes/". 
def remove_path_end(path):
    # remove trailing "/" if it exists. 
    if (path[len(path) - 1] == "/"):
        path = path[0:len(path) - 1]

    # location of the delimiter "/"
    delim_loc = 0
    j = len(path) - 1
    # iterating to find location of first "/" from right
    while (j >= 0):
        if (path[j] == "/"):
            delim_loc = j
            break
        j = j - 1
    # shorten the path, include ending "/"
    shortened_path = path[0:delim_loc + 1]
    return shortened_path

#=========1=========2=========3=========4=========5=========6=========7=

def main():
    print("This file is just for importing functions, don't run it. ")    

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main() 
