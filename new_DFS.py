import sys
import os

# FIRST ARGUMENT: the root directory path.

#=========1=========2=========3=========4=========5=========6=========7=

# RETURNS: a list of the paths of every file in the directory "path".

def DFS(path):
    stack = []
    transformed_paths = []
    stack.append(path);
    while len(stack) > 0:
        tmp = stack.pop(len(stack) - 1)
        # if this is a valid path.
        if(os.path.isdir(tmp)):
            # for every item in the "tmp" directory
            for item in os.listdir(tmp):
                # throws the path given by "tmp" + "item" onto the stack
                # for each "item".
                stack.append(os.path.join(tmp, item))
        # if it's not a valid path but it IS a valid file.
        elif(os.path.isfile(tmp)):
            # "tmp" format: '/home/ljung/shit/shitposting/'
            # appends to list of transformed paths
            # formatted like: '|home|ljung|shit|shitposting|'
            transformed_paths.append(str_encode(tmp))
    return transformed_paths

#=========1=========2=========3=========4=========5=========6=========7=

def str_encode(string):
    return string.replace("/","|")

def str_decode(string):
    return string.replace("|","/")

def get_fname_from_path(path):    
    filename = ""
    for c in path[::-1]:
        if c=="/" or c=="|":
            break
        filename = c+filename
    return filename

#=========1=========2=========3=========4=========5=========6=========7=

def main():
    root_path = sys.argv[1]

    allpaths = DFS(root_path)

    os.chdir(root_path)
    f1 = open(root_path + "transformed_paths.txt","w")

    f1.write(str(allpaths))
    f1.close()

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main() 
