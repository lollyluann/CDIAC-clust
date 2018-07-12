from tqdm import tqdm
import new_DFS
import sys
import os

#=========1=========2=========3=========4=========5=========6=========7=

# RETURNS distance between two files or directories
def path_dist(path1, path2):
    path1_folders = path1.split("/")
    path2_folders = path2.split("/")
    # "shared" contains the shared directory closest to the bottom
    # "common_index" contains the tree depth of "shared", e.g.
    # the common_index of the top level directory is 0. Its 
    # grandchildren have common index 2. 
    shared = ""
    common_index = 0
    min_folderlist_len = min(len(path1_folders), len(path2_folders))
    # iterate over the smaller of the lengths of the folderlists
    for i in range(min_folderlist_len):
        # if the directories match
        if path1_folders[i]==path2_folders[i]:
            # save the directory name in shared, save common index
            shared = path1_folders[i]
            common_index = i
        # once they are no longer equal, stop iterating. 
        else:
            break
    
    # compute distances to closest common ancestor
    p1_dist_to_shared = len(path1_folders)-1-common_index
    p2_dist_to_shared = len(path2_folders)-1-common_index

    total_dist = p1_dist_to_shared + p2_dist_to_shared
    return total_dist

#=========1=========2=========3=========4=========5=========6=========7=

def naive_max_dist(root):
    max_dist = 0
    all_paths = new_DFS.DFS(root)
    for path_a in tqdm(all_paths):
        for path_b in all_paths:
            dist = path_dist(path_a, path_b) 
            if (max_dist < dist):
                max_dist = dist
    return max_dist

#=========1=========2=========3=========4=========5=========6=========7=

def find_max_dist(root):
    stack = []
    count = 1
    all_paths = []
    stack.append(path);
    while len(stack) > 0:
        # pop a path off the stack 
        tmp = stack.pop(len(stack)-1)
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
            all_paths.append(tmp)
    return all_paths

#=========1=========2=========3=========4=========5=========6=========7=

def find_max_dist(root):
    # "initial_depth" is the depth until siblings are reached
    initial_depth = 0
    stack = []
    stack.append(root)
    while len(stack) > 0:
        tmp = stack.pop(len(stack)-1)
        children = []
        for item in os.listdir(tmp):
            if os.path.isdir(item):
                children.append(os.path.join(tmp, item))
                stack.append(os.path.join(tmp, item))
        if len(children)<2:
            continue
        else:
            #do DFS on each child
            break
                    
#=========1=========2=========3=========4=========5=========6=========7=

def main():
    root_path = sys.argv[1]
    max_dist = naive_max_dist(root_path)
    print("The max_dist is: ", max_dist)

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main() 










