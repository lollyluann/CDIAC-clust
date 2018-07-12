from tqdm import tqdm
import new_DFS
import sys
import os

#=========1=========2=========3=========4=========5=========6=========7=

# RETURNS distance between the files in two directories
# ONLY ACCEPTS DIRECTORY PATHS (NOT FILE PATHS)
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

def has_files(directory):
    for item in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, item)):
            return True
    return False

''' RETURNS: a list of the folders with files in them in the directory "path" '''
def DFS(path):
    stack = []
    all_dirs = []
    stack.append(path);
    while len(stack) > 0:
        # pop a path off the stack 
        tmp = stack.pop(len(stack) - 1)
        # if this is a directory and has at least one file in it
        if os.path.isdir(tmp):
            if has_files(tmp)==True:
                all_dirs.append(tmp)
            # for every item in the "tmp" directory
            for item in os.listdir(tmp):
                # throws the path given by "tmp" + "item" onto the stack
                stack.append(os.path.join(tmp, item))
    return all_dirs

def naive_max_dist(root):
    max_dist = 0
    all_paths = DFS(root)
    for path_a in tqdm(all_paths):
        for path_b in all_paths:
            dist = path_dist(path_a, path_b) 
            if (max_dist < dist):
                max_dist = dist
    return max_dist

#=========1=========2=========3=========4=========5=========6=========7=

''' PARAMETER: a full path
    RETURNS: the path without the filename '''
def get_dir_from_path(path):   
    ind = 0
    for c in path[::-1]:
        ind = ind+1
        if c=="/" or c=="@":
            break
    return path[:len(path)-ind+1]

def intracluster_dist(cluster_paths):
    distances = []
    for i in range(len(cluster_paths)-1):
        path1 = cluster_paths[i]
        path2 = cluster_paths[i+1]
        dirOf_path1 = get_dir_from_path(path1)
        dirOf_path2 = get_dir_from_path(path2)
        distances.append(path_dist(dirOf_path1, dirOf_path2))
    dists = np.array(distances)
    return np.mean(dists)

#=========1=========2=========3=========4=========5=========6=========7=

''' DEPRECATED '''
def find_max_dist(root):
    # "initial_depth" is the depth until siblings are reached
    initial_depth = 0
    max_dist = 0
    stack = []
    stack.append(root)
    while len(stack) > 0:
        tmp = stack.pop(len(stack)-1)
        children = []
        for item in os.listdir(tmp):
            if os.path.isdir(os.path.join(tmp,item)):
                children.append(os.path.join(tmp, item))
                stack.append(os.path.join(tmp, item))
        if len(children)<2:
            continue
        else:
            lengths = []
            for child in children:
                max_len = 0
                paths = new_DFS.DFS(child)
                for path in paths:
                    folders = path.split("/")
                    if len(folders)>max_len:
                        max_len = len(folders)
                lengths.append(max_len)
            lengths.sort()
            max_dist = lengths[0]+lengths[1]
            break
    return max_dist
                    
#=========1=========2=========3=========4=========5=========6=========7=

def main():
    root_path = sys.argv[1]
    max_dist = naive_max_dist(root_path)
    #max_dist = find_max_dist(root_path)
    print("The max_dist is: ", max_dist)
    
    t1 = "/home/ljung/CDIAC-clust/paths_work/test1"    
    t4 = "/home/ljung/CDIAC-clust/paths_work/test1/t2/t4"
    t2 = "/home/ljung/CDIAC-clust/paths_work/test1/t2"
    t5 = "/home/ljung/CDIAC-clust/paths_work/test1/t3/t5"
    t3 = "/home/ljung/CDIAC-clust/paths_work/test1/t3"
    print("t4 - t2", path_dist(t4,t2))
    print("t4 - t1", path_dist(t4,t1))
    print("t4 - t3", path_dist(t4, t3))    

    cluster_dirs = np.load("/home/ljung/CDIAC-clust

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main() 










