from tqdm import tqdm
import path_utilities
import csv
import os
import re

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def standalone_schema_id(path):
    with open(path, "r") as f:
        # read csv and get the header as a list
        reader = csv.reader(f)
        try:
            # list of all rows in csv 
            rows = list(list(rec) for rec in csv.reader(f, delimiter=','))
  
            header_list = []

            num_aligned_floats = 1
            float_loc = float('Inf') 
            for i in range(len(rows)):
    
                # if the row is empty, try the next line
                if (len(rows[i]) == 0):
                    continue
                 
                # number of nonempty cells
                num_nonempty = 0
                for cell in rows[i]:
                    if not (cell == ""):
                        num_nonempty = num_nonempty + 1
                fill_ratio = num_nonempty / len(rows[i])                
                if (fill_ratio == 0):
                    continue        

                old_float_loc = float_loc
                float_loc = float('Inf')
                
                row = rows[i]
                for j in range(len(row) - 1):
                    # if we have two consecutive float cells
                    if not re.search('[a-zA-Z]', row[j]) and not re.search('[a-zA-Z]', row[j + 1]):
                        # save its location in that row
                        float_loc = j

                # if there exists a float in the current row AND 
                # in the same place as the last... 
                if (float_loc != float('Inf') and float_loc == old_float_loc):
                    num_aligned_floats = num_aligned_floats + 1
                
                if (num_aligned_floats == 5):
                    shift_up = 5
                    first_cell = rows[i - shift_up][0]
                    while (not re.search('[a-zA-Z]', first_cell) or first_cell == "") and shift_up < i:
                        print (first_cell)
                        shift_up += 1
                        first_cell = rows[i - shift_up][0]
                    header_list = rows[i - shift_up]
                    print("found header")
                    break
        except UnicodeDecodeError:
            decode_probs = decode_probs + 1
        print(header_list) 

#RETURNS: a dictionary which maps filenames to csvs header lists. 
def get_better_dict(csv_dir, csv_path_list, fill_threshold, converted_status):
    header_dict = {}
    # number of files with no valid header
    bad_files = 0
    decode_probs = 0

    # This code is rather confusing because I wanted the function to 
    # be able to handle both types of inputs (lists of paths in names)
    # and just directory locations. 

    # CASE 1:
    # If we're reading in converted files, we only need the csv_dir
    # argument, so we get a list of the filenames from that directory. 
    # These filenames are in the form:
    # "@home@ljung@pub8@oceans@some_file.csv"
    if (converted_status):
        dir_list = os.listdir(csv_dir)

    # CASE 2:
    # Otherwise, we are reading in a list of the true, original 
    # locations of files that were csvs to begin with in the dataset.
    else:
        dir_list = csv_path_list
    
    # CASE 1: "path" looks like:"@home@ljung@pub8@oceans@some_file.csv" 
    # CASE 2: "path" is literally the path of that file in the original
    # dataset as a string. 
    for path in tqdm(dir_list):
        if (converted_status): 
            # get the new location of the current file in "csv_dir", 
            # i.e. not in original dataset. 
            filename = path
            path = os.path.join(csv_dir, path) 
        else:
            # convert to "@home@ljung@pub8@oceans@some_file.csv" form. 
            filename = path_utilities.str_encode(path)

        # So now in both cases, filename has the "@"s, and path is
        # the location of some copy of the file. 

        with open(path, "r") as f:
            # read csv and get the header as a list
            reader = csv.reader(f)
            try: 
                rows = list(list(rec) for rec in csv.reader(f, delimiter=','))
      
                header_list = []

                num_aligned_floats = 1
                float_loc = float('Inf') 
                for i in range(len(rows)):
        
                    # if the row is empty, try the next line
                    if (len(rows[i]) == 0):
                        continue
                     
                    # number of nonempty cells
                    num_nonempty = 0
                    for cell in rows[i]:
                        if not (cell == ""):
                            num_nonempty = num_nonempty + 1
                    fill_ratio = num_nonempty / len(rows[i])                
                    if (fill_ratio == 0):
                        continue        

                    old_float_loc = float_loc
                    float_loc = float('Inf')
                    
                    row = rows[i]
                    for j in range(len(row) - 1):
                        # if we have two consecutive float cells
                        if not re.search('[a-zA-Z]', row[j]) and not re.search('[a-zA-Z]', row[j + 1]):
                            # save its location in that row
                            float_loc = j

                    # if there exists a float in the current row AND 
                    # in the same place as the last... 
                    if (float_loc != float('Inf') and float_loc == old_float_loc):
                        num_aligned_floats = num_aligned_floats + 1
                    
                    if (num_aligned_floats == 5):
                        shift_up = 5
                        first_cell = rows[i - shift_up][0]
                        while (not re.search('[a-zA-Z]', first_cell) or first_cell == "") and shift_up < i:
                            shift_up += 1
                            first_cell = rows[i - shift_up][0]
                        header_list = rows[i - shift_up]
                        print("found header")
                        break
            except UnicodeDecodeError:
                decode_probs = decode_probs + 1                    
            # throw a key value pair in the dict, with filename as key
            header_dict.update({filename:header_list})
    print("Couldn't find the header in " + str(bad_files) + " files. Discarded.")    
    print("Number of UnicodeDecodeErrors: ", decode_probs)
    return header_dict

def main():
    print("nothing here. ")
if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main() 
