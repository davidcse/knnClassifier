from os import listdir
from sys import argv
from random import shuffle
import numpy

##############################
# SET FLAGS
#############################
debug = 0
verbose = 0
freqOption = True
punctOption = True
punctuations = ".,?!&*()%$#@/\\:;\'\"<>-"
shuffleOption = True

for i in argv:
    if i == "-d":
        debug = 1
    if i == "-v":
        verbose = 1
    if i == "--punct":
        punctOption = True
    if i == "--nopunct":
        punctOption = False
    if i == "--frequency":
        freqOption = True
    if i == "--binary":
        freqOption = False
    if i == "--shuffle":
        shuffleOption = True

print("Options enabled : \npunctuations:"+str(punctOption) + "\tfrequency: " + str(freqOption) +"\tshuffle: " +str(shuffleOption))
####################################
#           FUNCTIONS
###################################

# GETS ALL THE FILENAMES AT DIRECTORY
# @dirpath : base path to folder.
def get_dir_files_arr(dirpath):
    dircontents = listdir(dirpath)
    dircontents.sort()
    return dircontents

# build file paths fully, and return array.
def build_full_paths(base_path, file_name_array):
    fullpaths = []
    for f in file_name_array:
        fullpaths.append(base_path + "/" + f)
    return fullpaths


# EXTRACTS FEATURES FROM A FILE, INTO THE DICTIONARY BASED ON WORD FREQUENCY.
# @filedata : str . text corpus of one file.
# @wordcount_dict : dictionary, to be built with the features.
def extract_features_frequency(filedata, wordcount_dict):
    tokens = filedata.split()
    for tok in tokens:
        if tok not in wordcount_dict:
            wordcount_dict[tok]=1;
        else:
            wordcount_dict[tok] += 1


# EXTRACTS FEATURES FROM A FILE, INTO THE DICTIONARY BASED ON WORD PRESENCE.
# @filedata : str . text corpus of one file.
# @wordcount_dict : dictionary, to be built with the features.
def extract_features_binary(filedata, wordcount_dict):
    tokens = filedata.split()
    for tok in tokens:
        wordcount_dict[tok]=1;

#input array of readable file paths.
#output, dictionary of filepath to the preprocessed feature vector for filepath.
def build_preprocessed_dict(filepath_arr, freqOption, punctOption, punctSymbols):
    master_features_dict = {}
    for filepath in filepath_arr:
        if verbose: print("preprocessing :\t" + filepath)
        file_dict = {}
        fHandler = open(filepath,'r')
        text = fHandler.read()
        fHandler.close()

        # Remove punctuation option from text
        if(not punctOption):
            for punct in punctSymbols:
                text = text.replace(punct,'')

        # build feature vector with frequency or binary.
        if(freqOption):
            extract_features_frequency(text,file_dict)
        else:
            extract_features_binary(text,file_dict)

        # finished processing this file. Add to master dict.
        master_features_dict[filepath] = file_dict
    # after all files processed, return master dict.
    return master_features_dict




# takes negative files, positive files, and interleaves them, storing the order in a combined array.
# @negative_files_paths : file paths for negative training data.
# @positive_file_paths : file paths for positive training data.
# @combined_array: array to be filled with alternating both positive and negative files for training.
# @sentiment : 1 for positive, -1 for negative. Which sentiment to start the interleaving with.
def interleave_pos_neg_files(negative_files_paths, positive_file_paths, combined_array, sentiment_flag):
    for i in range(len(negative_files_paths) + len(positive_file_paths)):
        if(len(negative_files_paths)==0 and len(positive_file_paths)==0):
            break
        elif(len(positive_file_paths)>0 and len(negative_files_paths)==0):
            while len(positive_file_paths) > 0 :
                pos_tuple = (positive_file_paths.pop(),1)
                combined_array.append(pos_tuple)
            break
        elif(len(positive_file_paths)==0 and len(negative_files_paths)>0):
            while len(negative_files_paths) >0:
                neg_tuple = (negative_files_paths.pop(),-1)
                combined_array.append(neg_tuple)
            break
        else:
            if(sentiment_flag == -1):
                neg_tuple = (negative_files_paths.pop(),-1)
                combined_array.append(neg_tuple)
                sentiment_flag = 1
            elif(sentiment_flag == 1):
                pos_tuple = (positive_file_paths.pop(),1)
                combined_array.append(pos_tuple)
                sentiment_flag = -1

#######################################
#   INIT
#######################################
#declares base paths of folders containing txt data
neg_files_basepath = "./data/review_polarity/txt_sentoken/neg"
pos_files_basepath = "./data/review_polarity/txt_sentoken/pos"

# get files in the base paths as arrays
negative_files = get_dir_files_arr(neg_files_basepath)
positive_files = get_dir_files_arr(pos_files_basepath)

# use retrieved file arrays to build upon base paths
negative_full_paths = build_full_paths(neg_files_basepath, negative_files)
positive_full_paths = build_full_paths(pos_files_basepath, positive_files)


master_features_dict_neg = build_preprocessed_dict(negative_full_paths,freqOption,punctOption,punctuations)
master_features_dict_pos = build_preprocessed_dict(positive_full_paths,freqOption,punctOption,punctuations)

if(shuffleOption):
    shuffle(negative_full_paths)
    shuffle(positive_full_paths)

# build interleaved arrays of positive and negative data, array elements are tuples.
# tuple[0] =  full path to txt file, tuple[1] = labeled result
combined_data = []
interleave_pos_neg_files(negative_full_paths, positive_full_paths, combined_data, 1)
