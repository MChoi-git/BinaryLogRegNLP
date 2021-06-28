import numpy as np
import sys
import json

'''
Description: Using the dict.txt provided, convert each line in the train/test/validation data into formatted lines.
    -Format:
        label\tindex[word1]:value1\tindex[word2]:value2\t,...index[wordM]:valueM\n
    -Where:
        index[word#] is the index of the word in the dictionary
        value# is the value of this feature {0,1}
Notes: 
    -Maybe explore using numpy (should be faster for larger datasets)
        
'''

#Command line string
#python feature.py train_data.tsv valid_data.tsv test_data.tsv model1_train_output.tsv model1_valid_output.tsv model1_test_output.tsv dict.txt 1
#python feature.py train_data.tsv valid_data.tsv test_data.tsv model2_train_output.tsv model2_valid_output.tsv model2_test_output.tsv dict.txt 2

#Inputs
train_input = f"handout/largedata/{sys.argv[1]}"                #Path to training input .tsv file
validation_input = f"handout/largedata/{sys.argv[2]}"           #Path to the validation input .tsv file
test_input = f"handout/largedata/{sys.argv[3]}"                 #Path to the test input .tsv file

#Outputs
formatted_train_out = f"output/largeoutput/{sys.argv[4]}"         #Path to output .tsv file to which the feature extractions on the training data should be written
formatted_validation_out = f"output/largeoutput/{sys.argv[5]}"  #Path to output .tsv file to which the feature extractions on the validation data should be written
formatted_test_out = f"output/largeoutput/{sys.argv[6]}"        #Path to output .tsv file to which the feature extractions on the test data should be written

#Miscellaneous
dict_input = f"handout/{sys.argv[7]}"                           #Path to the dictionary input
feature_flag = int(sys.argv[8])                                 #Integer taking 1 or 2 that specifies whether to construct the Model1 feature set or Model2 feature set


#Turn the provided dictionary text file into an actual dictionary object for use later
def initDict(filename):
    new_dict = {}
    with open(filename, "r") as dict:
        for line in dict:
            key, value = line.strip().split(' ')
            new_dict.update({key: value})
    return new_dict


#Takes a text file and turns it into an array, where each array element corresponds to a line which has been split
def txt2splitarray(filename):
    with open(filename, "r") as f:
        file_contents = f.readlines()
    split_file_array = [line.split() for line in file_contents]
    return split_file_array


#Takes a split line and outputs an inorder dict with unique word counts
def line2freqdict(line):
    line_dict = {}
    words_seen = set()  #Aux data using set for time reduction (quadratic time bad)
    for word in line:
        if word not in words_seen:
            words_seen.add(word)
            line_dict[word] = 1
        else:
            line_dict[word] += 1
    return line_dict


#Takes a word frequency dictionary and converts it to an array of dictionaries, where the keys are indexes obtained from the provided dictionary text file
def freqdict2indexedfreqdict(freqdict, dict_ref):
    indexedfreqdict = {}
    label = list(freqdict.keys())[0]    #Get the label {0,1} for the review
    if label != '0' and label != '1':
        print(f"An error occurred when indexing the frequency dict with label: {label}.")
        return [label]
    if freqdict[label] > 1:     #Remove the labels occurrence
        freqdict[label] -= 1
    else:   #Remove the label all together
        del freqdict[label]
    for key in freqdict:    #Index all the keys wrt. the provided indexing dictionary
        if key in dict_ref:
            indexedfreqdict[dict_ref[key]] = freqdict[key]
    return [label, indexedfreqdict]


#Writes a dictionary to a log file for testing purposes
def dict2log(filename, array_of_dicts):
    f = open(filename, "w")
    for dict in array_of_dicts:
        f.write(json.dumps(dict))
    f.close()


#Converts an array of dictionaries into formatted strings according to bag-of-words distribution model
def dicts2formatstrings(array_of_dicts, model, trim):
    collection_of_formatted_strings = []
    for label_dict_pair in array_of_dicts:
        label = label_dict_pair[0]
        dict_ = label_dict_pair[1]
        format_string = f"{label}"
        for key in dict_:
            if model == 1:
                if dict_[key] >= 1:
                    format_string += f"\t{key}:1"
            elif model == 2:
                if dict_[key] < trim:
                    format_string += f"\t{key}:1"
            else:
                print("Error: Model selection error when formatting strings.")
                return False
        collection_of_formatted_strings.append(format_string + "\n")
    return collection_of_formatted_strings


#Writes the final formatted strings into the .tsv output file
def writeformatstrings(filename, format_strings):
    f = open(filename, "w")
    for line in format_strings:
        f.write(line)
    f.close()


#Big nasty function that calls every function above for each input/output file
def formatAll(input_filenames, output_filenames, misc_inputs):
    if len(input_filenames) != len(output_filenames):   #Sanity check
        print(f"Main: Formatting error, filename lists of different size.{input_filenames} | {output_filenames}")
        return False
    for input_data_filename, output_data_filename in zip(input_filenames, output_filenames):
        #Format the dictionary file into a dictionary
        formatted_dict = initDict(misc_inputs[0])

        #Split each review into a list of words
        train_array = txt2splitarray(input_data_filename)

        #Convert the list of split reviews into word frequency dictionaries
        word_freq_dict_collection = []
        for review in train_array:
            word_freq_dict_collection += [line2freqdict(review)]

        #Convert the words in the word freq dict into index numbers given by the provided reference dictionary file
        indexed_word_freq_dict_collection = []
        for review in word_freq_dict_collection:
            indexed_word_freq_dict_collection += [freqdict2indexedfreqdict(review, formatted_dict)]

        #Write the word frequency dictionary collection to a log file
        #dict2log("indexed_word_freq_dict_collection.log", indexed_word_freq_dict_collection)

        #Convert the word frequency dictionary collectin to formatted strings
        if misc_inputs[1] == 1:
            format_strings = dicts2formatstrings(indexed_word_freq_dict_collection, misc_inputs[1], 4)
            writeformatstrings(output_data_filename, format_strings)
        elif misc_inputs[1] == 2:
            format_strings_trimmed = dicts2formatstrings(indexed_word_freq_dict_collection, misc_inputs[1], 4)
            writeformatstrings(f"{output_data_filename}", format_strings_trimmed)
        else:
            print(f"Main: Formatting error, flag not set properly. Flag: {misc_inputs[1]}")
            return False

        #Print the formatted strings in a log file and a tsv file
        #dict2log("format_strings.log", format_strings)
        #dict2log("format_strings_trimmed.log", format_strings_trimmed)
    return True


#Main

#Put input filenames in a list for easy looping
inputFilenames = [train_input, validation_input, test_input]
outputFilenames = [formatted_train_out, formatted_validation_out, formatted_test_out]
miscInputs = [dict_input, feature_flag]

#Format everything and send them to the correct output .tsv files
if formatAll(inputFilenames, outputFilenames, miscInputs):
    print("Program has successfully formatted features.")
else:
    print("Error in features formatting.")




