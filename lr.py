import numpy as np
import sys
import math
import pandas as pd
import os

'''
Description: 
Notes:
    -All vectors should be numpy arrays
    -All matrices should be pandas dataframes
'''

#Model class containing parameters and learning rate
#   Honestly I might not even need this, but w/e I'll re-evaluate later
class binomialLogisticRegressionModel:
    def __init__(self, number_of_features, initial_learning_rate):
        self.model_parameters = np.zeros(number_of_features,dtype=float)    #Initializes parameter vector [0:M] to all zeros
        self.learning_rate = initial_learning_rate  #Sets the learning rate


#Turn dict.txt into a dictionary object
#Args:      <string>
#Returns:   <dict>
def init_dict(filename):
    new_dict = {}
    with open(filename, "r") as dict:
        for line in dict:
            key, value = line.strip().split(' ')
            new_dict.update({key: value})
    return new_dict


#Takes one formatted string example from .tsv and returns a list containing the label and feature vector np array
#Args:      <string>
#Returns:   [<string>, <np.array>]
def init_example(ith_example, num_attributes):
    split_ith_example = ith_example.strip().split()     #Strip whitespaces and turn into list of words
    label = split_ith_example[0]                        #Extract and save the label
    feature_vector = np.array(split_ith_example[1:])    #Extract and save the feature vector
    updated_feature_vector = np.zeros(num_attributes)
    for entry in feature_vector:                        #Convert the feature vector such that x_m = {0,1}, where m belongs to M
        pair = entry.split(":")
        index = int(pair[0])
        updated_feature_vector[index] = 1
    return [label, updated_feature_vector]


#Take one step in SGD, # features = length of dict.txt
#Args:      <binomial_logisitc_regression_model>, <np.array>, <string>
#Returns:   <np.array>
def step_sgd(model, features_vector, real_label):
    #For easier reading of the equation
    theta_j = model.model_parameters
    eta = model.learning_rate
    x_i = features_vector
    y_i = int(real_label)
    numerator = math.exp(np.dot(theta_j.T, x_i))
    denominator = math.exp(np.dot(theta_j.T, x_i)) + 1
    updated_theta_j = theta_j + (eta * x_i) * (y_i - (numerator / denominator))
    return updated_theta_j


#Trains the model iteratively by calling stepSGD until converged (or threshold reached)
#Args:      <binomial_logistic_regression_model>, <string>, <int>
#Returns:   <binomial_logistic_regression_model>
def train_model(model, tsv_dataset_filename, epochs, num_attributes):
    #Get the content of the training/validation/test data
    with open(tsv_dataset_filename, "r") as f:
        data = f.readlines()
    #Initialize the examples, separating labels from attributes
    dataset = [init_example(line, num_attributes) for line in data]     #Shuffle this to get a random example instead of current in-order implementation
    #Do a number of iterations of SGD
    for i in range(epochs):
        for j in range(len(dataset)):
            model.model_parameters = step_sgd(model, dataset[j][1], dataset[j][0])   #Change i to the shuffled variable
        print(f"Epoch: {i} has been completed.")
    return True


#Calculates the prediction based on model parameters and the data
def calculate_prediction(model, data):
    theta_j = model.model_parameters
    x_i = data
    probability = 1 / (math.exp(-np.dot(theta_j.T, x_i)) + 1)
    if probability >= 0.5:
        return 1
    elif probability < 0.5:
        return 0
    else:
        print("Error calculating probability.")
        return -1


#Predicts the labels for the specific dataset
def predict(model, dataset, num_attributes):
    #Get the content of the training/validation/test data
    with open(dataset, "r") as f:
        data = f.readlines()
    #Initialize the examples, separating labels from attributes
    dataset = [init_example(line, num_attributes) for line in data]
    predicted_labels = []
    num_incorrect_labels = 0
    for entry in dataset:
        label_value = calculate_prediction(model, entry[1])
        if entry[0] != str(label_value):
            num_incorrect_labels += 1
        predicted_labels.append(label_value)
    error = num_incorrect_labels / len(dataset)
    return [error, predicted_labels]


#Command line string
#python lr.py model1_train_output.tsv model1_valid_output.tsv model1_test_output.tsv dict.txt model1_train_out.labels model1_test_out.labels model1_metrics_out.txt 60
#python lr.py model2_train_output.tsv model2_valid_output.tsv model2_test_output.tsv dict.txt model2_train_out.labels model2_test_out.labels model2_metrics_out.txt 60

#Inputs
formatted_train_input = f"output/largeoutput/{sys.argv[1]}"
formatted_validation_input = f"output/largeoutput/{sys.argv[2]}"
formatted_test_input = f"output/largeoutput/{sys.argv[3]}"
dict_input = f"handout/{sys.argv[4]}"
train_out = f"output/largeoutput/{sys.argv[5]}"
test_out = f"output/largeoutput/{sys.argv[6]}"
metrics_out = f"output/largeoutput/{sys.argv[7]}"
numEpoch = sys.argv[8]

inputFilenames = [formatted_train_input, formatted_validation_input, formatted_test_input, dict_input]

outputFilenames = [train_out, test_out, metrics_out]



#Start 'er up
dict_txt = init_dict(dict_input)    #Create a usable dictionary
_M = len(dict_txt)  #Create constant variable M, where M = # features
#Initialize model object
blr_model = binomialLogisticRegressionModel(_M, 0.1)
#Train the model
train_model(blr_model, formatted_train_input, 60, _M)
#Predict with the trained model and display error
train_pair = predict(blr_model, formatted_train_input, _M)
print(f"Train error is: {train_pair[0]}")
with open(train_out, "w") as f:
    for label in train_pair[1]:
        f.write(f"{label}\n")
    f.close()
test_pair = predict(blr_model, formatted_test_input, _M)
print(f"Test error is: {test_pair[0]}")
with open(test_out, "w") as f:
    for label in test_pair[1]:
        f.write(f"{label}\n")
    f.close()
with open(metrics_out, "w") as f:
    f.write(f"Train error: {train_pair[0]}\nTestError: {test_pair[0]}")
    f.close()

#Sending model parameters to .csv
df = pd.DataFrame(blr_model.model_parameters)
os.remove("model_parameters.csv")
df.to_csv("model_parameters.csv", header=None, index=None, sep=',', mode='a')


