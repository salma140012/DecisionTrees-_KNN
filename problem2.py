import pandas as pd
import numpy as np

bankNote_dataset = pd.read_csv('BankNote_Authentication.csv')

# splitting data into features and target
features = bankNote_dataset.drop('class', axis=1)
target = bankNote_dataset['class']

# convert features and target to numpy arrays
features = np.array(features)
target = np.array(target)

# Divide your data into 70% for training and 30% for testing.
train_size = int(len(bankNote_dataset) * 0.7)
X_TrainSet = features[0:train_size]
Y_TrainSet = target[0:train_size]
x_TestSet = features[train_size:]
y_TestSet = target[train_size:]

# Each feature column should be normalized separately from all other
# features. Specifically, for both training and test objects, each feature
# should be transformed using the function: f(v) = (v - mean) / std, using the
# mean and standard deviation of the values of that feature column on the
# training data.
for m in range(4):
    calcMean = np.mean(X_TrainSet[:, m])
    calcStd = np.std(X_TrainSet[:, m])
    x_TestSet[:, m] = (x_TestSet[:, m] - calcMean) / calcStd
    X_TrainSet[:, m] = (X_TrainSet[:, m] - calcMean) / calcStd


# calculate Euclidean distance between two vectors
def calc_euclidean_distance(v1, v2):
    euclidean_dist = np.sqrt(np.sum((v1 - v2) ** 2))
    return euclidean_dist


# find the nearest neighbors of a given test instance in a dataset, where the distance between instances is
# calculated using Euclidean distance
# train : training data
# test_row : test instance
# num_neighbors : number of nearest neighbors to find
# ytrain :labels for training data
def find_k_nearest_neighbor(train, test_row, num_neighbors, ytrain):
    # an empty list that will store the distance between the test instance and each instance in the training data
    dist = list()
    # Loops through each instance in the training data
    for i in range(len(train)):
        # Calculates the Euclidean distance between the test instance (test_row) and the current instance in the
        # training data (train[i, :]), and appends it to the dist list along with the label for that instance (
        # ytrain[i]).
        dist.append((calc_euclidean_distance(test_row, train[i, :]), ytrain[i]))
    # Sorts the dist list by the distance between each instance and the test instance, in ascending order
    dist.sort()
    # Creates an empty list called final that will be used to store the num_neighbors nearest neighbors.
    final = list()
    # Loops through the first num_neighbors instances in the sorted dist list.
    for i in range(num_neighbors):
        # Appends the current instance and its label to the final list.
        final.append(dist[i])
    # Returns the final list, which contains the num_neighbors nearest neighbors to the test instance.
    return final


# function to resolve tie
def predict(predicted, ytrain):
    # initialized to zero which will be used to count the number of instances where the predicted value is 1 and 0
    # respectively
    count_One = 0
    count_Zero = 0
    # to iterate over each element of the predicted array.
    for i in range(len(predicted)):
        # If the predicted value is equal to 1, then count_One is incremented by 1, otherwise, count_Zero is
        # incremented by 1.
        if predicted[i] == 1:
            count_One = count_One + 1
        else:
            count_Zero += 1
    # check if the count of 1's and 0's is equal.
    # If yes, it means that there are equal instances of both classes in the predicted array
    if count_One == count_Zero:
        # find the first instance of each class in ytrain the function finds the index of the first occurrence of
        # each class (1 and 0) in the ytrain array. Then it checks which class has its first occurrence earlier in
        # the ytrain array and returns that class (1 or 0).
        firstOne = np.where(ytrain == 1)[0][0]
        firstZero = np.where(ytrain == 0)[0][0]
        if firstOne < firstZero:
            return 1
        else:
            return 0
    # If the count of 1's is greater than the count of 0's, it returns 1, otherwise, it returns 0
    elif count_One > count_Zero:
        return 1
    else:
        return 0


def makePrediction(train, test, num):
    # call find_k_nearest_neighbor function and store it in variable p
    p = find_k_nearest_neighbor(train, test, num, Y_TrainSet)
    # creates a new list called ys, and then loops through the pred list and appends the class labels (pred[i][1]) of
    # the num nearest neighbors to the ys list
    ys = list()
    for i in range(num):
        ys.append((p[i][1]))
    # calls the predict function and passes in the ys list (which contains the class labels of the num nearest
    # neighbors) and the Y_TrainSet variable as arguments.
    return predict(ys, Y_TrainSet)


# function for calculating accuracy and count of correct predictions
# p : predicted labels
# a : actual labels
def accuracy_check(p, a):
    # variable for counting correct predictions
    ccp = 0
    # iterates through each index of the actual list ('a')
    for i in range(len(a)):
        # checks if the predicted label for the current index i is equal to the actual label for the current index i
        if p[i] == a[i]:
            # increment count by 1
            ccp += 1
    # calculate accuracy by dividing counting of correct predictions by length of the actual list (multiply it by 100
    # to be in %)
    accur = (ccp / len(a)) * 100
    # return accuracy and count of correct predictions
    return accur, ccp


#k = [1, 2, 3, 4, 5, 6, 7, 8, 9]
for k in range(10):
    ypred = []  # xtest
    prediction = []
    for i in x_TestSet:
        prediction = makePrediction(X_TrainSet, i, k+1)
        ypred.append(prediction)
    print("k value: ", k+1)
    accuracy, n = accuracy_check(y_TestSet, ypred)
    print("Number of correctly classified instances :", n, "Total number of instances :", len(y_TestSet))
    print("Accuracy :", accuracy)
