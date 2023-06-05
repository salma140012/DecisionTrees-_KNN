import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# load the dataset
dataset = pd.read_csv('BankNote_Authentication.csv')

# split the dataset into features and target variable
features = dataset .drop('class', axis=1)# select all columns except the target variable
target = dataset ['class']# select the target variable column

# set the fixed random seed
random_seed = 40

# define the train_test_split ratio
test_size = 0.25

# loop through 5 random splits and train the model
for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=random_seed+i)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    print(f"Experiment {i+1}: Size of the tree = {clf.tree_.node_count}, Accuracy = {accuracy}")


# define the different train_test_split ratios
split_ratios = [(0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3)]



# create empty lists to store the results
accuracies = []
tree_sizes = []

# loop through the different split ratios and train the model with 5 different random seeds
for ratio in split_ratios:
    accuracy_list = []
    size_list = []
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=ratio[1], random_state=random_seed+i)
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predicted)
        accuracy_list.append(accuracy)
        size_list.append(clf.tree_.node_count)
    accuracies.append(accuracy_list)
    tree_sizes.append(size_list)

# calculate the mean, max and min accuracy and tree size for each split ratio
mean_accuracies = []
for x in accuracies:
    mean_accuracies.append(np.mean(x))

max_accuracies=[]
for x in accuracies:
    max_accuracies.append(np.max(x))


min_accuracies=[]
for x in accuracies:
    min_accuracies.append(np.min(x))

mean_sizes=[]
for x in tree_sizes:
    mean_sizes.append(np.max(x))

max_sizes=[]
for x in tree_sizes:
    max_sizes.append(np.max(x))

min_sizes=[]
for x in tree_sizes:
    min_sizes.append(np.min(x))



# print the results
print("Results:")
for i, ratio in enumerate(split_ratios):
    print(f"Split Ratio: {ratio[0]}-{ratio[1]}, Mean Accuracy: {mean_accuracies[i]}, Max Accuracy: {max_accuracies[i]}, Min Accuracy: {min_accuracies[i]:.4f}, Mean Tree Size: {mean_sizes[i]}, Max Tree Size: {max_sizes[i]}, Min Tree Size: {min_sizes[i]}")

# plot the results

plt.plot([ratio[0] for ratio in split_ratios], mean_accuracies, 'o-')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Accuracy')
plt.title('Mean Accuracy vs Training Set Size')
plt.show()


plt.plot([ratio[0] for ratio in split_ratios], mean_sizes, 'o-')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Tree Size')
plt.title('Mean Tree Size vs Training Set Size')

plt.show()