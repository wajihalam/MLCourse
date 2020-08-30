# import load_iris function from datasets module
from sklearn.datasets import load_iris

# save the dataset, which is actually a "bunch" object similar to python dictionary type
irisDS = load_iris()
print(type(irisDS))

# printing the content of the dataset: each column is a feature and each row is a sample
print(irisDS.data)

# print the feature name
print(irisDS.feature_names)

# retrieve the descriptive names of the classes and encoding schemes
# 0 = setosa, 1 = versicolor, 2 = virginica
print(irisDS.target_names)


# get the ground truth corresponding to each sample
print(irisDS.target)

