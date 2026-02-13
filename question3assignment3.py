import pandas as pd
from sklearn.model_selection import train_test_split

kidney_disease = pd.read_csv("kidney_disease.csv", delimiter = ",")

# X represents the feature matrix, y is the label vector
X = kidney_disease.drop("classification", axis=1)
y = kidney_disease["classification"]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=7)
# "test_size = 0.3" means testing using 30% of the data

"""
We should not train and test the model on the same data because we 
It is important to have different testing and training data so you can assess how well your model works with new data.
If the training and testing data are the same, it will be hard to tell if your model is overfitted, and you won't know how well your model works with different data.

The purpose of the testing set is to test the accuracy of our trained model.
The testing set is separate from the training set, so you can see how your model works with new data.

"""

