from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

kidney_disease = pd.read_csv("kidney_disease.csv")

kidney_disease = kidney_disease.replace({"notpresent": 0.0, 'present': 1.0,
                                               "normal": 1.0,"abnormal": 0.0,
                                               "yes":1.0, "no":0.0,
                                               "poor":0.0, "good":1.0,
                                               "ckd\t":1.0, "notckd\t":0.0,
                                               "ckd":1, "notckd":0,
                                               "\t":np.nan, "\tno":0,
                                               " yes":1, "\tyes":1,
                                               "\t?":np.nan})

k_values = [1,3,5,7,9]
accuracy_list = []

for num_neighbors in k_values:
    # fills missing data with KNN imputation
    imputer = KNNImputer(n_neighbors=num_neighbors)
    imputed_kidney_disease = pd.DataFrame(imputer.fit_transform(kidney_disease), columns=kidney_disease.columns)

    # creates feature matrix X of all columns except "classification" and create label vector y as "classification"
    X = imputed_kidney_disease.drop("classification", axis=1)
    y = imputed_kidney_disease["classification"]

    # splits the data into training data and testing data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=19)

    # trains the model w the data and predicts labels for the test data
    knn_model = KNeighborsClassifier(n_neighbors=num_neighbors, metric='euclidean')
    trained_knn_model = knn_model.fit(x_train, y_train)
    kidney_disease_prediction = trained_knn_model.predict(x_test)

    # calculates accuracy and then adds it to a list for our table
    accuracy = accuracy_score(y_test, kidney_disease_prediction)
    accuracy_list.append(accuracy)

# creates table that displays k-values and the accuracy
accuracy = pd.DataFrame({"value of K": k_values, "accuracy": accuracy_list})
print(accuracy)

# find the greatest accuracy and the k value
greatest_accuracy = 0
for i in range(0, len(accuracy_list)):
    if accuracy_list[i] > greatest_accuracy:
        greatest_accuracy = accuracy_list[i]
        k_value_greatest_accuracy = k_values[i]

print(f"highest test accuracy: {greatest_accuracy}\nassociated k-value: {k_value_greatest_accuracy}")

"""
changing the k-value will change how many data points are used in the prediction, which will affect the accuracy.
values of k that are too small may cause overfitting because the model only has a few data points to predict from, so it may not be accurate.
values of k that are too large may cause underfitting because the model will use too many data points, so the prediction may not be precise and accurate.

"""

