
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import pandas as pd
import numpy as np

kidney_disease = pd.read_csv("kidney_disease.csv")

# convert categorical features to 1s and 0s and clean data so that it can be KNN imputed (fill missing values)
kidney_disease = kidney_disease.replace({"notpresent": 0.0, 'present': 1.0,
                                               "normal": 1.0,"abnormal": 0.0,
                                               "yes":1.0, "no":0.0,
                                               "poor":0.0, "good":1.0,
                                               "ckd\t":1.0, "notckd\t":0.0,
                                               "ckd":1, "notckd":0,
                                               "\t":np.nan, "\tno":0,
                                               " yes":1, "\tyes":1,
                                               "\t?":np.nan})

# fill missing data with KNN imputation
num_neighbors = 5
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

# calculate performance using test labels and predicted labels, then print it out
print(f"confusion matrix: \n{confusion_matrix(y_test, kidney_disease_prediction)}")
print(f"accuracy: {accuracy_score(y_test, kidney_disease_prediction)}\nprecision: {precision_score(y_test, kidney_disease_prediction)}\nrecall: {recall_score(y_test, kidney_disease_prediction)}\nf1-Score: {f1_score(y_test, kidney_disease_prediction)}")

"""
in the context of kidney disease predictions:
true positive - the model predicts that the patient does have the disease, and they do have the disease
true negative - the model predicts that the patient does not have the disease, and the patient does not have the disease.
false positive - the model predicts that the patient does have the disease, but they do not have the disease
false negative - the model predicts that the patient does not have the disease, but they do have the disease

we cannot rely solely on accuracy to evaluate a model because accuracy ignores class imbalance, as well as, it does not account for false positives or false negatives.
if one class is more frequent, accuracy can become meaningless.
accuracy does not account for false positives or false negatives, and depending on the model, some mistakes can have a bigger impact.

the most important feature would be recall because it will show the amount of kidney diseases the model predicts.

"""
