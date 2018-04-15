# Artificial Neural Network

# Data Preprocessing
import numpy as np
import pandas as pd

dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:,1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:,2])

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part 2 - Building the ANN!

# Improting the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=6, activation="relu", input_dim=11))
classifier.add(Dropout(p=0.1))

# Adding the second hidden layer
classifier.add(Dense(units=6, activation="relu"))
classifier.add(Dropout(p=0.1))

# Adding the output layer
classifier.add(Dense(units=1, activation="sigmoid"))

# Compiling the ANN
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Part 3 - Making the predictions and evaluating model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Part 4 - Evaluating, Improving, and tuning the ANN

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, activation="relu", input_dim=11))
    classifier.add(Dense(units=6, activation="relu"))
    classifier.add(Dense(units=1, activation="sigmoid"))
    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier,batch_size=10, epochs=100)
accuracues = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=4)
mean = accuracues.mean()
variance = accuracues.std()


#Tuninng the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, activation="relu", input_dim=11))
    classifier.add(Dense(units=6, activation="relu"))
    classifier.add(Dense(units=1, activation="sigmoid"))
    classifier.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return classifier
classifier = KerasClassifier(build_fn=build_classifier)
parameters = {
        "batch_size":[25,32],
        "nb_epoch":[100,500],
        "optimizer":["adam","rmsprop"]
        }
grid_search = GridSearchCV(estimator=classifier,param_grid=parameters, scoring="accuracy", cv=10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_