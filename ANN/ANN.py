# Building an Artificial Neural Network - Geodemographic Segemnetation Model

# Data Pre-Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import keras
from keras.models import Sequential # TO INITIALIZE THE ANN
from keras.layers import Dense # TO CREATE THE LAYERS
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from keras.layers import Dropout


# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data

# Encoding the Independent Variable
labelencoder_X_country = LabelEncoder()
X[:, 1] = labelencoder_X_country.fit_transform(X[:, 1])

labelencoder_X_gender = LabelEncoder()
X[:, 2] = labelencoder_X_gender.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Make the Artificial Neural Network

# Initialize the Neural Network
classifier = Sequential()

'''
output_dim is the mean of number of independant Variables and 
number of dependant variables (x_train has 11 and y_train has 1 so avg is 6)
input_dim represents number of independant variables
'''
# Adding the Input Layer and the first Hidden Layer inlcuding Dropout
# Dropout Regularization to avoid over fitting
classifier.add(Dense(output_dim = 6, init = 'uniform',
                     activation = 'relu', input_dim = 11))
classifier.add(Dropout(p = 0.1))


# Adding the Second Hidden Layer including Dropout
# We do not need any input_dim because it knows what to expect. We have already created first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform',
                     activation = 'relu'))
classifier.add(Dropout(p = 0.1))
                                           
# Adding the Output Layer
classifier.add(Dense(output_dim = 1, init = 'uniform',
                     activation = 'sigmoid'))

# If dependant Variable has more than one category then, activation = 'softmax' 

# Compile the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])                                           


# Fitting classifier to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

'''
Use our ANN model to predict if the customer with the following informations will leave the bank: 

Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
'''
new_prediction = classifier.predict(sc.transform(np.array([[0, 0, 600, 0, 40, 3, 60000, 2, 1, 1, 20000]])))
new_prediction = (new_prediction > 0.5)


# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)


# Evaluating the Artificial Neural Network
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform',
                     activation = 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6, init = 'uniform',
                     activation = 'relu')) 
                                           
    # Adding the Output Layer
    classifier.add(Dense(output_dim = 1, init = 'uniform',
                     activation = 'sigmoid'))
    # If dependant Variable has more than one category then, activation = 'softmax' 

    # Compile the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()


# Parameter Tuning
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform',
                     activation = 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6, init = 'uniform',
                     activation = 'relu')) 
                                           
    # Adding the Output Layer
    classifier.add(Dense(output_dim = 1, init = 'uniform',
                     activation = 'sigmoid'))
    # If dependant Variable has more than one category then, activation = 'softmax' 

    # Compile the ANN
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

# Create a dictionary of the parameters, that need to be tuned
parameters = {
              'batch_size' : [25, 30], 
              'nb_epoch'   : [100, 400],
              'optimizer'  : ['adam', 'rmsprop']
             }

grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters, 
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)

optimized_parameters = grid_search.best_params_
max_accuracy = grid_search.best_score_

# Visualizing Training and Test Sets

'''
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
'''