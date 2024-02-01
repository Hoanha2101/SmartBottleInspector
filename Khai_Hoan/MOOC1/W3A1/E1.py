from utils import *
from sklearn.metrics import accuracy_score

### START CODE HERE ### (≈ 3 lines of code)
shape_X = X.shape
shape_Y = Y.shape
m = Y.shape[1]  # training set size
### END CODE HERE ###

print('The shape of X is: ' + str(shape_X))
print('The shape of Y is: ' + str(shape_Y))
print('I have m = %d training examples!' % (m))

clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.ravel())  # Reshape Y to 1D array using .ravel()

LR_predictions = clf.predict(X.T)
accuracy = accuracy_score(Y.ravel(), LR_predictions) * 100
print('Accuracy of logistic regression: %d %% (percentage of correctly labelled datapoints)' % accuracy)
