# necessary imports
import sklearn
from sklearn import metrics
from sklearn import svm

cn = sklearn.datasets.load_breast_cancer() # load data set

x = cn.data # set x to the entries/features
y = cn.target # set y to the labels

# Split the data into training entries, testing entries, training labels, and testing labels
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2) 

# Create an SVC
clf = svm.SVC()

# train the training set
clf.fit(x_train, y_train)

# Predict the labels for the training data
y_predict = clf.predict(x_test)

# compare the predicted labels to the actual labels and conver to percentage to output
acc = metrics.accuracy_score(y_test, y_predict)*100
print(f"{acc}% accuracy")
