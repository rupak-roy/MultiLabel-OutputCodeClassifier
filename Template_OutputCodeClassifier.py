
#OutputCodeClassifier
'''
'Error-Correcting Output Code-based strategies are fairly different from one-vs-the-rest and one-vs-one. With these strategies, each class is represented in a Euclidean space, where each dimension can only be 0 or 1. Another way to put it is that each class is represented by a binary code (an array of 0 and 1). The matrix which keeps track of the location/code of each class is called the code book. The code size is the dimensionality of the aforementioned space. Intuitively, each class should be represented by a code as unique as possible and a good code book should be designed to optimize classification accuracy. In this implementation, we simply use a randomly-generated code book as advocated in 3 although more elaborate methods may be added in the future.
At the fitting time, one binary classifier per bit in the code book is fitted. At prediction time, the classifiers are used to project new points in the class space and the class closest to the points is chosen.
In OutputCodeClassifier, the code_size attribute allows the user to control the number of classifiers that will be used. It is a percentage of the total number of classes.
A number between 0 and 1 will require fewer classifiers than one-vs-the-rest. In theory, log2(n_classes) / n_classes is sufficient to represent each class unambiguously. However, in practice, it may not lead to good accuracy since log2(n_classes) is much smaller than n_classes.
A number greater than 1 will require more classifiers than one-vs-the-rest. In this case, some classifiers will in theory correct for the mistakes made by other classifiers, hence the name "error-correcting". In practice, however, this may not happen as classifier mistakes will typically be correlated. The error-correcting output codes have a similar effect to bagging.' ~ machine learning mastery. The script is a simple template that we can follow to apply OutputCodeClassifier
'''

#OutputCodeClassifier
from sklearn import datasets
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
X, y = datasets.load_iris(return_X_y=True)
clf = OutputCodeClassifier(LinearSVC(random_state=0),
   code_size=2, random_state=0)
clf.fit(X, y).predict(X)