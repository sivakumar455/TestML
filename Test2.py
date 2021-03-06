
import pandas
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import random
import pickle

from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

class Test:
    def fit(self,X_train,Y_train):
        self.X_train=X_train
        self.Y_train=Y_train
    def predict(self,X_validation):
        res=[]
        for row in X_validation:
            #pred=random.choice(Y_train)
            pred= self.closet(row)
            res.append(pred)
        return res
    def closet(self,row):
        best_dist=euc(row,self.X_train[0])
        best_index= 0
        for i in range(1,len(X_train)):
            dis=euc(row,self.X_train[i])
            if dis<best_dist:
                best_dist=dis
                best_index=i
        return self.Y_train[best_index]

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# Print dataset

#print dataset.shape
#print dataset.head(20)
#print dataset.describe()
#print dataset.groupby('class').size()

#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#dataset.hist()
#scatter_matrix(dataset)
#plt.show()



array = dataset.values
#print array

X = array[:,0:4]
Y = array[:,4]
#print X

#print " Y data "
#print Y

validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)

#print "X_tarin :"
#print X_train
#print "Y_train"
#print Y_train

#print X_validation

# Make predictions on validation dataset
knn = KNeighborsClassifier()
#knn=Test()

knn.fit(X_train, Y_train)

#pickle.dump(knn,open("knn.pkl","wb"))

#predictions = knn.predict(X_validation)

pred = knn.predict([[5.9, 3.0, 5.1,1.8],[6.2,2.2,4.5,1.5]])

print pred[1]

#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))

