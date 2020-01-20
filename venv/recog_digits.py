#To init, pip install scikit-learn
from sklearn import svm, datasets

svc_set = svm.SVC(gamma=0.001, C=100.) #Support Vector Class
digits = datasets.load_digits()
print(digits.DESCR) #Prints out textual description of the dataset