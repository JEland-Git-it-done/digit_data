#To init, pip install scikit-learn
from sklearn import svm, datasets
import matplotlib.pyplot as plt

svc_set = svm.SVC(gamma=0.001, C=100.) #Support Vector Class
digits = datasets.load_digits()
print(digits.DESCR) #Prints out textual description of the dataset
print(digits.images[0])
plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()