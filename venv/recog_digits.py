#To init, pip install scikit-learn
from sklearn import svm, datasets
import matplotlib.pyplot as plt

svc_set = svm.SVC(gamma=0.001, C=100.) #Support Vector Class
digits = datasets.load_digits()
print(digits.DESCR) #Prints out textual description of the dataset
print(digits.images[0])
plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()
print("Printing number of digits", digits.target.size)
plt.subplot(321)
plt.imshow(digits.images[1791], cmap=plt.cm.gray_r, interpolation="nearest")
plt.subplot(322)
plt.imshow(digits.images[1792], cmap=plt.cm.gray_r, interpolation="nearest")
plt.subplot(323)
plt.imshow(digits.images[1793], cmap=plt.cm.gray_r, interpolation="nearest")
plt.subplot(324)
plt.imshow(digits.images[1794], cmap=plt.cm.gray_r, interpolation="nearest")
plt.subplot(325)
plt.imshow(digits.images[1795], cmap=plt.cm.gray_r, interpolation="nearest")
plt.subplot(326)
plt.imshow(digits.images[1796], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()
svc = svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
    gamma=0.001, kernel="rbf", max_iter=-1, probability=False,
    random_state=None, shrinking=True, tol=0.001, verbose=False)
svc.fit(digits.data[1:1790], digits.target[1:1790])


print(svc.predict(digits.data[1791:1797]))