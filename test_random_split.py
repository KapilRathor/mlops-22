from sklearn.model_selection import train_test_split
from utils import train_dev_test_split
from sklearn import datasets
from utils import preprocess_digits



digits = datasets.load_digits()
rsplit = 0
train_frac1, dev_frac1, test_frac1 = 0.8, 0.1, 0.1
data1, label1 = preprocess_digits(digits)
x_train1, y_train1, x_dev1, y_dev1, x_test1, y_test1 = train_dev_test_split(data1, label1, train_frac1, dev_frac1, rsplit)

rsplit = 0
train_frac2, dev_frac2, test_frac2 = 0.8, 0.1, 0.1
data2, label2 = preprocess_digits(digits)
x_train2, y_train2, x_dev2, y_dev2, x_test2, y_test2 = train_dev_test_split(data2, label2, train_frac2, dev_frac2, rsplit)

rsplit = 'none'
train_frac3, dev_frac3, test_frac3 = 0.8, 0.1, 0.1
data3, label3 = preprocess_digits(digits)
x_train3, y_train3, x_dev3, y_dev3, x_test3, y_test3 = train_dev_test_split(data3, label3, train_frac3, dev_frac3, rsplit)

rsplit = 'none'
train_frac4, dev_frac4, test_frac4 = 0.8, 0.1, 0.1
data4, label4 = preprocess_digits(digits)
x_train4, y_train4, x_dev4, y_dev4, x_test4, y_test4 = train_dev_test_split(data4, label4, train_frac4, dev_frac4, rsplit)

def test_split_same():
    assert x_train1 == x_train2

def test_split_different():
    assert x_train3 == x_train4




