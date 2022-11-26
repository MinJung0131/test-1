#PLEASE WRITE THE GITHUB URL BELOW!
#https://github.com/MinJung0131/test-1.git

import sys
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def load_dataset(dataset_path):
    dataset_df = pd.read_csv(dataset_path)
    return dataset_df    

def dataset_stat(dataset_df):	
    n_feats = dataset_df.count()
    n_class0 = dataset_df.groupby("target").size()['0']
    n_class1=dataset_df.groupby("target").size()['1']
    return n_feats, n_class0, n_class1

def split_dataset(dataset_df, testset_size):
    X = dataset_df.drop(columns="target", axis=1)
    y= dataset_df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testset_size, random_state=2)
    return X_train, X_test, y_train, y_test
    

def decision_tree_train_test(x_train, x_test, y_train, y_test):
    dt_cls = DecisionTreeClassfier()
    dt_cls.fit(X_train, y_train)
    accuracy = accuracy_score(dt_cls.predict(X_test), y_test)
    precision = sklearn.metrics.precision_score(y_test, dt_cls.predict(X_test),average='binary')
    recall = sklearn.metrics.recall_score(y_test, dt_cls.predict(X_test), average='binary')
    return accuracy, precision, recall


def random_forest_train_test(x_train, x_test, y_train, y_test):
    rf_cls = RandomForestClassifier()
    re_cls.fit(X_train, y_train)
    accuracy = accuracy_score(rf_cls.predict(X_test), y_test)
    precision = sklearn.metrics.precision_score(y_test, rf_cls.predict(X_test),average='binary')
    recall = sklearn.metrics.recall_score(y_test, rf_cls.predict(X_test),average='binary')
    return accuracy, precision, recall


def svm_train_test(x_train, x_test, y_train, y_test):
    svm_cls =SVC()
    svm_cls(X_train, y_train)
    accuracy = accuracy_score(svm_cls.predict(X_test), y_test)
    precision = sklearn.metrics.precision_score(y_test, svm_cls.predict(X_test), average='binary')
    recall = sklearn.metrics.recall_score(y_test, svm_cls.predict(X_test), average='binary')
    return accuracy, precision, recall


def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)
