#python trainer.py <path_to_training_data> <path_to_classifier>
#e.g. trainer.py ../training_data.dat ./classifier.dat
#################################################################
#
# Write for COMP9318 project
# trainer.py is used to training a classifier by input training data
# modified by Nijie Sun from ipython Notebooks examples in COMP9318 website
#
##################################################################

import sys
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

import matplotlib.pyplot as plt

from tester import add_features

#%matplotlib inline
#def test_LogisticRegression(train_X, train_y, test_X, test_y, debug=False):
def test_LogisticRegression(train_X, train_y, test_X, test_y, test_lr):
    # penalty='l1' is better than 'l2'. Why???
    #test_lr = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
                            #class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr',
                            #verbose=0, warm_start=False, n_jobs=1)

    #test_lr.fit_transform(train_X, train_y)
    test_lr.fit(train_X, train_y)
    train_error = test_lr.score(train_X, train_y)
    test_error = test_lr.score(test_X, test_y)
    #print(test_lr.predict_proba(test_X))
    #print (test_lr.predict(test_X))
    predict_y = test_lr.predict(test_X)
    f1_value = f1_score(y_true=test_y, y_pred=predict_y, pos_label=None, average='macro')
    #f1_value = f1_score(y_true=test_y, y_pred=predict_y, average='binary')

    #print("f1:", f1_value)
    #print (test_y)
    #if debug:
    #    print('training error:\t{}'.format(train_error))
    #    print('testing error:\t{}'.format(test_error))
    #with open('./classifier.dat', 'wb') as f:
    #    pickle.dump(test_lr, f)

    return train_error, test_error, f1_value, test_lr

def cv_LogisticRegression(classifier):
    train_error_total = 0
    test_error_total = 0
    f1_total = 0
    for train, test in kf:
        train_X = X[train]
        test_X = X[test]
        train_y = y[train]
        test_y = y[test]
        train_error, test_error, f1v, lr = test_LogisticRegression(train_X, train_y, test_X, test_y, classifier)
        train_error_total += train_error
        test_error_total += test_error
        f1_total += f1v
    #print('===================')
    #print('avg. training error:\t{}'.format(train_error_total / n_folds))
    #print('avg. testing error:\t{}'.format(test_error_total / n_folds))
    #print('avg. f1:\t{}'.format(f1_total / n_folds))

    return train_error_total / n_folds, test_error_total / n_folds, lr

def set_parameter(test_penalty, test_C):
    print(test_penalty, ' ', test_C)
    p_lr = LogisticRegression(penalty=test_penalty, dual=False, tol=0.0001, C=test_C, fit_intercept=True, intercept_scaling=1,
                            class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr',
                            verbose=0, warm_start=False, n_jobs=1)
    avg_train_error, avg_test_error, classifier = cv_LogisticRegression(p_lr)

if __name__ == '__main__':

    if (3 != len(sys.argv)):
        sys.stderr.write('Usage:' + '\n')
        sys.stderr.write(sys.argv[0] + ' path_to_training_data path_to_classifier' + '\n')
        exit()


    training_data_path = sys.argv[1]
    with open(training_data_path, 'rb') as f:
        training_set = pickle.load(f)

    training_set_features = add_features(training_set, is_tester=False)

    training_data = training_set_features[0][:]
    for i in range(len(training_set_features)):
        if i != 0:
            training_data.extend(training_set_features[i])

    #df = pd.DataFrame(training_data, columns=['token', 'tag', 'pre1tag', 'pre1result', 'pre2tag', 'pre2result', 'next1tag', 'next1result', 'next2tag', 'next2result', 'pre1isTITLE', 'uppercase', 'inTitlelist', 'result'])
    #df = pd.DataFrame(training_data,
    #                  columns=['token', 'tag', 'pre1tag', 'pre1result', 'pre2tag', 'pre2result', 'next1tag',
    #                           'next1result', 'next2tag', 'next2result', 'pre1isTITLE', 'next1isTITLE', 'uppercase', 'inTitlelist',
    #                           'result'])
    #df = pd.DataFrame(training_data,
    #                  columns=['token', 'tag', 'pre1tkoen', 'pre1tag', 'pre2tag', 'next1tag',
    #                           'next2tag', 'uppercase',
    #                           'inTitlelist', 'result'])
    df = pd.DataFrame(training_data,
                      columns=['token', 'tag', 'tokenissingular', 'pre1token', 'pre1tag', 'pre2token', 'pre2tag', 'next1token', 'next1tag',
                               'next2token', 'next2tag',
                               'uppercase',
                               'inTitlelist',
                               'next1inTitlelist', 'next2inTitlelist' ,
                               'result'])

    X = df.drop('result', 1)
    #X = X.drop('tag', 1)
    #for a in X:
    #    print(a)
    #print(X)
    X = X.T.to_dict().values()
    #print(X)
    vectorizer = DictVectorizer(dtype=float, sparse=True)
    X = vectorizer.fit_transform(X)

    y = df['result']
    #for b in y:
    #    print(b)
    #print(y)
    #encoder = LabelEncoder()
    #y = encoder.fit_transform(y)
    #for b in y:
    #    print(b)
    #print(y)

    n_folds = 10
    kf = KFold(n=X.shape[0], n_folds=n_folds, shuffle=True, random_state=42)
    #lr = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
    #                        class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr',
    #                        verbose=0, warm_start=False, n_jobs=1)
    lr = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=15.0, fit_intercept=True, intercept_scaling=1,
                            class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr',
                            verbose=0, warm_start=False, n_jobs=1)
    #lr = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=3.9, fit_intercept=True, intercept_scaling=1,
    #                        class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr',
    #                        verbose=0, warm_start=False, n_jobs=1)
    avg_train_error, avg_test_error, classifier = cv_LogisticRegression(lr)

    #Cs = [0.001, 0.01, 0.1, 1.0, 10.0, 15.0, 100.0, 1000.0]
    #Cs = [1.0, 5.0, 10.0, 13.0, 15.0, 25.0, 53.0]
    #Cs = [5.0, 8.0, 10.0, 13.0, 15.0, 25.0, 30.0, 35.0]
    #for c in Cs:
    #    set_parameter('l1', c)
    #for c in Cs:
    #    set_parameter('l2', c)

    classifier_path = sys.argv[2]
    with open(classifier_path, 'wb') as classifier_f:
       pickle.dump(classifier, classifier_f)

    with open('./vectorizer.dat', 'wb') as vectorizer_f:
        pickle.dump(vectorizer, vectorizer_f)