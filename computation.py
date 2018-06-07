import models as dbHandler
import csv
import numpy as np
import time
from sklearn import svm
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

def build_model(csv_file, classifier_type, file_name, num_features):
    errors = []
    id = -1
    data = []
    file_path = "./data_science/models/" + file_name + ".pkl"

    for line in csv_file.readlines():
        data.append(line.rstrip().split(","))

    X_train = [d[1:] for d in data]
    y_train = [d[0] for d in data]

    for x in X_train:
        if len(x) != int(num_features):
            errors.append("Number of features in cvs file does not match number of features expected (" + str(num_features) + ")")
            break

    clf = train_classifier(classifier_type, X_train, y_train)
    joblib.dump(clf, file_path)
    # test_input = ['12.34', '2.45', '2.46', '21.0', '98.0', '2.56', '2.11', '0.34', '1.31', '2.8', '0.8', '3.38', '438.0']

    id = dbHandler.insertModel(classifier_type, num_features, file_path)
    if id == -1:
        errors.append("Couldn't save model to database")

    return id, errors

def get_wine_data(test_size=0.30):
    RANDOM_STATE = 42
    features, target = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                        test_size=test_size,
                                                        random_state=RANDOM_STATE)
    return X_train, X_test, y_train, y_test


def make_prediction(classifier_type, features):
    cleaned_features = clean_features(features)
    clf = train_classifier_with_wine(classifier_type)
    res = clf.predict([cleaned_features])

    return "type " + str(res[0])

def clean_features(features):
    return [try_convert_to_float(f) for f in features]

def try_convert_to_float(f):
    try:
        return float(f)
    except ValueError:
        return 0.0

def train_classifier_with_wine(classifier_type):
    X_train, _, y_train, _ = get_wine_data()
    return train_classifier(classifier_type, X_train, y_train)

def train_classifier(classifier_type, X_train, y_train):
    classifiers = [
        {
            "name": "svm",
            "model": svm.SVC(decision_function_shape='ovo'),
        },
        {
            "name": "rf",
            "model": RandomForestClassifier(max_depth=2, random_state=0),
        },
        {
            "name": "gp",
            "model": GaussianProcessClassifier(1.0 * RBF(1.0)),
        },
        {
            "name": "nn",
            "model": MLPClassifier(alpha=1),
        },
    ]

    clf = [c for c in classifiers if c["name"] == classifier_type][0]["model"]
    clf.fit(X_train, y_train)

    return clf



# DEBUG: code below isn t needed for APIs

def write_csv():
    X_train, _, y_train, _ = get_wine_data(0.0)
    data = zip(X_train, y_train)
    with open('./data_science/train_data/wine.csv', 'wb') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for (X_train, y_train) in data:
            # print [y_train] + [x for x in X_train]
            line = [y_train] + [x for x in X_train]
            # print line
            writer.writerow(line)

def svm_test(X, y, input):
    clf = svm.SVC()
    clf.fit(X, y)

    return clf.predict(input)

# def get_svm_classifier(X_train, y_train):
#     clf = svm.SVC(decision_function_shape='ovo')
#     clf.fit(X_train, y_train)
#
#     return clf
#
# def get_random_forest_classifier(X_train, y_train):
#     clf = RandomForestClassifier(max_depth=2, random_state=0)
#     clf.fit(X_train, y_train)
#
#     return clf
#
# def get_gaussian_process_classifier(X_train, y_train):
#     clf = GaussianProcessClassifier(1.0 * RBF(1.0))
#     clf.fit(X_train, y_train)
#
#     return clf

# def fit_classifier_model(clf, X_train, y_train):
#     return clf.fit(X_train, y_train)

# MLPClassifier(alpha=1),

def test_classifier(clf, X_test, y_test):
    num_correct = 0
    for i, x in enumerate(X_test):
        output = clf.predict([x])
        expected = y_test[i]
        if output == [expected]:
            num_correct = num_correct + 1
    return float(num_correct) / float(len(y_test))


def predict_wine():
    X_train, X_test, y_train, y_test = get_wine_data()
    # for i in range(13):
    #     print "feature " + str(i)
    #     print str([x[i] for x in [X_train[0], X_train[20], X_train[30], X_train[40]]])
    #
    # print str([(i, y) for (i, y) in enumerate(y_train)])

    classifiers = [
        {
            "name": "svm",
            "model": svm.SVC(decision_function_shape='ovo'),
            "results": [],
        },
        {
            "name": "rf ",
            "model": RandomForestClassifier(max_depth=2, random_state=0),
            "results": [],
        },
        {
            "name": "guassian process",
            "model": GaussianProcessClassifier(1.0 * RBF(1.0)),
            "results": [],
        },
        {
            "name": "nn ",
            "model": MLPClassifier(alpha=1),
            "results": [],
        },
        # ("svm", svm.SVC(decision_function_shape='ovo')),
        # ("random forest",RandomForestClassifier(max_depth=2, random_state=0)),
        # ("guassian process", GaussianProcessClassifier(1.0 * RBF(1.0))),
        # ("neural network", MLPClassifier(alpha=1),)
    ]


    for c in classifiers:
        clf = c["model"].fit(X_train, y_train)
        print c["name"]
        for i in [0, 1, 3, 4, 5, 6]:
            print X_train[i]
            print clf.predict([X_train[i]])
        # print
        # print X_train[0]
        # print clf.predict([[u'13.49', u'3.59', u'2.19', u'13.49', u'88.0', u'1.62', u'19.5', u'88.0', u'0.88', u'0.48', u'0.58', u'0.88', u'5.7']])
        # print clf.predict([[u'13.49', u'3.59', u'2.19', u'19.50', u'88.0', u'1.62', u'0.48', u'0.58', u'0.88', u'5.7', u'0.81', u'1.82', u'5.800']])
        # print "Accuracy of " + clf_name + ": " + str(result)

    # num_iteration = 100
    #
    #
    # for c in classifiers:
    #     t0 = time.time()
    #     for _ in range(num_iteration):
    #         clf = c["model"].fit(X_train, y_train)
    #         result = test_classifier(clf, X_test, y_test)
    #         c["results"].append(result)
    #     c["time"] = time.time() - t0
    #
    # print "Over " + str(num_iteration) + " iterations:"
    # for c in classifiers:
    #     mean = np.mean(c["results"])
    #     variance = np.var(c["results"])
    #     # print c["name"]
    #     # print c["results"]
    #     # print mean, variance
    #     print "Accuracy of " + c["name"] + ":\t" + str(mean) + "\twith variance:\t" + str(variance) + "\in " + str(c["time"]) + " s"

if __name__ == "__main__":
    # predict_wine()
    write_csv()
