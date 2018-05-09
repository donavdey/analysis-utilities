import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

import plot_utilities as pu


def evaluate(target_feature, dataset="train_final.csv", id_feature="Id"):
    train_df = pd.read_csv(dataset)
    x = train_df.drop(target_feature, axis=1).copy()
    if id_feature in list(x):
        x = x.drop(id_feature, axis=1).copy()
    y = train_df[target_feature]

    classifiers = {
        "LogisticRegression": LogisticRegression(),
        "LogisticRegression C=10": LogisticRegression(C=10.0),
        "LogisticRegression C=100": LogisticRegression(C=100.0),
        "SVC": SVC(),
        "LinearSVC": LinearSVC(),
        "RandomForestClassifier": RandomForestClassifier(),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "GaussianNB": GaussianNB(),
        "Perceptron": Perceptron(),
        "SGDClassifier": SGDClassifier(),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "MLPClassifier Alpha 1e-10": MLPClassifier(solver='lbfgs', alpha=1e-10, hidden_layer_sizes=(5, 2),
                                                   random_state=1),
        "MLPClassifier Alpha 1e-07": MLPClassifier(solver='lbfgs', alpha=1e-7, hidden_layer_sizes=(5, 2),
                                                   random_state=1),
        "MLPClassifier Alpha 1e-05": MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2),
                                                   random_state=1),
        "MLPClassifier Alpha 0.001": MLPClassifier(solver='lbfgs', alpha=0.001, hidden_layer_sizes=(5, 2),
                                                   random_state=1),
        "MLPClassifier Alpha 0.1": MLPClassifier(solver='lbfgs', alpha=0.1, hidden_layer_sizes=(5, 2), random_state=1),
        "MLPClassifier Alpha 10": MLPClassifier(solver='lbfgs', alpha=10.0, hidden_layer_sizes=(5, 2), random_state=1),
        "MLPClassifier Alpha 1000": MLPClassifier(solver='lbfgs', alpha=1000, hidden_layer_sizes=(5, 2),
                                                  random_state=1),
        "MLPClassifier Alpha 1e-05 hidden layers size (5, 3)": MLPClassifier(solver='lbfgs', alpha=1e-5,
                                                                             hidden_layer_sizes=(5, 3), random_state=1),
        "MLPClassifier Alpha 1e-05 hidden layers size (10, 5)": MLPClassifier(solver='lbfgs', alpha=1e-5,
                                                                              hidden_layer_sizes=(10, 5),
                                                                              random_state=1),
        "MLPClassifier Alpha 1e-07 hidden layers size (10, 5)": MLPClassifier(solver='lbfgs', alpha=1e-7,
                                                                              hidden_layer_sizes=(10, 5),
                                                                              random_state=1),
        "MLPClassifier Alpha 1e-05 hidden layers size (8, 2)": MLPClassifier(solver='lbfgs', alpha=1e-5,
                                                                             hidden_layer_sizes=(8, 2), random_state=1),
        "MLPClassifier Alpha 1e-05 hidden layers size (10, 2)": MLPClassifier(solver='lbfgs', alpha=1e-5,
                                                                              hidden_layer_sizes=(10, 2),
                                                                              random_state=1),
        "MLPClassifier Alpha 1e-05 hidden layers size (8, 3)": MLPClassifier(solver='lbfgs', alpha=1e-5,
                                                                             hidden_layer_sizes=(8, 3), random_state=1)
    }

    cv = ShuffleSplit(n_splits=200, test_size=0.25, random_state=0)
    for name, classifier in classifiers.items():
        filename = name + ".png"
        pu.plot_learning_curve(classifier, x, y, cv=cv, title="Learning Curve " + name).savefig(
            "score/learning curve/" + filename)
