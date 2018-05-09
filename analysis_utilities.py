import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

DEFAULT_CLASSIFIERS = {
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


def evaluate(target_feature, dataset="train_final.csv", id_feature="Id", classifiers=DEFAULT_CLASSIFIERS):
    train_df = pd.read_csv(dataset)
    x = train_df.drop(target_feature, axis=1).copy()
    if id_feature in list(x):
        x = x.drop(id_feature, axis=1).copy()
    y = train_df[target_feature]
    cv = ShuffleSplit(n_splits=200, test_size=0.25, random_state=0)
    for name, classifier in classifiers.items():
        filename = name + ".png"
        plot_learning_curve(classifier, x, y, cv=cv, title="Learning Curve " + name).savefig(
            "score/learning curve/" + filename)


def plot_validation_curve(estimator, X, y, param_name, param_range, title="Validation Curve"):
    # param_range = np.logspace(-6, -1, 5)
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=10, scoring="accuracy", n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    return plt


def plot_learning_curve(estimator, X, y, title="Learning Curve", ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def plot_feature_importance(classifier, X, Y, title="Feature Importance"):
    importances = classifier.fit(X, Y).feature_importances_
    indices = np.argsort(importances)[::-1]

    # Plot the feature importances of the forest
    plt.figure()
    plt.title(title)
    plt.bar(range(X.shape[1]), importances[indices], color="r", align="center")
    plt.xticks(range(X.shape[1]), list(X.columns[indices]))
    plt.xlim([-1, X.shape[1]])
    return plt


def plot_regression_for_numerical_features(X_feature_name, Y_feature_name, data, is_optional_feature=False, order=1):
    values = data[data[X_feature_name].notna()][[X_feature_name, Y_feature_name]]
    if is_optional_feature:
        values = values[values[X_feature_name] != 0]
    fig = plt.figure(figsize=(20, 20))
    ax1 = fig.add_subplot(221)
    ax1.set_title("x-y")
    ax2 = fig.add_subplot(222)
    ax2.set_title("log(x)-y")
    ax3 = fig.add_subplot(223)
    ax3.set_title("log(x)-log(y)")
    ax4 = fig.add_subplot(224)
    ax4.set_title("x-log(y)")
    sns.regplot(x=values[X_feature_name], y=values[Y_feature_name], color="g", ax=ax1, order=order)
    sns.regplot(x=np.log(values[X_feature_name]), y=values[Y_feature_name], color="g", ax=ax2, order=order)
    sns.regplot(x=np.log(values[X_feature_name]), y=np.log(values[Y_feature_name]), color="g", ax=ax3, order=order)
    sns.regplot(x=values[X_feature_name], y=np.log(values[Y_feature_name]), color="g", ax=ax4, order=order)
    return plt
