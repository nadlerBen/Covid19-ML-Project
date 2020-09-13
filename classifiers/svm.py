import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_validate, TimeSeriesSplit, train_test_split, learning_curve
import datetime
import warnings
warnings.filterwarnings(action='ignore')


def load_split_data(start_date: str, end_date: str):
    covid = pd.read_csv('project_data_preprocessed.csv')
    covid['Date'] = pd.to_datetime(covid['Date_reported'])
    covid = covid.set_index(covid['Date'])

    train = covid.loc[start_date:end_date]
    test = covid.loc[end_date:]
    print('Train Dataset:', train.shape)
    print('Test Dataset:', test.shape)

    train = train.drop([' New_cases', ' Cumulative_cases', 'Unnamed: 0', 'Date_reported',
                        ' Country', 'Date'], axis=1)
    test = test.drop([' New_cases', ' Cumulative_cases', 'Unnamed: 0', 'Date_reported',
                      ' Country', 'Date'], axis=1)
    covid = covid.drop([' New_cases', ' Cumulative_cases', 'Unnamed: 0', 'Date_reported',
                        ' Country', 'Date'], axis=1)
    return covid, train, test


def train_test(train, test):
    X_train, X_test, y_train, y_test = train[train.columns[:-1]].to_numpy(), test[test.columns[:-1]].to_numpy(
    ), train[train.columns[-1]].to_numpy(), test[test.columns[-1]].to_numpy()
    return X_train, X_test, y_train, y_test


def create_model(data, train, test, cv=False):
    X = data[data.columns[:-1]].to_numpy()
    y = data[data.columns[-1]].to_numpy()

    svc = SVC(C=0.5, kernel='linear', probability=True,
              class_weight='balanced', max_iter=10000)
    if cv:
        eval_model(data, svc, X, y, cv=True)
        title = "Learning Curves (SVM)"
        plot_learning_curve(svc, title, X, y, ylim=(0.0, 1.00),
                            cv=TimeSeriesSplit(), n_jobs=4)
    else:
        X_train, X_test, y_train, y_test = train_test(train, test)
        svc.fit(X_train, y_train)
        print('########################')
        print('Evaluation of train: ')
        print('########################')
        eval_model(data, svc, X_train, y_train)
        print('########################')
        print('Evaluation of test: ')
        print('########################')
        eval_model(data, svc, X_test, y_test)
    return svc


def eval_model(data, estimator, X, y, cv=False):
    if cv:
        predictions = list()
        tscv = TimeSeriesSplit()
        score = cross_validate(estimator, X, y, cv=tscv,
                               scoring=(
                                   'f1_weighted', 'accuracy', 'precision_macro', 'recall_macro'),
                               return_train_score=True)
        print('Mean train and test accuracy scores:',
              score['train_accuracy'].mean(), score['test_accuracy'].mean())
        print('Mean train and test f1 scores:', score['train_f1_weighted'].mean(
        ), score['test_f1_weighted'].mean())
        print('Mean train and test precision scores:', score['train_precision_macro'].mean(
        ), score['test_precision_macro'].mean())
        print('Mean train and test recall scores:', score['train_recall_macro'].mean(
        ), score['test_recall_macro'].mean())

        count = 0
        for train_index, test_index in tscv.split(X):
            if count == 0:
                estimator.fit(X[train_index], y[train_index])
                predictions.extend(estimator.predict(X[train_index]))
                count += 1
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            prediction = predict_cross_val(estimator, X_train, X_test, y_train)
            predictions.extend(prediction)
    else:
        predictions = estimator.predict(X)
    error = 1.0 - float(accuracy_score(y, predictions))
    print("Accuracy: " + str(accuracy_score(y, predictions)))
    print("Confusion Matrix:")
    print(confusion_matrix(y, predictions))
    print("Classification Report:")
    print(classification_report(y, predictions))
    return error


def predict_cross_val(estimator, xtrain, xtest, ytrain):
    estimator.fit(xtrain, ytrain)
    return estimator.predict(xtest)


def grid_search(estimator, parameters, X, y):
    clf = GridSearchCV(estimator, parameters)
    clf.fit(X, y)
    print(clf.cv_results_)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    _, axes = plt.subplots(figsize=(10, 5))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                            train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1,
                      color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1,
                      color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
              label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
              label="Cross-validation score")
    axes.legend(loc="best")

    return plt


def main():
    covid, train, test = load_split_data('2020-01-01', '2020-08-01')
    svc = create_model(covid, train, test, cv=True)
    plt.show()


if __name__ == "__main__":
    main()
