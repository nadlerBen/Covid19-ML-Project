import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, learning_curve, ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import datetime
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings(action='ignore')


def load_and_split():
    covid = pd.read_csv('project_data_preprocessed.csv')
    covid = covid.drop([' New_cases', ' Cumulative_cases',
                        'Unnamed: 0', 'Date_reported', ' Country'], axis=1)
    scaler = MinMaxScaler()
    X = covid[covid.columns[:-1]].to_numpy()
    X = scaler.fit_transform(X)
    y = covid[covid.columns[-1]].to_numpy()
    return X, y


def create_model(X, y, cv=False):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    lr = LogisticRegression(max_iter=5000, multi_class='multinomial',
                            solver='lbfgs', penalty='l2', C=0.8, class_weight='balanced')
    if cv:
        score = cross_validate(lr, X, y, cv=10,
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
    else:
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        print('Accuracy score for logistic regression: ',
              accuracy_score(y_pred, y_test))
    return lr


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    _, axes = plt.subplots(figsize=(10, 5))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
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
    title = "Learning Curves (Logistic)"
    X, y = load_and_split()
    lr = create_model(X, y, cv=True)
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    plot_learning_curve(lr, title, X, y, ylim=(0.0, 1.00),
                        cv=10, n_jobs=4)

    plt.show()


if __name__ == "__main__":
    main()
