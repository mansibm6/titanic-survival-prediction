"""Reusable experiment functions extracted from kaggle_titanic_competition_submission.ipynb."""

from __future__ import annotations


def plot_count_dist(data, bin_df, label_column, target_column, figsize=(20, 5), use_bin_df=False):
    """
    Function to plot counts and distributions of a label variable and
    target variable side by side.
    ::param_data:: = target dataframe
    ::param_bin_df:: = binned dataframe for countplot
    ::param_label_column:: = binary labelled column
    ::param_target_column:: = column you want to view counts and distributions
    ::param_figsize:: = size of figure (width, height)
    ::param_use_bin_df:: = whether or not to use the bin_df, default False
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if use_bin_df:
        fig = plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        sns.countplot(y=target_column, data=bin_df)
        plt.subplot(1, 2, 2)
        sns.distplot(
            data.loc[data[label_column] == 1][target_column],
            kde_kws={"label": "Survived"},
        )
        sns.distplot(
            data.loc[data[label_column] == 0][target_column],
            kde_kws={"label": "Did not survive"},
        )
    else:
        fig = plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        sns.countplot(y=target_column, data=data)
        plt.subplot(1, 2, 2)
        sns.distplot(
            data.loc[data[label_column] == 1][target_column],
            kde_kws={"label": "Survived"},
        )
        sns.distplot(
            data.loc[data[label_column] == 0][target_column],
            kde_kws={"label": "Did not survive"},
        )


def fit_ml_algo(algo, X_train, y_train, cv):
    from sklearn import model_selection, metrics

    model = algo.fit(X_train, y_train)
    acc = round(model.score(X_train, y_train) * 100, 2)
    train_pred = model_selection.cross_val_predict(algo, X_train, y_train, cv=cv, n_jobs=-1)
    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)
    return (train_pred, acc, acc_cv, model)


def feature_importance(model, data):
    """
    Function to show which features are most important in the model.
    ::param_model:: Which model to use?
    ::param_data:: What data to use?
    """
    import pandas as pd

    fea_imp = pd.DataFrame({"imp": model.feature_importances_, "col": data.columns})
    fea_imp = fea_imp.sort_values(["imp", "col"], ascending=[True, False]).iloc[-30:]
    _ = fea_imp.plot(kind="barh", x="col", y="imp", figsize=(20, 10))
    return fea_imp
