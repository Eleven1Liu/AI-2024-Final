import collections
import itertools
import pickle

import numpy as np
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import LinearSVC

from data_utils import normalize_features, sample_nonan_data

MODELS = {
    'ada': AdaBoostRegressor(),
    'gbr': GradientBoostingRegressor(loss='absolute_error'),
    'linear': LinearRegression(),
    'linear_cls': LinearSVC(),
    'rf': RandomForestRegressor(),
}

METRICS = {
    'mae': mean_absolute_error,
    'mse': mean_squared_error
}


def do_train(train_data, val_data, features, model_name, target_col='available_rent_bikes', val_metric='mae', normalize=False):
    # drop instances with missing features
    train_data = sample_nonan_data(train_data, features)
    val_data = sample_nonan_data(val_data, features)

    # normalize or not
    if normalize:
        train_data = normalize_features(train_data, features)
        val_data = normalize_features(val_data, features,
                                      min_values=train_data[features].min(),
                                      max_values=train_data[features].max())

    # train
    model = train_by_features(train_data, features, model_name, target_col)

    # validation
    y_preds = predict_with_indexes(
        val_data, features, model, is_regressor=not model_name.endswith('_cls'))
    y_val = val_data[target_col].to_numpy().reshape(-1)
    score = METRICS[val_metric](y_preds, y_val)

    return score


def kfold_feature_selection(data, model_name, candidate_features,
                            target_col='available_rent_bikes',
                            min_feature=1, max_feature=5,
                            prune_threshold=2.0,
                            val_metric='mae', normalize=False,
                            kfold=5, seed=42):
    """Send different feature combinations to the given model."""
    score_dict = dict()
    k_splits = KFold(n_splits=kfold, shuffle=True, random_state=seed)  # shuffle to get instance everywhere

    print(f'Running {kfold}-fold cross validation.')
    for n_features in range(min_feature, max_feature+1):
        feature_combs = itertools.combinations(candidate_features, n_features)
        for features in feature_combs:
            # k-fold cross validation
            val_score = 0.
            for train_idx, val_idx in k_splits.split(data):
                X_train, X_val = data.iloc[train_idx], data.iloc[val_idx]
                score = do_train(X_train, X_val, list(features), model_name, target_col, val_metric, normalize)
                val_score += score
                print(f'{features}: {score:.4f}')

            score_dict[features] = val_score / kfold  # average score
            print(f'{features}: {score_dict[features]:.4f} (mean of {kfold}-fold)')

    # Sort by the validation score on the validation set
    selected_feature_scores = {k: v for k, v in score_dict.items() if v < prune_threshold}
    return sorted(selected_feature_scores.items(), key=lambda item: item[1])


def feature_selection(data, model_name, candidate_features,
                      target_col='available_rent_bikes',
                      min_feature=1, max_feature=5,
                      prune_threshold=2.0,
                      val_metric='mae', normalize=False,
                      val_size=0.2, seed=42):
    """Send different feature combinations to the given model."""

    # train-val split
    X_train, X_val, _, _ = train_test_split(data, data[target_col],
                                            test_size=val_size,
                                            random_state=seed)
    score_dict = dict()
    for n_features in range(min_feature, max_feature+1):
        feature_combs = itertools.combinations(candidate_features, n_features)
        for features in feature_combs:
            score = do_train(X_train, X_val, list(features), model_name, target_col, val_metric, normalize)
            score_dict[features] = score
            print(f'{features}: {score:.4f}')

    # Sort by the MAE score on the validation set
    selected_feature_scores = {k: v for k, v in score_dict.items() if v < prune_threshold}
    return sorted(selected_feature_scores.items(), key=lambda item: item[1])


def retrain_and_test(train_data, test_data, top_features_combinations, model_name, target_col='available_rent_bikes', log_dir='logs'):
    """Retrain with the whole training data and predict data with all features.

    Args:
        train_data (pd.DataFrame): Training data containing features and target values.
        test_data (pd.DataFrame): Test data containing features and target values.
        top_features_combinations (List[tuple]): List of feature column names in `train_data`.
            [   (('total', 'mrt_distances', 'dayOfWeek', 'hour', 'minute'), 0.4941824318323119),
                (('mrt_distances', 'dayOfWeek', 'hour', 'minute'), 0.4962141397992717), ... ]
        model_name (str): Model name defined in `MODEL` (e.g., ada, gbr, linear, rf).
        target_col (str, optional): Y. Defaults to 'available_rent_bikes'.
        log_dir (str, optional): Path to the log directory. Defaults to 'logs'.
    """

    preds_dict = collections.defaultdict(list)

    for features, _ in top_features_combinations:
        features = list(features)
        train_data = sample_nonan_data(train_data, features)
        model = train_by_features(train_data, features, model_name, target_col)

        # drop test data with missing values (TBD: guess with other combinations or predict with missing values)
        # figure out how to deal with missing values
        sampled_test_data = sample_nonan_data(test_data, features)
        y_preds = predict_with_indexes(
            sampled_test_data, features, model, is_regressor=not model_name.endswith('_cls'))
        preds_dict['features'].append(features)
        preds_dict[f'pred_{target_col}'].append(y_preds)

    # pandas is too slow here, use pkl
    filename = f'{log_dir}/predictions.pkl'
    print(f'Write predictions to {filename}.')
    with open(filename, 'wb') as f:
        pickle.dump(preds_dict, f)
    # pd.DataFrame(preds_dict).to_csv(filename, index=False)


def train_by_features(train_data, features, model_name, target_col='available_rent_bikes'):
    """Train model.

    Args:
        train_data (pd.DataFrame): Training data containing features and target values.
        features (List[str]): List of feature column names in `train_data`.
        model_name (str): Model name defined in `MODEL` (e.g., ada, gbr, linear, rf).
        target_col (str, optional): Y. Defaults to 'available_rent_bikes'.

    Returns:
        sklearn.*model: model trained by the given features
    """
    X = np.array(train_data[list(features)].values.tolist())
    y = train_data[target_col].to_numpy().reshape(-1)
    model = MODELS[model_name]
    model.fit(X, y)
    return model


def predict_with_indexes(test_data, features, model, is_regressor=True):
    """Test model

    Args:
        test_data (pd.DataFrame): Test data containing features and target values.
        features (List[str]): List of feature column names in `train_data`.
        model (sklearn.*model): A sklearn model.
        is_regressor (bool, optional): Whether the given model is a regressor or not. Defaults to True.

    Returns:
        numpy.ndarray: Prediction with shape (# test instances,)
    """
    X_test = np.array(test_data[list(features)].values.tolist())
    y_preds = model.predict(X_test)
    if is_regressor:
        y_preds = np.rint(y_preds)
    return y_preds
