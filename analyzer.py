import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split  # TBD: KFold
from sklearn.svm import LinearSVC

from data_utils import normalize_features, sample_nonan_data

MODELS = {
    'ada': AdaBoostRegressor(),
    'gbr': GradientBoostingRegressor(loss='absolute_error'),
    'linear': LinearRegression(),
    'linear_cls': LinearSVC(),
    'rf': RandomForestRegressor(),
}


def feature_selection(data, model_name, candidate_features,
                      target_col='available_rent_bikes',
                      min_feature=1, max_feature=5,
                      val_size=0.2, seed=42,
                      mae_threshold=2.0, normalize=False):
    """Send different feature combinations to the given model."""
    score_dict = dict()

    for n_features in range(min_feature, max_feature+1):
        feature_combs = itertools.combinations(candidate_features, n_features)

        for features in feature_combs:
            features = list(features)
            df_sample = sample_nonan_data(data, features)

            # train-val split
            X_train, X_val, y_train, y_val = train_test_split(df_sample, df_sample[target_col],
                                                              test_size=val_size,
                                                              random_state=seed)
            if normalize:
                X_train = normalize_features(X_train, features)
                X_val = normalize_features(X_val, features,
                                           min_values=X_train[features].min(),
                                           max_values=X_train[features].max())

            # train
            model = train_by_features(X_train, features, model_name, target_col)

            # validation
            _, y_preds = predict_with_indexes(X_val, features, model, is_regressor=not model_name.endswith('_cls'))
            df_sample_val = sample_nonan_data(X_val, features)
            y_val = df_sample_val[target_col].to_numpy().reshape(-1)

            score = mean_absolute_error(y_preds, y_val)
            score_dict[tuple(features)] = score
            print(f'{features} ({X_train.shape[0]}, {X_val.shape[0]}): {score:.4f}')

    # Sort by the MAE score on the validation set
    selected_feature_scores = {k: v for k, v in score_dict.items() if v < mae_threshold}
    return sorted(selected_feature_scores.items(), key=lambda item: item[1])


def retrain_and_test(train_data, test_data, top_features_combinations, model_name, target_col='available_rent_bikes'):
    """Retrain with the whole training data and predict data with all features.
    TBD: dump model

    Args:
        train_data (pd.DataFrame): Training data containing features and target values.
        test_data (pd.DataFrame): Test data containing features and target values.
        top_features_combinations (List[tuple]): List of feature column names in `train_data`.
            [   (('total', 'mrt_distances', 'dayOfWeek', 'hour', 'minute'), 0.4941824318323119),
                (('mrt_distances', 'dayOfWeek', 'hour', 'minute'), 0.4962141397992717), ... ]
        model_name (str): Model name defined in `MODEL` (e.g., ada, gbr, linear, rf).
        target_col (str, optional): Y. Defaults to 'available_rent_bikes'.
    """

    for features, _ in top_features_combinations:
        features = list(features)
        model = train_by_features(train_data, features, model_name, target_col)
        y_preds = predict_with_indexes(
            test_data, features, model, is_regressor=not model_name.endswith('_cls'))

        print(y_preds.shape)
        # TBD: rewrite logging if there is time ...
        new_preds = test_data.copy()
        new_preds['pred_{target_col}'] = y_preds
        filename = f'logs/{"_".join(features)}_preds.csv'
        test_data.to_csv(filename, index=False)
        print(f'Write predictions to {filename} ({features}).')


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
    df_sample = sample_nonan_data(train_data, features)
    X = np.array(df_sample[list(features)].values.tolist())
    y = df_sample[target_col].to_numpy().reshape(-1)
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
    df_sample = sample_nonan_data(test_data, features)
    X_test = np.array(df_sample[list(features)].values.tolist())
    y_preds = model.predict(X_test)
    if is_regressor:
        y_preds = np.rint(y_preds)
    return y_preds
