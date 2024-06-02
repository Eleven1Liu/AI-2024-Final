import argparse
import json
import os
import pickle

from datetime import datetime
import pandas as pd
from joblib import parallel_backend

from analyzer import feature_selection, grid_search, kfold_feature_selection, retrain_and_test, retrain_with_best_params_and_test
from analyzer import MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='data/train.csv', help='Path to training data')
    parser.add_argument('--test_data', type=str, default='data/test.csv', help='Path to test data')
    parser.add_argument('--pretrained_feature_file', type=str, default=None, help='Path to pretrained feature.')
    parser.add_argument('--pretrained_grid_file', type=str, default=None,
                        help='Path to pretrained grid search results (e.g., */grid_model.pkl)')
    parser.add_argument('--log_dir', type=str, default='logs', help='Path to test data')
    parser.add_argument('--model_name', type=str, choices=['ada', 'gbr', 'linear', 'linear_cls', 'rf'],
                        default='rf', help='Abbr. for each models')
    parser.add_argument('--min_features', type=int, default=1, help='Minimum number of features')
    parser.add_argument('--max_features', type=int, default=5, help='Maximum number of features')
    parser.add_argument('--val_size', type=float, default=0.2, help='Ratio of the validation set')
    parser.add_argument('--seed', type=int, default=42, help='Seed everywhere')
    parser.add_argument('--kfold', type=int, default=None, help='Number of fold for cross validation')
    parser.add_argument('--prune_threshold', type=float, default=1.9, help='Threshold for feature selection')
    parser.add_argument('--val_metric', type=str, choices=['mae', 'mse'],
                        default='mae', help='Metric for model selection')
    parser.add_argument("--normalize", action="store_true", help="Whether to normalize raw data")

    # emsemble regressor hyper-parameters
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    parser.add_argument('--max_depth', type=int, default=None, help='The maximum depth of the tree')
    parser.add_argument('--oob_score', type=float, default=False,
                        help='Whether to use out-of-bag samples to estimate the generalization score')
    args, _ = parser.parse_known_args()

    # create log dir
    log_dir = f'{args.log_dir}/{datetime.now().strftime("%Y%m%d%H%M%S")}'
    os.makedirs(log_dir, exist_ok=True)
    cpu_count = min(os.cpu_count(), 8)
    config = vars(args)

    # Setup X, y
    y_col = 'available_rent_bikes'
    candidate_feature_cols = ['total', 'mrt_distances', 'person_time', 'dayOfWeek', 'hour',
                              'minute'] + [f'available_rent_bikes_prev_{i}mins' for i in [2, 4, 10, 20, 30, 60]]
    is_ensemble_model = getattr(MODELS[args.model_name], '__module__', '').startswith('sklearn.ensemble')

    # train
    if os.path.exists(args.train_data):
        train_data = pd.read_csv(args.train_data)
        print(f'Finish reading {len(train_data)} train instances from {args.train_data}.')

        with parallel_backend('threading', n_jobs=cpu_count):
            if is_ensemble_model:
                # For models aggregate multiple estimators, no need to do feature selection.
                # Grid search the parameters (for avoiding overfitting issues)
                parameters = {  # hard code for now
                    'n_estimators': [100, 150],
                    'max_depth': [5, 10, 20, 30],
                }
                cls = grid_search(train_data, candidate_feature_cols, args.model_name,
                                  parameters, target_col='available_rent_bikes',
                                  kfold=args.kfold, seed=args.seed)
                best_params = cls.best_params_
                config['best_params'] = best_params
                with open(f'{log_dir}/grid_model.pkl', 'wb') as f:
                    pickle.dump(cls, f)
            else:
                kwargs = {
                    'data': train_data,
                    'model_name': args.model_name,
                    'candidate_features': candidate_feature_cols,
                    'target_col': y_col,
                    'min_feature': args.min_features,
                    'max_feature': args.max_features,
                    'prune_threshold': args.prune_threshold,
                    'normalize': args.normalize,
                    'seed': args.seed
                }
                # feature selection
                if args.kfold is not None:
                    selected_features = kfold_feature_selection(kfold=args.kfold, **kwargs)
                elif args.val_size is not None:
                    selected_features = feature_selection(val_size=args.val_size, **kwargs)

                config['selected_features'] = selected_features
                with open(f'{log_dir}/selected_features.pkl', 'wb') as f:
                    pickle.dump(selected_features, f)

        # dump logs
        with open(f'{log_dir}/logs.json', 'w', encoding="utf-8") as f:
            json.dump(config, f)
    # test
    if os.path.exists(args.test_data):
        test_data = pd.read_csv(args.test_data)

        # Train model with top-k features
        # load from pretrained
        if is_ensemble_model:
            # load from past grid search results
            if args.pretrained_grid_file is not None:
                with open(args.pretrained_grid_file, 'rb') as f:
                    cls = pickle.load(f)
                best_params = cls.best_params_

            retrain_with_best_params_and_test(train_data, test_data, candidate_feature_cols,
                                              args.model_name, best_params=best_params,
                                              target_col=y_col, log_dir=log_dir)

        else:
            # load from pretrained features from file
            if args.pretrained_feature_file is not None:
                with open(args.pretrained_feature_file, 'rb') as f:
                    selected_features = pickle.load(f)
            print(f'Load pretrained features from {args.pretrained_feature_file}')
            retrain_and_test(train_data, test_data, selected_features, args.model_name,
                             target_col=y_col, log_dir=log_dir)


main()
