import argparse
import json
import os
import pickle

from datetime import datetime
import pandas as pd
from joblib import parallel_backend

from analyzer import feature_selection, kfold_feature_selection, retrain_and_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='data/train.csv', help='Path to training data')
    parser.add_argument('--test_data', type=str, default='data/test.csv', help='Path to test data')
    parser.add_argument('--pretrained_feature_file', type=str, default=None, help='Path to pretrained feature.')
    parser.add_argument('--log_dir', type=str, default='logs', help='Path to test data')
    parser.add_argument('--model_name', type=str, choices=['ada', 'gbr', 'linear', 'linear_cls', 'rf'],
                        default='rf', help='Abbr. for each models')
    parser.add_argument('--max_features', type=int, default=1, help='Maximum number of features')
    parser.add_argument('--min_features', type=int, default=1, help='Minimum number of features')
    parser.add_argument('--val_size', type=float, default=0.2, help='Ratio of the validation set')
    parser.add_argument('--seed', type=int, default=42, help='Seed everywhere')
    parser.add_argument('--kfold', type=int, default=None, help='Number of fold for cross validation')
    parser.add_argument('--prune_threshold', type=float, default=1.9, help='Threshold for feature selection')
    parser.add_argument('--val_metric', type=str, choices=['mae', 'mse'],
                        default='mae', help='Metric for model selection')
    parser.add_argument("--normalize", action="store_true", help="Whether to normalize raw data")
    args, _ = parser.parse_known_args()

    # create log dir
    log_dir = f'{args.log_dir}/{datetime.now().strftime("%Y%m%d%H%M%S")}'
    os.makedirs(log_dir, exist_ok=True)
    cpu_count = min(os.cpu_count(), 16)

    # Setup X, y
    y_col = 'available_rent_bikes'
    candidate_feature_cols = ['total', 'mrt_distances', 'person_time', 'dayOfWeek', 'hour', 'minute']

    if args.pretrained_feature_file is not None:
        # load from pretrained
        with open(args.pretrained_feature_file, 'rb') as f:
            selected_features = pickle.load(f)
        print(f'Load pretrained features from {args.pretrained_feature_file}')
    else:
        # train
        train_data = pd.read_csv(args.train_data)
        print(f'Finish reading {len(train_data)} train instances from {args.train_data}.')

        with parallel_backend('threading', n_jobs=cpu_count):
            kwargs = {
                'data': train_data,
                'model_name': args.model_name,
                'candidate_features': candidate_feature_cols,
                'target_col': y_col,
                'min_feature': args.min_features,
                'max_feature': args.max_features,
                'prune_threshold': args.prune_threshold,
                'normalize': args.normalize
            }

            if args.kfold is not None:
                selected_features = kfold_feature_selection(kfold=args.kfold, **kwargs)
            elif args.val_size is not None:
                selected_features = feature_selection(val_size=args.val_size, seed=args.seed, **kwargs)

            # dump logs and features
            with open(f'{log_dir}/logs.json', 'w', encoding="utf-8") as f:
                config = vars(args)
                config['selected_features'] = selected_features
                json.dump(config, f)
            with open(f'{log_dir}/selected_features.pkl', 'wb') as f:
                pickle.dump(selected_features, f)

    # test
    if os.path.exists(args.test_data):
        test_data = pd.read_csv(args.test_data)
        retrain_and_test(train_data, test_data, selected_features, args.model_name,
                         target_col=y_col, log_dir=log_dir)


main()
