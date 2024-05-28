import argparse
import os

import pandas as pd
from joblib import parallel_backend

from analyzer import feature_selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='data/train.csv', help='Path to training data')
    parser.add_argument('--test_data', type=str, default='data/test.csv', help='Path to test data')
    parser.add_argument('--model_name', type=str, default='rf', help='Abbr. for each models')
    parser.add_argument('--max_features', type=int, default=5, help='Maximum number of features')
    parser.add_argument('--min_features', type=int, default=1, help='Minimum number of features')
    parser.add_argument('--val_size', type=float, default=0.2, help='Ratio of the validation set')
    parser.add_argument('--seed', type=int, default=42, help='Seed everywhere')
    parser.add_argument('--kfold', type=int, default=5, help='Number of fold for cross validation')
    parser.add_argument('--mae_threshold', type=float, default=1.9, help='MAE threshold for feature selection')
    parser.add_argument("--normalize", action="store_true", help="Whether to normalize raw data")
    args, _ = parser.parse_known_args()

    # create directories
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    # read data
    train_data = pd.read_csv(args.train_data)
    # test_data = pd.read_csv(args.test_data) # TBD

    y_col = 'available_rent_bikes'
    candidate_feature_cols = ['total', 'mrt_distances', 'person_time', 'dayOfWeek', 'hour', 'minute']
    cpu_count = min(os.cpu_count(), 16)
    with parallel_backend('threading', n_jobs=cpu_count):
            selected_features = feature_selection(train_data, args.model_name,
                                                  candidate_feature_cols,
                                                  target_col=y_col,
                                                  min_feature=args.min_features,
                                                  max_feature=args.max_features,
                                                  val_size=args.val_size,
                                                  seed=args.seed,
                                                  mae_threshold=args.mae_threshold,
                                                  normalize=args.normalize)
            print(selected_features)

main()
