import argparse
import os

import pandas as pd
from joblib import parallel_backend

from analyzer import feature_selection, retrain_and_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='data/train.csv', help='Path to training data')
    parser.add_argument('--test_data', type=str, default='data/test.csv', help='Path to test data')
    parser.add_argument('--log_dir', type=str, default='logs', help='Path to test data')
    parser.add_argument('--model_name', type=str, choices=['ada', 'gbr', 'linear', 'linear_cls', 'rf'],
                        default='rf', help='Abbr. for each models')
    parser.add_argument('--max_features', type=int, default=5, help='Maximum number of features')
    parser.add_argument('--min_features', type=int, default=1, help='Minimum number of features')
    parser.add_argument('--val_size', type=float, default=0.2, help='Ratio of the validation set')
    parser.add_argument('--seed', type=int, default=42, help='Seed everywhere')
    parser.add_argument('--kfold', type=int, default=5, help='Number of fold for cross validation')
    parser.add_argument('--mae_threshold', type=float, default=1.9, help='MAE threshold for feature selection')
    parser.add_argument("--normalize", action="store_true", help="Whether to normalize raw data")
    args, _ = parser.parse_known_args()

    # create log dir
    os.makedirs(args.log_dir, exist_ok=True)

    # train
    train_data = pd.read_csv(args.train_data)

    y_col = 'available_rent_bikes'
    candidate_feature_cols = ['total', 'mrt_distances', 'person_time', 'dayOfWeek', 'hour', 'minute']
    cpu_count = min(os.cpu_count(), 16)
    with parallel_backend('threading', n_jobs=cpu_count):
        # selected_features = feature_selection(train_data, args.model_name,
        #                                       candidate_feature_cols,
        #                                       target_col=y_col,
        #                                       min_feature=args.min_features,
        #                                       max_feature=args.max_features,
        #                                       val_size=args.val_size,
        #                                       seed=args.seed,
        #                                       mae_threshold=args.mae_threshold,
        #                                       normalize=args.normalize)
        # print(selected_features)

        selected_features = [
            (('total', 'mrt_distances', 'dayOfWeek', 'hour', 'minute'), 0.4941824318323119),
            (('mrt_distances', 'dayOfWeek', 'hour', 'minute'), 0.4962141397992717),
            (('mrt_distances', 'person_time', 'dayOfWeek', 'hour', 'minute'), 0.5264069758586551),
            (('total', 'mrt_distances', 'person_time', 'dayOfWeek', 'minute'), 0.6680057370598838),
            (('mrt_distances', 'person_time', 'dayOfWeek', 'minute'), 0.6698334729520473),
            (('total', 'mrt_distances', 'person_time', 'hour', 'minute'), 0.9269984515015358),
            (('mrt_distances', 'person_time', 'hour', 'minute'), 0.9330274414236032),
            (('total', 'mrt_distances', 'person_time', 'minute'), 1.0231767065214632),
            (('mrt_distances', 'person_time', 'minute'), 1.0269844896301374),
            (('total', 'mrt_distances', 'dayOfWeek', 'hour'), 1.3655075939248602),
            (('mrt_distances', 'dayOfWeek', 'hour'), 1.365701882938094),
            (('mrt_distances', 'person_time', 'dayOfWeek', 'hour'), 1.4984388089254437),
            (('total', 'mrt_distances', 'person_time', 'dayOfWeek', 'hour'), 1.498718046353413),
            (('total', 'mrt_distances', 'person_time', 'dayOfWeek'), 1.5674358388546188),
            (('mrt_distances', 'person_time', 'dayOfWeek'), 1.5686479831442135),
            (('total', 'mrt_distances', 'person_time', 'hour'), 1.772554768613713),
            (('mrt_distances', 'person_time', 'hour'), 1.772814967126139),
            (('mrt_distances', 'person_time'), 1.831048663468129),
            (('total', 'mrt_distances', 'person_time'), 1.831258091539106)
        ]

        if os.path.exists(args.test_data):
            test_data = pd.read_csv(args.test_data)
            retrain_and_test(train_data, test_data, selected_features, args.model_name, target_col=y_col)


main()
