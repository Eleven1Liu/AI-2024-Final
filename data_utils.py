import pickle


def normalize_features(data, features, min_values=None, max_values=None):
    """Min/max normalization.

    Args:
        data (pd.DataFrame): A dataframe with data.
        features (list): List of feature column names.
        min_values (float, optional): Given min values for validation. Defaults to None.
        max_values (float, optional): Given max values for validation. Defaults to None.
    """
    normalized_data = data.copy()
    min_values = min_values if min_values is not None else data[features].min()
    max_values = max_values if max_values is not None else data[features].max()
    for feature in features:
        min_v = min_values[feature]
        max_v = max_values[feature]
        normalized_data[feature] = (data[feature] - min_v) / (max_v-min_v)

    return normalized_data[features]


def sample_nonan_data(df, features):
    """Filter examples if all of the given features is not None."""
    mask = True
    for feature in features:
        mask &= ~(df[feature].isna())

    # dropped_cnt = len(df) - len(df[mask])
    # print(f'{features}: Dropped {(dropped_cnt*100/len(df)):.2f}% data.')
    return df[mask]


def dump_pickle(filename, data):
    """Dump data to file name."""
    print(f'Write predictions to {filename}.')
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
