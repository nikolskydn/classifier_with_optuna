import pandas as pd
from dateutil.parser import parse
from typing import List


def parse_datetime(dt_string):
    if isinstance(dt_string, str):
        return parse(dt_string)
    return None


def read_data(
    csv_name: str,
    timestamp_col: str,
    target_col: str,
    groupby_col: str,
    group_keys_list: List
) -> pd.DataFrame:
    df = pd.read_csv(csv_name, sep=';')
    df[timestamp_col] = df[timestamp_col].apply(parse_datetime)
    df.dropna(subset=[target_col], inplace=True)
    df[groupby_col] = df[group_keys_list[0]].astype(str)
    for key in group_keys_list[1:]:
        df[groupby_col] += '_' + df[key].astype(str)

    df.sort_values([groupby_col, timestamp_col], inplace=True)

    return df


if __name__ == '__main__':
    import os
    from utils.load_env import load_section_vars


    config_file = 'config.env'

    if os.path.exists(config_file):
        load_section_vars(config_file, 'common')
        load_section_vars(config_file, 'train')
    else:
        raise Exception(f'File {config_file} not found.')

    data_dir = os.environ['DATA_DIR']
    train_csv = os.environ['TRAIN_CSV']
    target_col = os.getenv('TARGET_COL')
    timestamp_col = os.getenv('TIMESTAMP_COL')
    groupby_col = os.getenv('GROUPBY_COL')
    group_keys_list = os.getenv('GROUP_KEYS_LIST').split(',')
    features_list = os.getenv('FEATURES_LIST').split(',')

    df = read_data(
        csv_name=os.path.join(data_dir, train_csv),
        timestamp_col=timestamp_col,
        target_col=target_col,
        groupby_col=groupby_col,
        group_keys_list=group_keys_list)

    print(df.head(3).to_string())
