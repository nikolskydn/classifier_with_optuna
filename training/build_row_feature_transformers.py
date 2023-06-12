import os
import holidays
import pandas as pd
import cloudpickle
from sklearn.base import BaseEstimator, TransformerMixin


class DaysSinceFirstRowCreator(TransformerMixin, BaseEstimator):
    def __init__(self, groupby_col, timestamp_col):
        self.groupby_col = groupby_col
        self.timestamp_col = timestamp_col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['days_since_first_row'] = (
            X[self.timestamp_col] -
            X.groupby(self.groupby_col)[self.timestamp_col].transform('min')
        ).dt.days
        return X


class DaysSinceNRowsBackCreater(TransformerMixin, BaseEstimator):
    def __init__(self, groupby_col, numbers_back_list, timestamp_col):
        self.groupby_col = groupby_col
        self.numbers_back_list = numbers_back_list
        self.timestamp_col = timestamp_col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X = X.sort_values(by=[self.groupby_col, self.timestamp_col])
        for n in self.numbers_back_list:
            X[f'days_since_{n}_rows_back'] = \
                X.groupby(self.groupby_col)[self.timestamp_col] \
                    .transform(lambda x: x.diff(n).dt.days)
        return X

class LagsCreator(BaseEstimator, TransformerMixin):

    def __init__(self, groupby_col, features_list, lags_list):
        self.groupby_col = groupby_col
        self.features_list = features_list
        self.lags_list = lags_list

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for lag in self.lags_list:
            for feature in self.features_list:
                X[f"{feature}_lag_{lag}"] = X.groupby(
                    self.groupby_col)[feature].shift(lag)
        return X



class DateTimeCreator(TransformerMixin, BaseEstimator):
    def __init__(self, timestamp_col, is_year=False, is_month=False,
                 is_day=False, is_hour=False, is_minute=False, is_second=False,
                 is_weekday=False, is_season=False):
        self.timestamp_col = timestamp_col
        self.is_year = is_year
        self.is_month = is_month
        self.is_day = is_day
        self.is_hour = is_hour
        self.is_minute = is_minute
        self.is_second = is_second
        self.is_weekday = is_weekday
        self.is_season = is_season

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X[self.timestamp_col] = pd.to_datetime(X[self.timestamp_col])

        if self.is_year:
            X['year'] = X[self.timestamp_col].dt.year
        if self.is_month:
            X['month'] = X[self.timestamp_col].dt.month
        if self.is_day:
            X['day'] = X[self.timestamp_col].dt.day
        if self.is_hour:
            X['hour'] = X[self.timestamp_col].dt.hour
        if self.is_minute:
            X['minute'] = X[self.timestamp_col].dt.minute
        if self.is_second:
            X['second'] = X[self.timestamp_col].dt.second
        if self.is_weekday:
            X['weekday'] = X[self.timestamp_col].dt.dayofweek

        if self.is_season:
            if 'month' not in X.columns:
                X['month'] = X[self.timestamp_col].dt.month
            X['season'] = (X['month'] + 1) % 12 // 3

        return X


class HolidaysCreator(BaseEstimator, TransformerMixin):
    def __init__(self, timestamp_col, country='RU'):
        self.country = country
        self.timestamp_col = timestamp_col
        self.holidays = holidays.CountryHoliday(self.country)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['is_holiday'] = X[self.timestamp_col].map(lambda x: int(x in self.holidays))
        return X

class RollingStatisticsCreator(BaseEstimator, TransformerMixin):
    def __init__(self, groupby_col, features_list, windows_list,
                 statistics_list, timestamp_col):
        self.groupby_col = groupby_col
        self.features_list = features_list
        self.windows_list = windows_list
        self.statistics_list = statistics_list
        self.timestamp_col = timestamp_col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        result = X.copy()
        grouped = X.sort_values(by=self.timestamp_col).groupby(self.groupby_col)
        for column in self.features_list:
            for window in self.windows_list:
                for stat in self.statistics_list:
                    if isinstance(stat, str):
                        stat_name = stat
                        stat_func = getattr(pd.Series, stat)
                    else:
                        stat_name = (
                            stat.__name__
                            if hasattr(stat, "__name__")
                            else stat.__class__.__name__
                        )
                        stat_func = stat

                    result[f"{column}_{stat_name}_window_{window}"] = (
                        grouped[column]
                        .rolling(window=window, min_periods=1)
                        .apply(stat_func)
                        .reset_index(level=0, drop=True)
                    )

        return result

class FieldsRemover(BaseEstimator, TransformerMixin):
    def __init__(self, columns_list) -> None:
        self.columns_list = columns_list

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=self.columns_list, axis=1)
        return X


if __name__ == '__main__':
    from utils.load_env import load_section_vars


    config_file = 'config.env'

    if os.path.exists(config_file):
        load_section_vars(config_file, 'train')
    else:
        raise Exception(f'File {config_file} not found.')

    trained_models_dir = os.environ['TRAINED_MODELS_DIR']
    timestamp_col = os.getenv('TIMESTAMP_COL')
    groupby_col = os.getenv('GROUPBY_COL')

    print('delete all pkls transformers and models')

    for file in os.listdir(trained_models_dir):
        if file.endswith('.pkl'):
            os.remove(os.path.join(trained_models_dir, file))

    print('generate new transformers')

    transformers_list = [
        DateTimeCreator(is_month=True, timestamp_col=timestamp_col),  #  0
        DateTimeCreator(is_season=True, timestamp_col=timestamp_col),  #  1
        DateTimeCreator(
            is_month=True, is_season=True, timestamp_col=timestamp_col),  #  2
        FieldsRemover(columns_list=[groupby_col, timestamp_col]), # 3
    ]

    for i in range(len(transformers_list)):
        with open(f'./{trained_models_dir}/transformer_{i}.pkl', 'wb') as f:
            cloudpickle.dump(transformers_list[i], f)
