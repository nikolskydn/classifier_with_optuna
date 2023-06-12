import os
import sys
import joblib
import numpy as np
import pandas as pd
from dateutil.parser import parse
from sklearn.feature_selection import RFECV
from utils.load_env import load_section_vars
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight

def parse_datetime(dt_string):
    if isinstance(dt_string, str):
        return parse(dt_string)
    return None

models = ['catboost', 'xgboost', 'lightgbm']



config_file = 'config.env'

if os.path.exists(config_file):
    load_section_vars(config_file, 'common')
    load_section_vars(config_file, 'train')
else:
    raise Exception(f"File {config_file} not found.")

data_dir = os.environ['DATA_DIR']
test_csv = os.environ['TEST_CSV']
trained_models_dir = os.environ['TRAINED_MODELS_DIR']
features_list = os.getenv('FEATURES_LIST').split(',')
target_col = os.getenv('TARGET_COL')
timestamp_col = os.getenv('TIMESTAMP_COL')
groupby_col = os.getenv('GROUPBY_COL')
group_keys_list = os.getenv('GROUP_KEYS_LIST').split(',')


df = pd.read_csv(os.path.join(data_dir, test_csv), sep=';')
df[timestamp_col] = df[timestamp_col].apply(parse_datetime)
df[groupby_col] = df[group_keys_list[0]].astype(str)
for key in group_keys_list[1:]:
    df[groupby_col] += '_' + df[key].astype(str)
df.sort_values([groupby_col, timestamp_col], inplace=True)
df.dropna(subset=[target_col], inplace=True)
groups = df['customer_id'].astype(str) + '_' + df['merchant_id'].astype(str)

X = df[features_list]
y = df[target_col]

unique_classes = np.unique(y)
class_weights = compute_class_weight('balanced', classes=unique_classes, y=y)

selected_features_all = []

detail_report = 'Details for each model:\n\n'
for model in models:
    print(f'{model}')
    pipeline = joblib.load(os.path.join(trained_models_dir, f'{model}.pkl'))
    X_transformed = pipeline[:-1].transform(X)
    all_features = X_transformed.columns.tolist()
    group_kfold = GroupKFold(n_splits=5)
    rfecv = RFECV(
        estimator=pipeline.named_steps['classifier'],
        step=1,
        cv=group_kfold,
        scoring='f1_weighted',
        verbose=0)
    rfecv.fit(X_transformed, y, groups=groups)

    selected_features = {
        X_transformed.columns[i]
        for i in range(len(rfecv.support_)) if rfecv.support_[i]}
    selected_features_all.append(selected_features)

    discarded_features_model = set(all_features) - selected_features

    detail_report += f"Selected features for *{model}*:\n" \
                     f"{', '.join(selected_features)}\n\n"
    detail_report += f"Features to discard for *{model}*:\n" \
                     f"{', '.join(discarded_features_model)}\n\n"

union_selected_features = set.union(*selected_features_all)
discarded_features = set(all_features) - union_selected_features

summary_report = f"*Selected features* for all models (union):\n" \
                 f"{', '.join(union_selected_features)}\n\n"
summary_report += f"*Features to discard* for all models:\n" \
                  f"{', '.join(discarded_features)}"

print(summary_report)
print(detail_report)




