import os
import joblib
import pandas as pd
from utils.get_models_list import get_models_list
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    roc_auc_score, recall_score)
from utils.load_env import load_section_vars
from utils.read_data import read_data



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

csv_file_name = os.path.join(data_dir, test_csv)

df_test = read_data(
    csv_name=csv_file_name,
    timestamp_col=timestamp_col,
    target_col=target_col,
    groupby_col=groupby_col,
    group_keys_list=group_keys_list)


X_test = df_test[features_list].copy()
y_test = df_test[target_col].copy()

models_list = get_models_list()

for model in models_list:
    optimized_pipeline = joblib.load(
        f'{trained_models_dir}/{model}.pkl')
    y_pred = optimized_pipeline.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    print(f'{f1=}')
    print(f'{roc_auc=}')
    print(f'{accuracy=}')
    print(f'{recall=}')

    cm = confusion_matrix(y_pred, y_test)
    labels = unique_labels(y_pred, y_test)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print("Confusion Matrix:")
    print("Rows: True vals")
    print("Columns: Predicted vals")
    cm_df = cm_df.rename_axis("Pred", axis="columns")
    cm_df = cm_df.rename_axis("True", axis="rows")
    print(cm_df.to_string())

