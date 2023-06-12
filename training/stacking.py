import os
import logging
# import joblib
# import dill
import cloudpickle
import pandas as pd
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import (f1_score, roc_auc_score, accuracy_score,
                             recall_score, confusion_matrix)
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from utils.read_data import read_data
from utils.load_env import load_section_vars
from utils.get_models_list import get_models_list
from utils.slack_notificator import SlakNotificator


logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s][%(asctime)s][%(filename)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
__logger = logging.getLogger(__name__)

models = get_models_list()
__logger.info(f'Models for staking: {models}')

config_file = 'config.env'

if os.path.exists(config_file):
    load_section_vars(config_file, 'common')
    load_section_vars(config_file, 'train')
else:
    raise Exception(f"File {config_file} not found.")

data_dir = os.environ['DATA_DIR']
train_csv = os.environ['TRAIN_CSV']
trained_models_dir = os.environ['TRAINED_MODELS_DIR']
n_trials = int(os.environ['N_TRIALS'])
project_name = os.environ['PROJECT_NAME']
optuna_storage_url = os.environ['OPTUNA_STORAGE_URL']
target_col = os.getenv('TARGET_COL')
timestamp_col = os.getenv('TIMESTAMP_COL')
groupby_col = os.getenv('GROUPBY_COL')
group_keys_list = os.getenv('GROUP_KEYS_LIST').split(',')
features_list = os.getenv('FEATURES_LIST').split(',')
test_csv_file_name = os.path.join(data_dir, train_csv)

all_trained_files = os.listdir(trained_models_dir)

transformer_files = sorted(
    [file for file in all_trained_files if
     file.startswith('transformer_') and file.endswith('.pkl')],
    key=lambda x: int(x.split('_')[1].split('.')[0])
)
__logger.info(f'{transformer_files=}')


df_test = read_data(
    csv_name=test_csv_file_name,
    timestamp_col=timestamp_col,
    target_col=target_col,
    groupby_col=groupby_col,
    group_keys_list=group_keys_list)

model_pipelines = {
    model: cloudpickle.load(open(
        f'{trained_models_dir}/{model}.pkl', 'rb'))
    for model in models
}
__logger.info(f'{list(model_pipelines.keys())}')

stacking_model = StackingClassifier(
    estimators=[(model, pipeline) for model, pipeline in model_pipelines.items()],
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1
)

stacking_model.fit(df_test[features_list].copy(), df_test[target_col].copy())

with open(f'{trained_models_dir}/stacking.pkl', 'wb') as file_model:
    cloudpickle.dump(stacking_model, file_model)

print(df_test.head(3).to_string())

print(df_test[features_list].head(3).to_string())

with open('trained_models/stacking.pkl', 'rb') as file_model:
    stacking_model_loaded = cloudpickle.load(file_model)


predictions = stacking_model_loaded.predict(df_test[features_list].copy())
df_test["predicted"] = predictions
accuracy = accuracy_score(df_test[target_col], df_test["predicted"])
recall = recall_score(df_test[target_col], df_test["predicted"])
f1 = f1_score(df_test[target_col], df_test["predicted"])
roc_auc = roc_auc_score(df_test[target_col], df_test["predicted"])
__logger.info(f'{accuracy=}')
__logger.info(f'{recall=}')
__logger.info(f'{f1=}')
__logger.info(f'{roc_auc=}')

cm = confusion_matrix(df_test['predicted'], df_test[target_col])
labels = unique_labels(df_test['predicted'], df_test[target_col])
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_df = cm_df.rename_axis("Pred", axis="columns")
cm_df = cm_df.rename_axis("True", axis="rows")
__logger.info("""
Confusion Matrix:
"Rows: True vals"
"Columns: Predicted vals"
{cm_df.to_string()}
""")

final_message = f':white_check_mark: stacking_model completed :white_check_mark:\n' \
                f'{models=}'
__logger.debug(final_message)

try:
    logger = SlakNotificator()
    logger.send_message(
        text=final_message,
        nested_text=f'{f1=}\n{accuracy=}\n{recall=}\n{roc_auc=}\n{cm_df.to_string()}',
        username="TrainingBot",
        icon_emoji=":steam_locomotive:")
except Exception as e:
    __logger.warning(f'slack notification failed {e}')
