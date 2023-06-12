import os
import joblib
import pandas as pd
from utils.get_models_list import get_models_list
from utils.load_env import load_section_vars

models_list = get_models_list()

config_file = 'config.env'

if os.path.exists(config_file):
    load_section_vars(config_file, 'common')
    load_section_vars(config_file, 'train')
else:
    raise Exception(f"File {config_file} not found.")

DATA_DIR = os.environ['DATA_DIR']
TEST_CSV = os.environ['TEST_CSV']
TRAINED_MODELS_DIR = os.environ['TRAINED_MODELS_DIR']

csv_file_name = os.path.join(DATA_DIR, TEST_CSV)

importance_df = pd.DataFrame(columns=['model', 'feature', 'importance'])
for model in models_list:
    optimized_pipeline = joblib.load(
        f'{TRAINED_MODELS_DIR}/best_pipeline_{model}.pkl')

    if model == 'lightgbm':
        feature_names = optimized_pipeline.named_steps['classifier'] \
            .feature_name_
        feature_importances = optimized_pipeline.named_steps['classifier'] \
            .feature_importances_

    if model == 'catboost':
        feature_names = optimized_pipeline.named_steps['classifier'] \
            .feature_names_
        feature_importances = optimized_pipeline.named_steps['classifier'] \
            .get_feature_importance()

    if model == 'xgboost':
        feature_names = optimized_pipeline.named_steps['classifier'] \
            .feature_names_in_
        feature_importances = optimized_pipeline.named_steps['classifier'] \
            .feature_importances_

    temp_df = pd.DataFrame(
        {'model': model,
         'feature': feature_names,
         'importance': feature_importances})

    temp_df = temp_df.sort_values(by='importance', ascending=False)
    importance_df = pd.concat([importance_df , temp_df], ignore_index=True)

# print(importance_df.to_string())

percent_to_keep = 0.8
thresholds = {}

for model in models_list:
    model_df = importance_df[importance_df['model'] == model]
    sorted_df = model_df.sort_values(by='importance', ascending=False)
    num_features = len(sorted_df)
    threshold_index = int(num_features * percent_to_keep)
    threshold = sorted_df.iloc[threshold_index]['importance']
    thresholds[model] = threshold

filtered_dfs = []
for model in models_list:
    threshold = thresholds[model]
    filtered_df = importance_df[
        (importance_df['model'] == model) &
        (importance_df['importance'] > threshold)]
    filtered_dfs.append(filtered_df)
final_df = pd.concat(filtered_dfs)

print(final_df.to_string())
print(final_df['feature'].unique().tolist())
print(final_df['feature'].unique().size)
