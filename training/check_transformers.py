import os
import cloudpickle
import pandas as pd
from sklearn.pipeline import Pipeline
from utils.load_env import load_section_vars


data = {
    'time': pd.to_datetime([
        '2023-01-01 12:12:12', '2023-02-02 12:12:12', '2023-03-03 11:11:11',
        '2023-04-04 11:11',    '2023-04-05 11',       '2023-04-06 22:22:22',
        '2023-04-07',          '2023-05-08',          '2023-06-09',
        '2023-07-10']),
    'user_id': ['Vasya', 'Vasya', 'Vasya', 'Vasya', 'Vasya', 'Masha', 'Masha', 'Masha', 'Masha', 'Masha'],
    'payment': [200, 300, 200, 400, 300, 200, 300, 200, 200, 300],
    'promo_1': [20, 30, 20, 40, 30, 20, 0, 0, 0, 0],
    'promo_2': [0, 0, 0, 0, 0, 0, 0, 20, 20, 30]
}

df = pd.DataFrame(data)

config_file_name = 'config.env'

if os.path.exists(config_file_name):
    load_section_vars(config_file_name, 'yandex')
    load_section_vars(config_file_name, 'common')
    load_section_vars(config_file_name, 'train')
else:
    raise Exception(f"File {config_file_name} not found.")

path_to_transformers  = os.environ['TRAINED_MODELS_DIR']
all_files = os.listdir(path_to_transformers)

transformer_files = sorted(
    [file for file in all_files
     if file.startswith('transformer_') and file.endswith('.pkl')],
    key=lambda x: int(x.split('_')[1].split('.')[0])
)

print(transformer_files)

transformers_list = []
for file in transformer_files:
    with open(os.path.join(path_to_transformers, file), 'rb') as f:
        transformer_number = file.split('_')[1].split('.')[0]
        transformer = cloudpickle.load(f)
        transformers_list.append((f'transformer_{transformer_number}',
                                  transformer))

pipeline = Pipeline(transformers_list)

df['original_user_id'] = df['user_id']
df['original_time'] = df['time']
df_transformed = pipeline.fit_transform(df)


print(df_transformed.to_string())
