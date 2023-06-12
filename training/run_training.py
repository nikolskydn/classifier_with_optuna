import os
import optuna
import logging
import argparse
import numpy as np
import cloudpickle
from typing import List
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from utils.read_data import read_data
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from utils.load_env import load_section_vars
from utils.get_models_list import get_models_list
from utils.slack_notificator import SlakNotificator
from training.create_objective import create_objective
from sklearn.utils.class_weight import compute_class_weight


__logger = logging.getLogger(__name__)

def run_training(
    model: str,
    train_csv_name: str,
    n_trials:int,
    study_name: str,
    trained_transformers_list,
    optuna_storage_url: str,
    timestamp_col: str,
    target_col: str,
    groupby_col: str,
    group_keys_list: List,
    features_list: List,
    trained_models_dir: str,
    is_drop_study=False
) -> str:

    df = read_data(
        csv_name=train_csv_name,
        timestamp_col=timestamp_col,
        target_col=target_col,
        groupby_col=groupby_col,
        group_keys_list=group_keys_list)

    groups = df[groupby_col]

    X = df[features_list]
    y = df[target_col]

    unique_classes = np.unique(y)

    class_weights = compute_class_weight(
        'balanced', classes=unique_classes, y=y)

    __logger.debug(f'{class_weights=}')


    objective_func = create_objective(
        model=model,
        transformers_list=trained_transformers_list,
        groups=groups,
        class_weights=class_weights,
        X=X,
        y=y)

    __logger.info(f'{model}')


    study_name = f'{study_name}-{model}'
    if is_drop_study:
        optuna.delete_study(study_name=study_name, storage=optuna_storage_url)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=optuna_storage_url,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=123)
    )

    study.optimize(objective_func, n_trials=n_trials)
    __logger.info(f'{study.best_trial=}')

    best_params = study.best_trial.params
    best_trial = study.best_trial

    if model == 'lightgbm':
        best_params.update({
            'n_jobs': -1,
            'random_state': 123,
        })
        optimized_classifier = LGBMClassifier(**best_params)

    if model == 'catboost':
        best_params.update({
            'verbose': False,
            'thread_count': -1,
            'random_state': 123,
        })
        optimized_classifier = CatBoostClassifier(**best_params)
    if model == 'xgboost':
        best_params.update({
            'n_jobs': -1,
            'random_state': 123,
        })
        optimized_classifier = XGBClassifier(**best_params)

    optimized_pipeline = Pipeline(
        steps=trained_transformers_list +
              [('classifier', optimized_classifier)])

    optimized_pipeline.fit(X, y)

    trained_model_name = f'{trained_models_dir}/{model}.pkl'
    with open(trained_model_name, "wb") as model_file:
        cloudpickle.dump(optimized_pipeline , model_file)

    __logger.debug(f'{trained_model_name} dumped')

    return best_trial


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(levelname)s][%(asctime)s][%(filename)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    models_list = get_models_list()

    config_file = 'config.env'

    if os.path.exists(config_file):
        load_section_vars(config_file, 'common')
        load_section_vars(config_file, 'train')
    else:
        raise Exception(f'File {config_file} not found.')

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

    train_csv_file_name = os.path.join(data_dir, train_csv)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    all_files = os.listdir(trained_models_dir)

    transformer_files = sorted(
        [file for file in all_files if
         file.startswith('transformer_') and file.endswith('.pkl')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )
    __logger.info(f'{transformer_files=}')

    transformers_list = []
    for file in transformer_files:
        with open(os.path.join(trained_models_dir, file), 'rb') as f:
            transformer_number = file.split('_')[1].split('.')[0]
            transformer = cloudpickle.load(f)
            transformers_list.append(
                (f'transformer_{transformer_number}', transformer))

    models_scores = {}

    for model in models_list:
        res = run_training(
            model=model,
            train_csv_name=train_csv_file_name,
            n_trials=n_trials,
            study_name=f'{project_name}_basic_features',
            trained_transformers_list=transformers_list,
            optuna_storage_url=optuna_storage_url,
            timestamp_col=timestamp_col,
            target_col=target_col,
            groupby_col=groupby_col,
            group_keys_list=group_keys_list,
            features_list=features_list,
            trained_models_dir=trained_models_dir
        )

        score = res.value
        models_scores[model] = score

    try:
        notificator = SlakNotificator()
        notificator.send_message(
            text=f'train',
            username='TrainingBot',
            nested_text=f'{models_scores=}',
            icon_emoji=":rocket:")
    except Exception as exc:
        __logger.warning(exc)
