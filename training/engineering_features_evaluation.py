import os
import logging
import cloudpickle
from typing import List
from utils.load_env import load_section_vars
from training.run_training import run_training
from utils.get_models_list import get_models_list
from utils.slack_notificator import SlakNotificator


__logger = logging.getLogger(__name__)

def sequential_engineering_features_evaluation(
    models_list: List,
    trained_models_dir: str,
    n_trials: int,
    project_name: str,
    optuna_storage_url: str,
    train_csv_file_name: str,
    timestamp_col: str,
    target_col: str,
    groupby_col: str,
    group_keys_list: List,
    features_list: List
) -> None:

    all_files = os.listdir(trained_models_dir)

    transformer_files = sorted(
        [file for file in all_files if
         file.startswith('transformer_') and file.endswith('.pkl')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )
    __logger.info(f'{transformer_files=}')

    last_transformer_file = transformer_files[-1]
    transformer_files = transformer_files[:-1]
    filename = os.path.join(trained_models_dir, last_transformer_file)
    with open(filename, 'rb') as f:
        last_transformer_number = \
            last_transformer_file.split('_')[1].split('.')[0]
        last_transformer = cloudpickle.load(f)

    best_scores = {}
    basic_transformers_list = [
        (str(last_transformer_number), last_transformer)]
    report = []

    for model in models_list:
        res = run_training(
            model=model,
            train_csv_name=train_csv_file_name,
            n_trials=n_trials,
            study_name=f'{project_name}_basic_features',
            trained_transformers_list=basic_transformers_list,
            optuna_storage_url=optuna_storage_url,
            timestamp_col=timestamp_col,
            target_col=target_col,
            groupby_col=groupby_col,
            group_keys_list=group_keys_list,
            features_list=features_list,
            trained_models_dir=trained_models_dir
        )

        score = res.value

        best_scores[model] = score

    try:
        notificator = SlakNotificator()
        notificator.send_message(
            text=f'Only last_transformer',
            username='TrainingBot',
            nested_text=f'{best_scores=}\nbasic_transformers_list: '
                        f'{[t[0] for t in basic_transformers_list]}',
            icon_emoji=":rocket:")
    except Exception as exc:
        __logger.warning(exc)

    for file in transformer_files:
        filename = os.path.join(trained_models_dir, file)
        __logger.info(f'study with {filename} start')
        with open(filename, 'rb') as f:
            transformer_number = file.split('_')[1].split('.')[0]
            transformer = cloudpickle.load(f)

            models_score = {}
            for model in models_list:
                transformer_tuple = (str(transformer_number), transformer)

                res = run_training(
                    model=model,
                    train_csv_name=train_csv_file_name,
                    n_trials=n_trials,
                    study_name=f'{project_name}_{transformer_number}',
                    trained_transformers_list=[transformer_tuple] +
                                              basic_transformers_list,
                    optuna_storage_url=optuna_storage_url,
                    timestamp_col=timestamp_col,
                    target_col=target_col,
                    groupby_col=groupby_col,
                    group_keys_list=group_keys_list,
                    features_list=features_list,
                    trained_models_dir=trained_models_dir
                )

                score = res.value

                models_score[model] = score

            try:
                notificator = SlakNotificator()
                notificator.send_message(
                    text=f'*{transformer_number=}*',
                    username="TrainingBot",
                    nested_text=f'{models_score=}',
                    icon_emoji=":rocket:")
            except Exception as exc:
                __logger.warning(f'send_message_error: {exc}')

    final_message = f':zap: training completed :zap:'
    __logger.info(final_message)

    try:
        notificator = SlakNotificator()

        notificator.send_message(
            text=final_message,
            username='TrainingBot',
            nested_text=f'optuna-dashboard {optuna_storage_url}',
        icon_emoji=':rocket:')
    except Exception as exc:
        __logger.warning(exc)


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

    sequential_engineering_features_evaluation(
        models_list=models_list,
        trained_models_dir=trained_models_dir,
        n_trials=n_trials,
        project_name=project_name,
        train_csv_file_name=train_csv_file_name,
        optuna_storage_url=optuna_storage_url,
        timestamp_col=timestamp_col,
        target_col=target_col,
        groupby_col = groupby_col,
        group_keys_list=group_keys_list,
        features_list = features_list)


