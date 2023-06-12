import re
import os
import optuna
from scipy.stats import ttest_ind
from utils.load_env import load_section_vars
from utils.get_models_list import get_models_list


def __ttest_learning_curves(
        optuna_storage_url:str, project_name: str, model: str) -> None:
    storage = optuna.storages.RDBStorage(url=optuna_storage_url)
    study_summaries = storage.get_all_studies()

    basic_study_name = [
        s.study_name for s in study_summaries if s.study_name.startswith(f'{project_name}_basic_features') and s.study_name.endswith(model)
    ][0]
    basic_study = optuna.study.load_study(study_name=basic_study_name, storage=storage)
    basic_best_trial_value = basic_study.best_trial.values[0]
    print(f'\n{basic_study_name=} {basic_best_trial_value}')
    basic_values = []
    for trial in basic_study.trials:
        basic_values.append(trial.value)

    print("""
    H0: There are no statistically significant differences.
    H1: There is a statistically significant difference.
    """)


    pattern = re.compile(f'{project_name}_(\d+)-{model}')
    matching_studies = [s.study_name for s in study_summaries if re.match(pattern, s.study_name)]

    for study_name in matching_studies:
        current_study = optuna.study.load_study(study_name=study_name, storage=storage)
        current_best_trial_value = current_study.best_trial.values[0]
        single = '-'
        if current_best_trial_value > basic_best_trial_value:
            single = '+'
        current_values = []
        for trial in current_study.trials:
            current_values.append(trial.value)
        t_statistic, p_value = ttest_ind(basic_values, current_values)

        hypothesis = 'H1' if p_value < 0.05 else 'H0'
        print(f'{study_name} {current_best_trial_value} {single} {hypothesis}')


if __name__ == '__main__':
    models_list = get_models_list()

    config_file = 'config.env'

    if os.path.exists(config_file):
        load_section_vars(config_file, 'common')
        load_section_vars(config_file, 'train')
    else:
        raise Exception(f'File {config_file} not found.')

    optuna_storage_url = os.environ['OPTUNA_STORAGE_URL']
    project_name = os.environ['PROJECT_NAME']

    for model in models_list:
        __ttest_learning_curves(
            optuna_storage_url=optuna_storage_url,
            project_name=project_name,
            model=model)
