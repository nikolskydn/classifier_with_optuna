import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import optuna
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from typing import List, Dict, Callable, Union, Tuple


def create_objective(
        model: str,
        groups: pd.Series,
        transformers_list: List[Union[Tuple[str, BaseEstimator]]],
        class_weights: Dict[int, float],
        X: pd.DataFrame,
        y: pd.Series,
        folds_number: int = 5) -> Callable[[optuna.Trial], float]:

    def objective(trial: optuna.Trial) -> float:
        group_kfold = GroupKFold(n_splits=folds_number)
        scores = []

        for train_index, test_index in group_kfold.split(X, y, groups=groups):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            if model == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int(
                        'n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 4, 8),
                    'learning_rate': trial.suggest_float(
                        'learning_rate', 1e-6, 1e-1, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 31, 255),
                    'min_child_samples': trial.suggest_int(
                        'min_child_samples', 5, 100),
                    'min_child_weight': trial.suggest_float(
                        'min_child_weight', 1e-5, 1, log=True),
                    'subsample': trial.suggest_float('subsample', 0.1, 1.0),
                    'colsample_bytree': trial.suggest_float(
                        'colsample_bytree', 0.1, 1.0),
                    'reg_alpha': trial.suggest_float(
                        'reg_alpha', 1e-5, 1.0, log=True),
                    'reg_lambda': trial.suggest_float(
                        'reg_lambda', 1e-5, 1.0, log=True),
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'n_jobs': -1,
                    'random_state': 123,
                    'scale_pos_weight': class_weights[1] / class_weights[0],
                    # 'device_type': 'gpu'
                }

                classifier = LGBMClassifier(**params)

            if model == 'catboost':
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 1000),
                    'depth': trial.suggest_int('depth', 4, 8),
                    'learning_rate': trial.suggest_float(
                        'learning_rate', 1e-6, 1e-1, log=True),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10),
                    'border_count': trial.suggest_int('border_count', 32, 255),
                    'loss_function': 'Logloss',
                    'verbose': False,
                    'random_state': 123,
                    'class_weights': class_weights,
                    # 'task_type': 'GPU'
                }

                classifier = CatBoostClassifier(**params)

            if model == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int(
                        'n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 4, 8),
                    'learning_rate': trial.suggest_float(
                        'learning_rate', 1e-6, 1e-1, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
                    'min_child_weight': trial.suggest_int(
                        'min_child_weight', 1, 10),
                    'objective': 'binary:logistic',
                    'n_jobs': -1,
                    'random_state': 123,
                    'scale_pos_weight': class_weights[1] / class_weights[0],
                    # 'tree_method': 'gpu_hist'
                }

                classifier = XGBClassifier(**params)

            pipeline = Pipeline(
                steps=transformers_list + [('classifier', classifier)])

            pipeline.fit(X_train, y_train)

            # y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            # score = roc_auc_score(y_test, y_pred_proba)

            y_pred = pipeline.predict(X_test)
            # score = f1_score(y_test, y_pred, average='weighted')  # <--
            score = f1_score(y_test, y_pred, average='binary')

            scores.append(score)

        return np.mean(scores)

    return objective


if __name__ == '__main__':
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from optuna import Trial, create_study


    data = {
        'time': pd.to_datetime([
            '2023-01-01 12:12:12', '2023-01-02 13:13:13', '2023-01-03 14:14:14',
            '2023-01-04 15:15:15', '2023-01-05 16:16:16', '2023-01-06 17:17:17',
            '2023-01-07 18:18:18', '2023-01-08 19:19:19', '2023-01-09 20:20:20',
            '2023-01-10 21:21:21', '2023-01-11 22:22:22', '2023-01-12 23:23:23',
            '2023-01-13 00:00:00', '2023-01-14 01:01:01', '2023-01-15 02:02:02',
            '2023-01-16 03:03:03', '2023-01-17 04:04:04', '2023-01-18 05:05:05'
        ]),
        'user_id': [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2],
        'total': [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200,
                  1300,
                  1400, 1500, 1600, 1700, 1800, 1900],
        'target': [0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0]
    }

    df = pd.DataFrame(data)
    df = df.set_index('time')

    X = df.drop(columns='target')
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    transformers_list = [('std_scaler', StandardScaler())]

    class_weights = y_train.value_counts(normalize=True).to_dict()

    groups = X_train['user_id']

    objective = create_objective(
        'lightgbm',
        groups,
        transformers_list,
        class_weights,
        X_train,
        y_train,
        folds_number=2)

    study = create_study(direction='maximize')

    study.optimize(objective, n_trials=10)

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
