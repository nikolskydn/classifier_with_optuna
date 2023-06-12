# Configuration

Please fill in the following configuration parameters in the `config.env` file:

    [common]
    PROJECT_NAME = your_project_name
    DATA_DIR = data

    [train]
    TRAIN_CSV = train.csv
    TEST_CSV = test.csv
    TRAINED_MODELS_DIR = trained_models
    N_TRIALS = 50
    OPTUNA_STORAGE_URL = postgresql://user:passw@addres/db
    TARGET_COL = yuor_target
    TIMESTAMP_COL = times
    GROUPBY_COL = gtoup_id
    GROUP_KEYS_LIST = your_kye1,your_key2
    FEATURES_LIST = time,group_id,f1,f2,f3

    [slack]
    WEB_HOOK_URL = your_hook
    CHANNEL = your_channel
    SLACK_API_TOKEN = your_key

    [your_cloud]
    API_KEY = your_key
    ACCESS_KEY = your_key
    SECRET_KEY = your_key
    BUCKET_NAME = your_name

# Sequential Engineering Feature Selection

- Prepare yuor dataset.

- Create transformers by `build_row_feature_transformers.py`
or other script (`build_aggregated_transformers.py`).

- Launch `engineering_features_evaluation.py` for sequential feature selection.

# Model Training with Selected Features

Use `run_training.py` to train models using the final set of features. Stacking 
with `staking.py`.
