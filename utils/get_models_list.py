import argparse

all_models_list = ['catboost', 'xgboost', 'lightgbm']

def get_models_list():
    stringified_list = "', '".join(all_models_list)
    models_help_list = f"'{stringified_list}'"
    models_help_list = models_help_list.rsplit(", ", 1)
    models_help_list = " and ".join(models_help_list)

    parser = argparse.ArgumentParser(description="Select models.")

    parser.add_argument(
        '--models',
        nargs='*',
        default=all_models_list,
        choices=all_models_list,
        help=f"Choose any combination of {models_help_list}")

    args = parser.parse_args()
    models_list = args.models

    return models_list


if __name__ == '__main__':

   models_list = get_models_list()
   print(models_list)
