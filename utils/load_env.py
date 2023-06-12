import configparser
import dotenv
import os


def load_section_vars(config_file, section):
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(config_file)

    if section not in config.sections():
        raise Exception(f"Section {section} not found in {config_file}")

    section_config = config[section]

    for key, value in section_config.items():
        os.environ[key] = value


if __name__ == '__main__':

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = '../config.env'
    full_config_file = os.path.join(script_dir, config_file)

    if os.path.exists(full_config_file):
        load_section_vars(full_config_file, 'common')
    else:
        raise Exception(f"File {full_config_file} not found.")

    DATA_DIR = os.environ['DATA_DIR']
    print(DATA_DIR)

