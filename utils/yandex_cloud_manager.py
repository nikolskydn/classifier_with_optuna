import os
import boto3
import fnmatch
import hashlib
from botocore.exceptions import ClientError
from utils.load_env import load_section_vars


class YandexCloudModelsManager:
    def __init__(self, bucket_name, access_key, secret_key, trained_models_dir):
        self.session = boto3.session.Session()
        self.s3 = self.session.client(
            service_name='s3',
            endpoint_url='https://storage.yandexcloud.net',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        self.bucket_name = bucket_name
        self.trained_models_dir = trained_models_dir
        print(trained_models_dir)

    def list_files_in_yandex_bucket(self, prefix=''):
        paginator = self.s3.get_paginator('list_objects_v2')
        response_iterator = paginator.paginate(
            Bucket=self.bucket_name, Prefix=prefix)

        for response in response_iterator:
            for content in response.get('Contents', []):
                print(f"    {content['Key']}")
    def upload_model(self, local_path, remote_path):
        try:
            self.s3.upload_file(local_path, self.bucket_name, remote_path)
            print(f'    {local_path} -> {remote_path}')
        except ClientError as e:
            print(f'Error uploading {local_path} -> {remote_path}: {e}')

    def upload_models(self, project_name):
        print(f'Upload pkl-models from {self.trained_models_dir}')
        for root, _, files in os.walk(self.trained_models_dir):
            for file in fnmatch.filter(files, '*.pkl'):
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(
                    local_path, self.trained_models_dir)
                remote_path = os.path.join(
                    project_name, relative_path).replace('\\', '/')
                self.upload_model(local_path, remote_path)

    def file_hash(self, file_path):
        try:
            with open(file_path, 'rb') as file:
                file_hash = hashlib.sha256(file.read()).hexdigest()
            return file_hash
        except FileNotFoundError:
            return None
    def remote_file_hash(self, remote_path):
        try:
            response = self.s3.get_object(
                Bucket=self.bucket_name, Key=remote_path)
            file_content = response['Body'].read()
            file_hash = hashlib.sha256(file_content).hexdigest()
            return file_hash
        except ClientError as e:
            print(f"Error getting file hash of {remote_path}: {e}")
            return None

    def download_file(self, remote_path, local_path):
        try:
            self.s3.download_file(self.bucket_name, remote_path, local_path)
            print(f'    {remote_path} -> {local_path}')
        except ClientError as e:
            print(f'Error downloading {remote_path} -> {local_path}: {e}')

    def download_models_from_directory(self, remote_directory, local_directory):
        print(f'Download pkl-models from {remote_directory}')
        paginator = self.s3.get_paginator('list_objects_v2')
        response_iterator = paginator.paginate(
            Bucket=self.bucket_name, Prefix=remote_directory)

        for response in response_iterator:
            for content in response.get('Contents', []):
                object_key = content['Key']
                if object_key.endswith('.pkl'):
                    local_path = os.path.join(
                        local_directory,
                        os.path.relpath(object_key, remote_directory))
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)

                    remote_file_hash = self.remote_file_hash(object_key)
                    local_file_hash = self.file_hash(local_path)

                    if local_file_hash != remote_file_hash:
                        self.download_file(object_key, local_path)
                    else:
                        print(f'    {local_path} exist')

    def delete_directory(self, directory_name):
        print(f'Delete {directory_name}')
        paginator = self.s3.get_paginator('list_objects_v2')
        response_iterator = paginator.paginate(
            Bucket=self.bucket_name, Prefix=directory_name)

        for response in response_iterator:
            for content in response.get('Contents', []):
                object_key = content['Key']
                self.s3.delete_object(Bucket=self.bucket_name, Key=object_key)
                print(f"    deleted {object_key}")

    def delete_file(self, file_key):
        try:
            self.s3.delete_object(Bucket=self.bucket_name, Key=file_key)
            print(f"Deleted {file_key}")
        except ClientError as e:
            print(f"Error deleting {file_key}: {e}")

    def print_directory_tree(self):
        print('View: ')
        self.list_files_in_yandex_bucket(prefix="")


if __name__ == "__main__":


    config_file_name = 'config.env'

    if os.path.exists(config_file_name):
        load_section_vars(config_file_name, 'yandex')
        load_section_vars(config_file_name, 'common')
        load_section_vars(config_file_name, 'train')
    else:
        raise Exception(f"File {config_file_name} not found.")

    API_KEY = os.environ['API_KEY']
    ACCESS_KEY = os.environ['ACCESS_KEY']
    SECRET_KEY = os.environ['SECRET_KEY']
    BUCKET_NAME = os.environ['BUCKET_NAME']
    TRAINED_MODELS_DIR = os.environ['TRAINED_MODELS_DIR']
    PROJECT_NAME = os.environ['PROJECT_NAME']

    remote_directory = f'{PROJECT_NAME}'
    local_directory = "downloaded_models"

    model_manager = YandexCloudModelsManager(
        BUCKET_NAME, ACCESS_KEY, SECRET_KEY, TRAINED_MODELS_DIR)

    model_manager.delete_directory(f'{remote_directory}/')
    model_manager.print_directory_tree()

    model_manager.upload_models(remote_directory)
    model_manager.print_directory_tree()

    model_manager.download_models_from_directory(
        remote_directory, local_directory)


