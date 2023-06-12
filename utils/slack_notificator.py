import os
import json
import socket
import inspect
import requests
from utils.load_env import load_section_vars

class SlakNotificator:
    def __init__(self, config_file='config.env'):
        if os.path.exists(config_file):
            load_section_vars(config_file, 'common')
            load_section_vars(config_file, 'slack')
        else:
            raise Exception(f"File {config_file} not found.")

        PROJECT_NAME = os.environ['PROJECT_NAME']
        WEB_HOOK_URL = os.environ['WEB_HOOK_URL']
        CHANNEL = os.environ['CHANNEL']
        SLACK_API_TOKEN = os.environ['SLACK_API_TOKEN']

        self.webhook_url = WEB_HOOK_URL
        self.channel = CHANNEL
        self.slack_api_tocken = SLACK_API_TOKEN
        self.project_name = PROJECT_NAME

    def _get_caller_script_name(self):
        current_frame = inspect.currentframe()
        caller_frame = inspect.getouterframes(current_frame, 2)[2][0]
        caller_script_name = os.path.basename(caller_frame.f_globals['__file__'])
        return caller_script_name

    def send_message(self, text, nested_text=None, color='good',
                     username="ML Trainer", icon_emoji=':scikit_learn:'):
        log_prefix = f'[{self.project_name}][{self._get_caller_script_name()}]' \
                     f'[{socket.gethostname()}]'
        payload = {
            "channel": self.channel,
            "username": username,
            "text": f'{log_prefix} {text}'
        }

        if icon_emoji:
            payload["icon_emoji"] = icon_emoji

        if nested_text:
            payload["attachments"] = [
                {
                    "color": color,
                    "title": "Details",
                    "text": f"```{nested_text}```",
                    "mrkdwn_in": ["text"]
                }
            ]

        headers = {"Content-Type": "application/json"}
        response = requests.post(
            self.webhook_url, data=json.dumps(payload), headers=headers)

        if response.status_code != 200:
            raise ValueError(
                f"Request to Slack API failed with status code "
                f"{response.status_code}, response: {response.text}")

    def upload_file(
            self, file_path, title=None, initial_comment=None, filetype='png'):
        """
        SLACK_API_TOKEN:
        - Go to https://api.slack.com/apps.
        - Click "Create New App".
        - Click "Create App".
        - "Add features and functionality", "OAuth & Permissions".
        - "Scopes" section, "Bot Token Scopes", "Add an OAuth Scope".
        - Add the "chat:write" permission.
        - "Install App", "Install App to Workspace".
        - In "OAuth & Permissions" section get token starts with xoxb-.
        """
        if not os.path.isfile(file_path):
            raise ValueError(f"File '{file_path}' not found")

        with open(file_path, "rb") as f:
            file_content = f.read()

        file_name = os.path.basename(file_path)

        if filetype == 'csv':
            file_mimetype = "text/csv"
        elif filetype == 'png':
            file_mimetype = "image/png"
        else:
            raise ValueError("Unsupported filetype.")

        payload = {
            "channels": self.channel,
            "filename": file_name,
            "filetype": filetype,
            "title": title or file_name
        }

        if initial_comment:
            payload["initial_comment"] = initial_comment

        files = {"file": (file_name, file_content, file_mimetype)}
        api_url = "https://slack.com/api/files.upload"
        headers = {"Authorization": f"Bearer {self.slack_api_tocken}"}

        response = requests.post(
            api_url, data=payload, files=files, headers=headers)
        response_json = response.json()

        if not response_json.get("ok"):
            raise ValueError(
                f"File upload to Slack API failed, response: {response_json}")


if __name__ == "__main__":


    notificator = SlakNotificator()

    sample_traceback = "Traceback bla bla bla"
    notificator.send_message(
        "division by zero",
        nested_text=sample_traceback,
        color='danger',
        username="TrainingBot",
        icon_emoji=":robot_face:")
