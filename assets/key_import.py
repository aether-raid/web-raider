# assets/key_import.py

import json

# get the keys out
keys = json.load(open('assets/keys.json', 'r'))

AZURE_ENDPOINT = keys[0]['api_base']
AZURE_KEY = keys[0]['api_key']
AZURE_MODEL = keys[0]['model']
AZURE_API_VERSION = keys[0]['api_version']

GITHUB_TOKEN = keys[1]['PAT']