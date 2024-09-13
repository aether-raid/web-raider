# assets/relevance.py

import json
from textwrap import dedent
from openai import AzureOpenAI
from .prompts import Prompts

# get the keys out
keys = json.load(open('assets/keys.json', 'r'))

endpoint = keys['api_base']
key=keys['api_key']
model_name=keys['model']

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_version=keys['api_version'],
    api_key=key
)

def call_relevance(codebases, query):
    counter = 1
    codebases_info = ''

    for url, content in codebases.items():
        codebases['index'] = counter - 1
        codebases_info += f'Codebase {counter}\n\n'
        codebases_info += f'URL: {url}\n\n'
        codebases_info += f'Topics:\n{content["topics"]}\n\n'
        codebases_info += f'README:\n{content["readme"]}\n\n'
        # codebases_info += f'Description:\n{content["description"]}\n\n'

        counter += 1

    # response_format = json.dumps({
    #     "json_schema": {
    #         'name': 'codebase_relevance',
    #         'description': 'to determine if a codebase is relevant to a user provided query',
    #         "schema": {
    #             "codebase_names": {
    #                 "type": "array",
    #                 "items": {"type": "string"}
    #             },
    #             "relevance": {
    #                 "type": "array",
    #                 "items": {"type": "boolean"}
    #             }
    #         },
    #         'strict': True
    #     },
    #     "type": "json_schema"
    # })

    relevance = client.chat.completions.create(
        model=model_name,
        messages = [
            {
                'role': 'system',
                'content': dedent(Prompts.RELEVANCE_PROMPT)
            },
            {
                'role': 'user',
                'content': f'Codebase Information: {codebases_info}\n\nUser Query: {query}'
            },
        ],
        temperature=0,
        # response_format=response_format
    )

    response = relevance.choices[0].message.content
    return response
