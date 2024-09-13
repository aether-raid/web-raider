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

def consolidate_codebases_info(codebases):
    """Helper function to help consolidateinformation about codebases"""
    codebases_info = ''

    for i in range(len(codebases)):
        codebases_info += f'Codebase {i + 1}\n\n'
        codebases_info += f'URL: {codebases[i]["url"]}\n\n'
        codebases_info += f'Topics:\n{codebases[i]["info"]["topics"]}\n\n'
        codebases_info += f'README:\n{codebases[i]["info"]["readme"]}\n\n'
        # codebases_info += f'Description:\n{content["description"]}\n\n'

    return codebases_info

def call_relevance(codebases, query):
    codebases_info = consolidate_codebases_info(codebases)

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

def call_pro_con(codebases, query):
    codebases_info = consolidate_codebases_info(codebases)

    pro_con = client.chat.completions.create(
        model=model_name,
        messages = [
            {
                'role': 'system',
                'content': dedent(Prompts.PRO_CON_PROMPT)
            },
            {
                'role': 'user',
                'content': f'Codebase Information: {codebases_info}\n\nUser Query: {query}'
            },
        ],
        temperature=0,
        # response_format=response_format
    )

    response = pro_con.choices[0].message.content
    return response

def call_scorer(codebases, query, pro_con):
    codebases_info = consolidate_codebases_info(codebases)

    scorer = client.chat.completions.create(
        model=model_name,
        messages = [
            {
                'role': 'system',
                'content': dedent(Prompts.SCORER_PROMPT)
            },
            {
                'role': 'user',
                'content': f'Codebase Information: {codebases_info}\n\nUser Query: {query}\n\nPro/Con Information: {pro_con}'
            },
        ],
        temperature=0,
    )

    response = scorer.choices[0].message.content
    return response