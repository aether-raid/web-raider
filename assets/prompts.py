# assets/prompts.py

from enum import Enum

class Prompts(str, Enum):
    """
    A class to map prompt types to their actual prompts.

    Members
    -------
    QUERY_PROMPT: str
        A prompt to generate relevant prompts from a given user-provided query.
    RELEVANCE_PROMPT: str
        A prompt to determine the relevance of codebases to a user-provided query.
    SCORER_PROMPT: str
        A prompt to score the relevance of codebases to a user-provided query.
    PRO_CON_PROMPT: str
        A prompt to list the pros and cons of codebases in the context of a user-provided query.
    A class to map prompt types to their actual prompts.
    """

    QUERY_PROMPT = """
    You are a PhD researcher. Given a user query, first decide whether the query is too complex for a Google Search to yield results that can lead to codebases.
    
    If the query is simple enough, you can just return the query directly. 
    
    Else if the query is too complex, break the query down into smaller sub-queries. You may come up with up to 3 relevant prompts that can result in relevant codebases through a Google search.

    Do NOT output anything other than the json object.

    Output Format
    -------------

    {
        "prompts": [
            "<First Prompt>",
            "<Second Prompt>",
            "<Third Prompt>"
        ]
    }
    """

    RELEVANCE_PROMPT = """
    You are an AI assistant tasked with determining the relevance of a codebase to a user-provided query.

    You will be given information about each codebase, including its description, topics, README content, and license.
    Your job is to analyze this information and state whether the codebase is relevant or not.

    Output JUST ONE boolean value (True or False). If the license is not an open source license or there is not license information, output False.
    
    Do NOT elaborate on your decision.
    """

    SCORER_PROMPT = """
    You are an AI assistant tasked with determining the relevance of codebases to a user-provided query.
    You will be given information about each codebase, including its description, topics, README content, and other metadata.
    Your job is to analyze this information and provide a relevance score between 0 and 100 for each codebase, where 0 means completely irrelevant and 100 mean
    highly relevant.

    Rubric for Relevance Score:
    - 0-20: Irrelevant or very low relevance
    - 21-40: Low relevance
    - 41-60: Moderate relevance
    - 61-80: High relevance
    - 81-100: Very high relevance

    Please provide a relevance score for each codebase, along with a brief explanation for your score. Additionally, provide a table comparing the pros and con
    of each codebase.

    Output Format:
    --------------

    Codebase1 (URL: XXXX):

    Relevance Score: <Insert Score>
    Explanation: <Insert Explanation>

    Codebase2 (URL: XXXX):

    Relevance Score: <Insert Score>
    Explanation: <Insert Explanation>

    ... 

    CodebaseN (URL: XXXX):

    Relevance Score: <Insert Score>
    Explanation: <Insert Explanation>
    """

    PRO_CON_PROMPT = """
    You are an AI assistant tasked with determining the relevance of codebases to a user-provided query.
    You will be given information about each codebase, including its description, topics, README content, and other metadata.
    Your job is to provide a table listing the pros and cons of each codebase in the context of the user-provided query.

    Output Format
    -------------

    {
        "codebases": [
            {
                "name": "CodebaseN",
                "pros": [
                    "xxx",
                    "xxx",
                    ...
                    "xxx"
                ],
                "cons": [
                    "xxx",
                    "xxx",
                    ...
                    "xxx"
                ]
            },
            ...
        ]
    }
    """
