import dspy

lm = dspy.AzureOpenAI(
    api_base='https://raid-ses-openai.openai.azure.com/openai/deployments/gpt4o/chat/completions?api-version=2024-02-15-preview/',
    api_version='2024-02-15-preview',
    model='azure/gpt4o',
    api_key='0e576c031e5446e19cc6d866ad3d9f3f'
)

PROMPT = """
You are an AI assistant tasked with determining the relevance of a codebase to a user-provided query. 
You will be given information about the codebase, including its description, topics, README content, and other metadata. 
Your job is to analyze this information and provide a relevance score between 0 and 100, where 0 means completely irrelevant and 100 means highly relevant.

Here is the information about the codebase:
{codebase_info}

Here is the user query:
{user_query}

Please provide a relevance score and a brief explanation for your score.
"""
"""
