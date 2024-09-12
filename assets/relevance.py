import dspy

lm = dspy.AzureOpenAI(
    api_base='https://raid-ses-openai.openai.azure.com/openai/deployments/gpt4o/chat/completions?api-version=2024-02-15-preview/',
    api_version='2024-02-15-preview',
    model='azure/gpt4o',
    api_key='***REMOVED***'
)

PROMPT = """
You are an AI assistant tasked with determining the relevance of multiple codebases to a user-provided query. 
You will be given information about each codebase, including its description, topics, README content, and other metadata. 
Your job is to analyze this information and provide a relevance score between 0 and 100 for each codebase, where 0 means completely irrelevant and 100 means highly relevant.

Rubric for Relevance Score:
- 0-20: Irrelevant or very low relevance
- 21-40: Low relevance
- 41-60: Moderate relevance
- 61-80: High relevance
- 81-100: Very high relevance

Here is the information about the codebases:
{codebases_info}

Here is the user query:
{user_query}

Please provide a relevance score for each codebase, along with a brief explanation for your score. Additionally, provide a table comparing the pros and cons of each codebase.

Example Table:
| Codebase | Pros | Cons |
|----------|------|------|
| Codebase1 | - Pro1 \n - Pro2 | - Con1 \n - Con2 |
| Codebase2 | - Pro1 \n - Pro2 | - Con1 \n - Con2 |
"""

def determine_relevance(codebases_info, user_query):
    """
    Determine the relevance of multiple codebases to a user-provided query using Azure GPT-4o.

    Parameters:
    - codebases_info (str): Information about the codebases.
    - user_query (str): The user's query.

    Returns:
    - dict: A dictionary containing the relevance scores, explanations, and a comparison table.
    """
    prompt = PROMPT.format(codebases_info=codebases_info, user_query=user_query)
    response = lm(prompt)
    response = lm(prompt)
    return response
