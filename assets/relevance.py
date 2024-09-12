import dspy

lm = dspy.AzureOpenAI(
    api_base='https://raid-ses-openai.openai.azure.com/openai/deployments/gpt4o/chat/completions?api-version=2024-02-15-preview/',
    api_version='2024-02-15-preview',
    model='azure/gpt4o',
    api_key='0e576c031e5446e19cc6d866ad3d9f3f'
)

PROMPT = """
You are 
"""