import dspy

lm = dspy.AzureOpenAI(
    api_base='https://raid-ses-openai.openai.azure.com/openai/deployments/gpt4o/chat/completions?api-version=2024-02-15-preview/',
    api_version='2024-02-15-preview',
    model='azure/gpt4o',
    api_key='***REMOVED***'
)

PROMPT = """
You are 
"""