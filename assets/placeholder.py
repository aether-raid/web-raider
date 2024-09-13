# previously relevance.py

# TO IMPLEMENT
# 1. separate model for relevance checker (true/false)
# 2. separate model for pros and cons comparison
# 3. separate model for relevance score evaluation AFTER (1) is done + can consider the input of (2)

# HAVE TRIED
# 1. dspy. not very successful.
# 2. prannaya suggested structured outputs! which was nice :) then we tried it... didn't work very well mainly cuz our gpt version too old.

# import dspy
import json
import openai

# get the bloody keys out
keys = json.load(open('assets/keys.json', 'r'))

# lm = dspy.AzureOpenAI(
#     api_base=keys['api_base'],
#     api_version=keys['api_version'],
#     model=keys['model'],
#     api_key=keys['api_key']
# )

# dspy.settings.configure(lm=lm)

PROMPT = """
You are an AI assistant tasked with determining the relevance of multiple codebases to a user-provided query.
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

Codebase1:

Relevance Score: <Insert Score>
Explanation: <Insert Explanation>

Codebase2:

Relevance Score: <Insert Score>
Explanation: <Insert Explanation>

... 

CodebaseN:

Relevance Score: <Insert Score>
Explanation: <Insert Explanation>

Table ranking pros and cons:
 | Codebase | Pros | Cons |
 |----------|------|------|
 | Codebase1 | - Pro1 \n - Pro2 | - Con1 \n - Con2 |
 | Codebase2 | - Pro1 \n - Pro2 | - Con1 \n - Con2 |
"""

INPUTS = """
Here is the information about the codebases:
{codebases_info}

Here is the user query:
{user_query}
"""

# class CodeRecommender(dspy.Signature):
#     """Evaluate relevance of codebase provided."""

#     # context = dspy.InputField(desc="role of the recommender")
#     text = dspy.InputField(desc="Information about the codebases and the original user query.")
#     score = dspy.OutputField(desc="0 to 100 depending on how relevant the codebase is to my query.", prefix='Score: ')
#     explanation = dspy.OutputField(desc='Explanation for the given score based on relevance of codebase to my query.', prefix='Explanation: ')

def determine_relevance(codebases, user_query):
    """
    Determine the relevance of multiple codebases to a user-provided query using Azure GPT-4o.

    Parameters:
    - codebases (dict): A dictionary where the key is the URL of the codebase and the value is an inner dictionary
containing the codebase object and context.
    - user_query (str): The user's query.

    Returns:
    - dict: A dictionary containing the relevance scores, explanations, and a comparison table.
    """

    # codebases is of this format now: {url: content}
    counter = 1
    codebases_info = ''
    
    for url, content in codebases.items():
        codebases_info += f'Codebase {counter}\n\n'
        codebases_info += f'URL: {url}\n\n'
        codebases_info += f'Topics:\n{content["topics"]}\n\n'
        codebases_info += f'README:\n{content["readme"]}\n\n'
        # codebases_info += f'Description:\n{content["description"]}\n\n'

        counter += 1

    # codebases_info = "\n\n".join([f"URL: {url}\nContext: {info['context']}" for url, info in codebases.items()])
    text = INPUTS.format(codebases_info=codebases_info, user_query=user_query)
    # relevance = dspy.ChainOfThought(CodeRecommender)(
    #     # context=PROMPT,
    #     text=text
    # )
    
    # return relevance.score + '\n' + relevance.explanation
