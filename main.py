from web_raider.shortlist import codebase_shortlist
from bs4 import BeautifulSoup
import requests
import re
from googlesearch import search
from web_raider.article import CodeArticle
from web_raider.codebase import Codebase, CodebaseType
from web_raider.url_classifier import url_classifier
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import json
import html
from openai import OpenAI

client = OpenAI(
    base_url= "http://localhost:11434/v1/",


    api_key = 'ollama'
)

def llamas(prompt):
    chat_completion = client.chat.completions.create(
        messages =[
            {
                'role': 'user',
                'content': f'rephrase the question "{prompt}" and only produce the rephrased question with nothing else',

            }
        ],
        model = 'llama2',
    )

    #rephrased_question = chat_completion['choices'][0]['message']['content']
        
    return chat_completion

def classifier(results):
    codebases = []
    seen_urls = set()

    for url in results:
        #print("analysing", url)
        if url in seen_urls:
            continue

        seen_urls.add(url)

        try:

            if url_classifier(url) == 'Codebase':
                codebases.append(url)
            
            elif url_classifier(url) == 'Article':
                pass

            elif url_classifier(url) == 'Forum':
                pass

            else:
                continue
        except:
             pass
    #print(f'Found {len(codebases)} codebases!')

    return codebases


def extract_links(text):
        """
        Extracts all the links from the HTML content.
        """
        links = ""
        try:
            # Extract all the links using regex
            links = re.findall(r'(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)', text)
            #print(links)
        except:
            pass

        return links

def get_html_content(url :str):
        """
        Fetches the HTML content of the web page from a single link.
        """
        try:
            response = requests.get(url)
            response.raise_for_status()  # Check if the request was successful
            return response.text  # Return the HTML content of the page
        except:
            print(f"Failed to fetch {url}")
            return 0
        
path = "C:\\Users\\65881\\Downloads\\questions.jsonl\\questions.jsonl"
limit = 5
lines = []
with open(path, "r") as file:
    for i in range(limit):
        lines.append(file.readline())
        
#print(lines)

class Question(BaseModel):
    Id: int
    AnswerIds: List[int]
    Repos: List[str]
    Title: str
    Body: str

ans_title_repo = {}
for line in lines:
    json_data = json.loads(line)
    question = Question(**json_data)
    ans_title_repo[question.Title] = question.Repos

title_repo = {}

for line in lines:
    json_data = json.loads(line)
    org_question = Question(**json_data)
    print("Original question: ", org_question.Title)
    rep_question = llamas(org_question.Title)
    print("Rephrased question: ", rep_question.choices[0].message.content)
    results = search(rep_question.choices[0].message.content, num_results=10)
    print(results)

    link_list = []

    for link in results:
        content = get_html_content(link)
        link_list.append(extract_links(content))

    links = []
    for link in link_list:
        links += classifier(link)
############################################
    links += classifier(results)
    title_repo[org_question.Title] = links

test = title_repo
ans = ans_title_repo

pass_rate = 0
for key in ans.keys():
    for link in test[key]:
        if link in ans[key]:
            print("passed")
            pass_rate += 1
        else:
            print()
for v in test.values():
    for l in v:
        print(v)
print(pass_rate/limit)