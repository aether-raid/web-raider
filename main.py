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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import statistics
import pickle

# - llm_rephrase(prompt): Rephrases a question using the OpenAI API.
# - llm_prompt(prompt): Sends a prompt to the OpenAI API and returns the response.
# - check_url_status(url, timeout=15): Checks if a URL is accessible.
# - filter_dead_links(urls): Filters out dead links using parallel requests.
# - classifier(results): Classifies URLs into codebases, articles, and forums.
# - extract_links(text): Extracts all the links from the HTML content.
# - get_html_content(url): Fetches the HTML content of the web page from a single link.
# - clean_text(text): Cleans and normalizes text content.
# - chunk_text(text, chunk_size=300, overlap=60): Splits text into overlapping chunks.
# - process_and_vectorize_content(classified_links): Processes and vectorizes content from articles and forums.
# - process_questions(path, limit=5): Processes questions from a JSONL file and classifies links.
# - analyze_similarity_and_extract_links(question, processed_content, top_k=25): Analyzes chunk similarity using LSA and extracts codebase links from top chunks.
# - create_candidate_list(classified_links, analysis_results): Creates and sorts a candidate list based on occurrences.
# - get_repo_content(url, max_files=5): Extracts relevant content from a repository.
# - rerank_candidates_with_llm(question, candidates, known_repos, max_candidates=5): Re-ranks candidate links using LLM based on repository content.
# - extract_code_from_repo(url): Extracts code from a repository URL.
# - extract_from_top_candidates(ranked_candidates, k=3): Extracts code from the top k ranked repositories.
# - evaluate_model_accuracy(results, known_repos): Evaluates model accuracy by comparing found repositories with known repositories.

cheatcode = " github" # to scope down the search results to stackoverflow


client = OpenAI(
    base_url="http://localhost:11434/v1/",
    api_key='ollama'
)

class Question(BaseModel):
    Id: int
    AnswerIds: List[int]
    Repos: List[str]
    Title: str
    Body: str

class QueryType(BaseModel):
    id: int
    question: str
    justification: str
    choices: List[str]
    confidence: float

# load json object from jsonl file
def check_json_file(question, file_name='results.jsonl'):
    try:
        with open(file_name, 'r', encoding="UTF-8") as f:
            for line in f:
                data = json.loads(line)
                if data["question"] == question:
                    #return QueryType(**data)
                    return QueryType(**data)
    except:
        return None
    return None

# check for existing query in jsonl file, if it is not there evaluate the query type and save object to json file.
def check_query_type(question, file_name='results.jsonl'):
    """use llm to evaluate the most suitable type of platform for answering the question, if it is a codebase, article or forum"""
    if check_json_file(question) == None:
        try:
            # prompt engineering
            query_type = llm_prompt(
                f"""Given the question: {question}, what is the most suitable type of platform to find an answer? Is it a codebase, article, or forum
                give your answer in terms of the platform that is most likely to provide the best answer to the question.
                and the confidence level of your answer.
                in the following dictionary format:                     
                """ + f"""
                {{
                    "question": "{question}",
                    "justification": "<brief-analysis-here>",
                    "choices": [<"codebase", "article", "forum">],
                    "confidence": "<XX.XX>"
                }}
                Remember to consider the nature of the question and the type of information that is likely to be found on each platform.
                1. Codebases are repositories of code that can provide direct solutions to programming problems.
                2. Articles are detailed explanations or tutorials that can provide in-depth knowledge on a topic.
                3. Forums are discussion platforms where users can ask questions and receive answers from the community.
                4. Justification should be a brief analysis of why you think the platform you chose is the most suitable.
                5. Choices should be a list of strings with the options "codebase", "article", and "forum" where the most suitable option is placed first, and the least suitable is placed last, remove the other options if they are not suitable.
                6. Confidence level should be a number between 0.00-100.00, with an accuracy of up to 2 decimal points.
                7. ONLY reply with the dictionary format and do not add any other unnecessary text or symbols.
                """)
            # save object to json object
            json_obj = json.loads(query_type.choices[0].message.content)
            
            # save object to jsonl file if object is valid and not already in the file
            with open(file_name, 'a') as f:
                json.dump(json_obj, f)
                f.write('\n')

            return json_obj["choices"]
        except Exception as e:
            print(f"LLM evaluation failed: {str(e)}")
            return None
    return check_json_file(question).choices


def llm_rephrase(prompt):
    """Rephrases a question using the OpenAI API."""
    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': f'slightly rephrase the question "{prompt}" and only produce the rephrased question with nothing else. You are alowed to use some of the original words in the question, but try not to end up with someting too close to the original question. You must not change the meaning of the question or add unnecessarey information or words as much as possible.',
            }
        ],
        model='llama3.2',
        temperature=0,
    )
    return chat_completion

def llm_prompt(prompt):
    """Rephrases a question using the OpenAI API."""
    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': f'{prompt}',
            }
        ],
        model='llama3.2',
        temperature=0,
    )
    return chat_completion


def check_url_status(url, timeout=15):
    """Checks if a URL is accessible."""
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return 200 <= response.status_code < 400
    except:
        try:
            # Some servers block HEAD requests, try GET as fallback
            response = requests.get(url, timeout=timeout, stream=True)
            return 200 <= response.status_code < 400
        except:
            return False

def filter_dead_links(urls):
    """Filters out dead links using synchronous requests."""
    return [url for url in urls if check_url_status(url)]

def classifier(results):
    """Classifies URLs into codebases, articles, and forums."""
    codebases = []
    articles = []
    forums = []
    #seen_urls = set()

    # First filter out dead links
    live_urls = filter_dead_links(results)
    
    for url in live_urls:

        try:
            url_type = url_classifier(url)
            
            if url_type == 'Codebase':
                codebases.append(url)
            elif url_type == 'Article':
                articles.append(url)
            elif url_type == 'Forum':
                forums.append(url)
            else:
                continue
        except:
             pass

    return {
        'codebases': codebases,
        'articles': articles,
        'forums': forums
    }

def extract_links(text):
    """Extracts all the links from the HTML content."""
    links = []
    try:
        # Extract all the links using regex
        links = re.findall(r'(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)', text)
    except:
        pass
    return links

def get_html_content(url: str):
    """Fetches the HTML content of the web page from a single link."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        content_type = response.headers.get('Content-Type', '').lower()
        if 'xml' in content_type:
            return BeautifulSoup(response.text, 'lxml-xml', features="xml")  # Use lxml for XML content
        else:
            return BeautifulSoup(response.text, 'html.parser')  # Use html.parser for HTML content
    except:
        #print(f"Failed to fetch {url}")
        return 0

def clean_text(text: str) -> str:
    """Cleans and normalizes text content."""
    if not text:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Remove HTML tags using BeautifulSoup
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    
    # Basic text cleaning
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces
    text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines
    text = text.strip()
    
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 150) -> List[str]:
    """Splits text into overlapping chunks."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        if len(chunk) < chunk_size * 0.5 and chunks:
            chunks[-1] = chunks[-1] + chunk
            break
            
        chunks.append(chunk)
        start = end - overlap
        
    return chunks

def process_and_vectorize_content(classified_links: dict) -> dict:
    """Processes and vectorizes content from articles and forums."""
    processed_content = {
        'articles': [],
        'forums': []
    }
    
    # Process articles and forums
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for content_type in ['articles', 'forums']:
            for url in classified_links[content_type]:
                futures.append(executor.submit(fetch_and_process_content, url, content_type, processed_content))
        
        for future in futures:
            future.result()
    
    # Vectorize all chunks
    all_chunks = []
    chunk_metadata = []
    
    # Collect all chunks with their metadata
    for content_type in ['articles', 'forums']:
        for doc in processed_content[content_type]:
            for chunk_idx, chunk in enumerate(doc['chunks']):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    'url': doc['url'],
                    'type': content_type,
                    'chunk_index': chunk_idx
                })
    
    # Vectorize if we have chunks
    if all_chunks:
        print("\nVectorizing chunks...")
        vectorizer = TfidfVectorizer(max_features=5000)
        vectors = vectorizer.fit_transform(all_chunks)
        
        return {
            'content': processed_content,
            'vectors': vectors,
            'metadata': chunk_metadata,
            'vectorizer': vectorizer
        }
    
    return None

def fetch_and_process_content(url, content_type, processed_content):
    """Fetches and processes content from a URL."""
    try:
        print(f"Fetching content from: {url}")
        content = get_html_content(url)
        
        if content:
            # Clean the content
            cleaned_content = clean_text(content)
            # Chunk the content
            chunks = chunk_text(cleaned_content)
            
            processed_content[content_type].append({
                'url': url,
                'chunks': chunks,
                'original_length': len(cleaned_content)
            })
            print(f"Successfully processed {len(chunks)} chunks")
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")

def process_questions(path: str, limit: int = 5):
    """Processes questions from a JSONL file and classifies links."""
    lines = []
    with open(path, "r") as file:
        for i in range(limit):
            line = file.readline()
            if line:
                lines.append(line)

    for line in lines:
        # try:
            json_data = json.loads(line)
            org_question = Question(**json_data)
            check_query_type(org_question.Title)


def analyze_similarity_and_extract_links(question: str, processed_content: dict, top_k: int = 25):
    """Analyzes chunk similarity using LSA and extracts codebase links from top chunks."""
    if not processed_content:
        return None
        
    vectors = processed_content['vectors']
    metadata = processed_content['metadata']
    content = processed_content['content']
    vectorizer = processed_content['vectorizer']
    
    # Vectorize the question
    question_vector = vectorizer.transform([question])
    
    # Calculate cosine similarity between question and all chunks
    similarities = cosine_similarity(question_vector, vectors)[0]
    
    # Get indices of top k most similar chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    codebase_links = set()
    
    # Process top chunks
    for idx in top_indices:
        chunk_metadata = metadata[idx]
        content_type = chunk_metadata['type']
        url = chunk_metadata['url']
        chunk_idx = chunk_metadata['chunk_index']
        
        # Get the original chunk text
        doc = next(doc for doc in content[content_type] if doc['url'] == url)
        chunk_text = doc['chunks'][chunk_idx]
        
        # Extract potential codebase links from chunk
        chunk_links = extract_links(chunk_text)
        if chunk_links:
            # Classify links to find codebases
            classified = classifier(chunk_links)
            codebase_links.update(classified['codebases'])
        
        results.append({
            'url': url,
            'content_type': content_type,
            'similarity_score': similarities[idx],
            'chunk_text': chunk_text,
            'found_codebase_links': classified['codebases'] if chunk_links else []
        })
    
    return {
        'top_chunks': results,
        'all_codebase_links': list(codebase_links)
    }

def create_candidate_list(classified_links: dict, analysis_results: dict) -> dict:
    """Creates and sorts a candidate list based on occurrences."""
    # Combine all codebase links
    all_links = set(classified_links['codebases'])
    all_links.update(analysis_results['all_codebase_links'])
    
    # Count occurrences in chunks
    link_counts = Counter()
    
    # Count in original classified links
    for link in classified_links['codebases']:
        link_counts[link] += 1
    
    # Count in top chunks
    for chunk in analysis_results['top_chunks']:
        for link in chunk['found_codebase_links']:
            link_counts[link] += 1        
    
    unique_links = set()
    sorted_candidates = []

    for link, count in link_counts.most_common():
        if link not in unique_links:
            unique_links.add(link)
            sorted_candidates.append({
                'url': link,
                'occurrences': count
            })

    return sorted_candidates

def get_repo_content(url: str, max_files: int = 5) -> str:
    """Extracts relevant content from a repository."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        content_summary = []
        
        # Get repository description
        description = soup.find('p', {'class': 'f4 my-3'})
        if description:
            content_summary.append(f"Description: {description.get_text().strip()}")

        # Get README content
        readme = soup.find('article', {'class': 'markdown-body'})
        if readme:
            content_summary.append(f"README: {readme.get_text()[:1000]}")  # Limit README length

        # Get code files
        code_elements = soup.find_all(['pre', 'div'], class_=['highlight', 'blob-code'])
        files_added = 0
        for elem in code_elements:
            if files_added >= max_files:
                break
            code = elem.get_text().strip()
            if len(code) > 50:  # Skip very small snippets
                content_summary.append(f"Code Sample {files_added + 1}:\n{code[:500]}")
                files_added += 1

        return "\n\n".join(content_summary)
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return ""
    
def evaluate_candidates_with_llm(question: str, candidates: List[dict], known_repos: List[str], max_candidates: int = 5) -> dict:
    """
    Evaluates candidate links using LLM based on repository content.
    
    Arguments:
    - question: The original question (str)
    - candidates: List of candidate repository links with occurrence counts (list of dict)
    - known_repos: List of known correct repositories (list of str)
    - max_candidates: Maximum number of candidates to analyze (int)
    
    Returns:
    - Dictionary with the best candidate link and its accuracy score (dict)
    """
    if not candidates:
        return {'best_candidate': None, 'accuracy': 0}, known_repo_content

    # Get content for top candidates
    candidates_with_content = []
    for candidate in candidates[:max_candidates]:
        content = get_repo_content(candidate['url'])
        if content:
            candidates_with_content.append({
                'url': candidate['url'],
                'content': content,
                'occurrences': candidate['occurrences']
            })

    if not candidates_with_content:
        return {'best_candidate': None, 'accuracy': 0}, known_repo_content

    # Prepare evaluation prompt
    known_repo_content = get_repo_content(known_repos[0]) if known_repos else ""

    try:
        # Continue evaluating other candidates if accuracy is low
        for candidate in candidates_with_content:
            evaluation_prompt = f"""
            Question: {question}

            Evaluate the following GitHub repository content to determine if it answers the question.
            Use the known repository content as a reference model answer. Rate the candidate repository from 0-100 based on how well it answers the question. 
            IMPORTANT: You must ONLY return a numeric score.
            RULES:
                1. score MUST be a number (e.g. 75.50, 32.40, etc.)
                2. DO NOT use text like "The rate is" or "out of 100" only the number and nothing else.
                Known Repository Content:
                    {known_repo_content}
    
                Candidate Repository Content:
                    {candidates_with_content[0]['content']}
                """

            evaluation = llm_prompt(evaluation_prompt)
            result = evaluation.choices[0].message.content.strip()
            accuracy = float(result)

            if accuracy >= 70:
                print('best_candidate: ', candidate['url'], '\naccuracy: ', accuracy )
                return {'best_candidate': candidate['url'], 'accuracy': accuracy}, known_repo_content

        return {'best_candidate': candidates_with_content[0]['url'], 'accuracy': accuracy}, known_repo_content

    except Exception as e:
        print(f"LLM evaluation failed: {str(e)}")
        return {'best_candidate': None, 'accuracy': 0}, known_repo_content

def extract_code_from_repo(url: str) -> dict:
    """Extracts code from a repository URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        code_blocks = []
        
        # Find code elements
        if 'github.com' in url.lower():
            # GitHub specific extraction
            code_elements = soup.find_all(['pre', 'div'], class_=['highlight', 'highlight-source', 'blob-code', 'js-file-line'])
        else:
            # Generic code extraction
            code_elements = soup.find_all(['pre', 'code', 'div'], class_=['code', 'snippet', 'source'])
            
        for element in code_elements:
            code = element.get_text().strip()
            if len(code) > 50:  # Filter out small snippets
                code_blocks.append(code)
        
        return {
            'url': url,
            'code_blocks': code_blocks,
            'success': True
        }
    except Exception as e:
        print(f"Error extracting code from {url}: {e}")
        return {
            'url': url,
            'code_blocks': [],
            'success': False,
            'error': str(e)
        }

def extract_from_top_candidates(ranked_candidates: List[dict], k: int = 3) -> List[dict]:
    """Extracts code from the top k ranked repositories."""
    results = []
    
    for candidate in ranked_candidates[:k]:
        url = candidate['url']
        print(f"\nExtracting code from: {url}")
        
        extraction_result = extract_code_from_repo(url)
        if extraction_result['success'] and extraction_result['code_blocks']:
            results.append({
                'url': url,
                'rank_score': candidate.get('combined_score', candidate.get('occurrences', 0)),
                'llm_reasoning': candidate.get('reasoning', 'N/A'),
                'code_blocks': extraction_result['code_blocks']
            })
    
    return results

def evaluate_model_accuracy(results: dict, known_repos: dict) -> dict:
    """Evaluates model accuracy by comparing found repositories with known repositories."""
    total_matches = 0
    total_repos = 0
    question_metrics = {}
    total_accuracy = 0
    question_count = 0
    
    for title, data in results.items():
        # Get all found repos
        found_repos = set()
        if 'classified_links' in data:
            # Add repos from classified links
            for repo in data['classified_links']['codebases']:
                found_repos.add(repo)

        # Get known repos
        known = set(known_repos[title])

        # Find matches
        matches = found_repos.intersection(known)
        
        # Update counts
        matches_count = len(matches)
        known_count = len(known_repos[title])  # Use original count
        total_matches += matches_count
        total_repos += known_count
        
        # Calculate accuracy
        accuracy = matches_count / known_count if known_count > 0 else 0
        total_accuracy += accuracy
        question_count += 1
        
        # Store metrics
        question_metrics[title] = {
            'accuracy': accuracy,
            'matches_found': list(matches),
            'total_found': len(found_repos),
            'total_known': known_count,
            'all_found_repos': list(found_repos)
        }
        
        # Print detailed results for debugging
        print(f"\nQuestion: {title}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Found {matches_count} out of {known_count} known repositories")
        if matches:
            print("Matched repositories:")
            for repo in matches:
                print(f"- {repo}")

    # Calculate overall accuracy as the average accuracy of each question
    overall_accuracy = total_accuracy / question_count if question_count > 0 else 0
    
    return {
        'overall_accuracy': overall_accuracy,
        'total_matches': total_matches,
        'total_repos': total_repos,
        'question_metrics': question_metrics
    }

if __name__ == "__main__":
    print("Running main function")
    """
    Main function to process questions and evaluate results.
    - Reads questions from a JSONL file.
    - Processes each question to classify and vectorize links.
    - Analyzes content similarity and extracts codebase links.
    - Creates and re-ranks candidate lists.
    - Extracts code from top repositories.
    - Evaluates model accuracy.
    """
    path = "../web-raider/questions.jsonl"
    results, known_repos = process_questions(path, limit=1000)
    accuracy_list = []



        