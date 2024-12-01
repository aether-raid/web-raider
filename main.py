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
import aiohttp
import asyncio
from cachetools import cached, TTLCache

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
# - normalize_github_url(url): Normalizes GitHub URLs to a standard format.
# - get_repo_content(url, max_files=5): Extracts relevant content from a repository.
# - rerank_candidates_with_llm(question, candidates, known_repos, max_candidates=5): Re-ranks candidate links using LLM based on repository content.
# - extract_code_from_repo(url): Extracts code from a repository URL.
# - extract_from_top_candidates(ranked_candidates, k=3): Extracts code from the top k ranked repositories.
# - evaluate_model_accuracy(results, known_repos): Evaluates model accuracy by comparing found repositories with known repositories.

cheatcode = " stackoverflow" # to scope down the search results to stackoverflow


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


def llm_rephrase(prompt):
    """Rephrases a question using the OpenAI API."""
    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': f'slightly rephrase the question "{prompt}" and only produce the rephrased question with nothing else. you are alowed to use the original words in the question. You must not change the meaning of the question or add unecessaey information or words as much as possible.',
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

# Cache for URL status checks
url_status_cache = TTLCache(maxsize=1000, ttl=3600)

@cached(url_status_cache)
async def async_check_url_status(url, timeout=15):
    """Checks if a URL is accessible using asynchronous requests."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.head(url, timeout=timeout) as response:
                return 200 <= response.status < 400
        except:
            try:
                async with session.get(url, timeout=timeout) as response:
                    return 200 <= response.status < 400
            except:
                return False

async def async_filter_dead_links(urls):
    """Filters out dead links using asynchronous requests."""
    tasks = [async_check_url_status(url) for url in urls]
    url_status = await asyncio.gather(*tasks)
    return [url for url, is_live in zip(urls, url_status) if is_live]

def filter_dead_links(urls):
    """Wrapper to run async_filter_dead_links synchronously."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(async_filter_dead_links(urls))

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
        return response.text  # Return the HTML content of the page
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

    ans_title_repo = {}
    title_repo = {}
    
    for line in lines:
        try:
            json_data = json.loads(line)
            org_question = Question(**json_data)
            print("\nProcessing question:", org_question.Id)
            print("Original question:", org_question.Title)
            
            # Store known repos with normalized URLs
            known_repos = list(org_question.Repos)
            ans_title_repo[org_question.Title] = known_repos
            
            # Get rephrased question
            rep_question = llm_rephrase(org_question.Title)
            rephrased = rep_question.choices[0].message.content  + cheatcode
            print("Rephrased question:", rephrased)
            
            # Get search results
            search_results = list(search(rephrased, stop=10))  # Convert generator to list
            print("Found initial results:", len(search_results))

            # Extract links from content
            link_list = []
            for link in search_results:
                content = get_html_content(link)
                content = clean_text(content)
                if content:
                    extracted_links = extract_links(content)
                    link_list.extend(extracted_links)

            # Normalize and classify all links
            classified_links = classifier(list(link_list + search_results))
            
            # Process and vectorize content
            processed_data = process_and_vectorize_content(classified_links)
            
            # Store results with additional metadata
            title_repo[org_question.Title] = {
                'classified_links': classified_links,
                'processed_content': processed_data,
                'original_question': org_question.Title,
                'rephrased_question': rephrased,
                'question_id': org_question.Id
            }

            # Print statistics
            print(f"\nClassified results:")
            print(f"- Codebases: {len(classified_links['codebases'])}")
            print(f"- Articles: {len(classified_links['articles'])}")
            print(f"- Forums: {len(classified_links['forums'])}")
            
            if processed_data:
                print(f"\nProcessed content:")
                print(f"- Total chunks: {processed_data['vectors'].shape[0]}")
                print(f"- Vector dimensions: {processed_data['vectors'].shape[1]}")

        except Exception as e:
            print(f"Error processing question: {str(e)}")
            continue

    return title_repo, ans_title_repo

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
    
def rerank_candidates_with_llm(question: str, candidates: List[dict], known_repos: List[str], max_candidates: int = 5) -> List[dict]:
    """
    Re-ranks candidate links using LLM based on repository content.
    
    Arguments:
    - question: The original question (str)
    - candidates: List of candidate repository links with occurrence counts (list of dict)
    - known_repos: List of known correct repositories (list of str)
    - max_candidates: Maximum number of candidates to analyze (int)
    
    Returns:
    - List of re-ranked candidates (list of dict)
    """
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
        return candidates

    # Prepare evaluation prompt
    evaluation_prompt = f"""
    Question: {question}
    
    Rate these GitHub repositories for their relevance and solution quality.
    Consider:
    1. Direct relevance to the question
    2. Code quality and implementation
    3. Documentation clarity
    4. Solution completeness
    
    Rate each repository from 0-10 and explain why.
    
    Repositories to evaluate:
    
    {"="*50}
    """ + "\n\n".join([
        f"Repository: {repo['url']}\n\nContent:\n{repo['content']}\n{'='*50}"
        for repo in candidates_with_content
    ]) + """
    
    Format your response exactly like this:
    {
        "rankings": [
            {
                "url": "repository_url",
                "score": score_number,
                "reasoning": "explanation"
            }
        ]
    }
    """
    
    try:
        # Get LLM evaluation
        evaluation = llm_prompt(evaluation_prompt)
        eval_data = json.loads(evaluation.choices[0].message.content.strip())
        
        # Update all candidates with scores
        for candidate in candidates:
            matching_rank = next(
                (r for r in eval_data['rankings'] 
                 if r['url'] == candidate['url']),
                None
            )
            
            if matching_rank:
                candidate['llm_score'] = float(matching_rank['score'])
                candidate['reasoning'] = matching_rank['reasoning']
                # Combined score considers both LLM score and occurrence count
                candidate['combined_score'] = (
                    candidate['llm_score'] * np.log(candidate['occurrences'] + 1)
                )
            else:
                # Keep candidates that weren't evaluated but with lower priority
                candidate['llm_score'] = 0
                candidate['reasoning'] = "Not evaluated"
                candidate['combined_score'] = np.log(candidate['occurrences'] + 1)
        
        # Sort by combined score
        return sorted(candidates, key=lambda x: x.get('combined_score', 0), reverse=True)
        
    except Exception as e:
        print(f"LLM ranking failed: {str(e)}")
        # Fall back to occurrence-based ranking
        for candidate in candidates:
            candidate['combined_score'] = candidate['occurrences']
        return sorted(candidates, key=lambda x: x['occurrences'], reverse=True)

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

def normalize_github_url(url: str) -> str:
    """Normalizes GitHub URLs to a standard format."""
    url = url.lower().strip('/')
    # Remove .git extension
    url = re.sub(r'\.git$', '', url)
    # Remove http/https prefix
    url = re.sub(r'^https?://', '', url)
    # Remove www.
    url = re.sub(r'^www\.', '', url)
    # Standardize github.com format
    url = re.sub(r'github\.com/', 'github.com/', url)
    return url

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
    path = "C:\\Users\\LENOVO\\OneDrive\\Documents\\Desktop\\RAiD-Repo\\web-raider\\questions.jsonl"
    results, known_repos = process_questions(path, limit=50)
    
    print("\nProcessing Results:")
    for title, data in results.items():
        print(f"\n{'='*50}")
        print(f"Question: {title}")
        
        # Print classification results
        classified_links = data['classified_links']
        print("\nClassified Links:")
        print(f"- Codebases: {len(classified_links['codebases'])}")
        print(f"- Articles: {len(classified_links['articles'])}")
        print(f"- Forums: {len(classified_links['forums'])}")
        
        # Print known repositories
        known_repos_list = known_repos[title]
        print(f"\nKnown repositories: {len(known_repos_list)}")
        
        # Print processed content statistics
        processed_content = data['processed_content']
        if processed_content:
            articles = processed_content['content']['articles']
            forums = processed_content['content']['forums']
            
            print(f"\nProcessed Content Statistics:")
            print(f"- Articles processed: {len(articles)}")
            print(f"- Forums processed: {len(forums)}")
            print(f"- Total vectors: {processed_content['vectors'].shape[0]}")
            
            # Analyze similarity and extract links
            print("\nAnalyzing content similarity...")
            analysis = analyze_similarity_and_extract_links(
                question=title,
                processed_content=processed_content,
                top_k=5
            )
            
            if analysis:
                print("\nTop Similar Chunks:")
                for i, chunk in enumerate(analysis['top_chunks'], 1):
                    print(f"\n{i}. From {chunk['content_type']} ({chunk['url']}):")
                    print(f"Similarity Score: {chunk['similarity_score']:.4f}")
                    print("Preview:", chunk['chunk_text'][:200] + "...")
                    if chunk['found_codebase_links']:
                        print("Found codebase links:")
                        for link in chunk['found_codebase_links']:
                            print(f"- {link}")
                
                # Repository matching
                known = set(known_repos_list)
                found = set(analysis['all_codebase_links'])
                matches = known.intersection(found)
                
                print(f"\nRepository Matching:")
                print(f"Found {len(matches)} out of {len(known)} known repositories")
                if matches:
                    print("Matched repositories:")
                    for repo in matches:
                        print(f"- {repo}")
                
                # Create and rank candidate list
                candidates = create_candidate_list(
                    classified_links=classified_links,
                    analysis_results=analysis
                )
                
                print("\nCandidate List (by occurrences):")
                for i, candidate in enumerate(candidates, 1):
                    print(f"{i}. {candidate['url']} (occurrences: {candidate['occurrences']})")
                '''
                # Re-rank with LLM
                ranked_candidates = rerank_candidates_with_llm(
                    question=title,
                    candidates=candidates,
                    known_repos=known_repos_list
                )
                
                print("\nRe-ranked Candidates:")
                for i, candidate in enumerate(ranked_candidates, 1):
                    print(f"\n{i}. {candidate['url']}")
                    print(f"Occurrences: {candidate['occurrences']}")
                    print(f"LLM Score: {candidate.get('llm_score', 'N/A')}")
                    print(f"Combined Score: {candidate.get('combined_score', 'N/A')}")
                    print(f"Reasoning: {candidate.get('reasoning', 'N/A')}")
                
                # Extract code from top repositories
                print("\nExtracting code from top repositories...")
                code_results = extract_from_top_candidates(ranked_candidates, k=3)
                
                print("\nExtracted Code:")
                for i, result in enumerate(code_results, 1):
                    print(f"\n{i}. Repository: {result['url']}")
                    print(f"Rank Score: {result['rank_score']}")
                    print(f"LLM Reasoning: {result['llm_reasoning']}")
                    print("\nCode Blocks:")
                    for j, code_block in enumerate(result['code_blocks'], 1):
                        print(f"\nBlock {j}:")
                        # Print first 500 characters of code with ellipsis if longer
                        print(code_block[:500] + ("..." if len(code_block) > 500 else ""))
                        print("-" * 50)
                '''
    print("\nEvaluating Results...")
    metrics = evaluate_model_accuracy(results, known_repos)
    
    print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.2%}")
    print(f"Total Matches: {metrics['total_matches']}")
    print(f"Total Known Repositories: {metrics['total_repos']}")