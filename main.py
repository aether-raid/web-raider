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
import base64
import json

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

def llamas(prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': f'rephrase the question "{prompt}" and only produce the rephrased question with nothing else',
            }
        ],
        model='mistral',
    )
    return chat_completion

def check_url_status(url, timeout=15):
    """Check if URL is accessible."""
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
    """Filter out dead links using parallel requests."""
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Map URLs to their status
        url_status = list(executor.map(check_url_status, urls))
        # Return only live URLs
        return [url for url, is_live in zip(urls, url_status) if is_live]

def classifier(results):
    """
    Classifies URLs into codebases, articles, and forums.
    Returns dictionary containing filtered lists.
    """
    codebases = []
    articles = []
    forums = []
    seen_urls = set()

    # First filter out dead links
    live_urls = filter_dead_links(results)
    
    for url in live_urls:
        if url in seen_urls:
            continue

        seen_urls.add(url)

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
        print(f"Failed to fetch {url}")
        return 0

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
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

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
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
    """Process and vectorize content from articles and forums."""
    processed_content = {
        'articles': [],
        'forums': []
    }
    
    # Process articles and forums
    for content_type in ['articles', 'forums']:
        print(f"\nProcessing {content_type}...")
        
        for url in classified_links[content_type]:
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
                continue
    
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

def process_questions(path: str, limit: int = 5):
    """Process questions from JSONL file and classify links."""
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
            known_repos = [repo.lower().rstrip('/') for repo in org_question.Repos]
            ans_title_repo[org_question.Title] = known_repos
            
            # Get rephrased question
            rep_question = llamas(org_question.Title)
            rephrased = rep_question.choices[0].message.content
            print("Rephrased question:", rephrased)
            
            # Get search results
            search_results = []
            for result in search(rephrased, num_results=10):
                search_results.append(result.lower().rstrip('/'))  # Normalize URLs
            print("Found initial results:", len(search_results))

            # Extract links from content
            link_list = []
            for link in search_results:
                content = get_html_content(link)
                if content:
                    extracted_links = [link.lower().rstrip('/') for link in extract_links(content)]
                    link_list.extend(extracted_links)

            # Normalize and classify all links
            normalized_links = list(set(link_list + search_results))  # Remove duplicates
            classified_links = classifier(normalized_links)
            
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

def analyze_similarity_and_extract_links(question: str, processed_content: dict, top_k: int = 5):
    """
    Analyze chunk similarity using LSA and extract codebase links from top chunks.
    
    Args:
        question: Original or rephrased question
        processed_content: Dictionary containing vectors and metadata
        top_k: Number of top results to return
    """
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
    """
    Create and sort candidate list based on occurrences.
    
    Args:
        classified_links: Original classified links dictionary
        analysis_results: Results from LSA analysis containing extracted links
    """
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
    
    # Sort links by occurrence count
    sorted_candidates = [
        {
            'url': link,
            'occurrences': count
        }
        for link, count in link_counts.most_common()
    ]
    
    return sorted_candidates

def rerank_candidates_with_llm(question: str, candidates: List[dict], known_repos: List[str]) -> List[dict]:
    """Re-rank candidate links using LLM evaluation."""
    # Simpler prompt to reduce JSON parsing errors
    evaluation_prompt = f"""
    Analyze these GitHub repositories for the question: "{question}"
    Return scoring in this exact JSON format:
    {{"rankings": [
        {{"url": "[repo_url]", "score": [0-10], "reasoning": "[explanation]"}}
    ]}}

    Repositories to evaluate:
    {[cand['url'] for cand in candidates]}
    """
    
    try:
        evaluation = llamas(evaluation_prompt)
        response_text = evaluation.choices[0].message.content.strip()
        
        # Clean response text for better JSON parsing
        if not response_text.startswith('{'): 
            response_text = '{' + response_text.split('{', 1)[1]
        response_text = response_text.replace('\n', ' ').replace('\\', '')
        
        eval_data = json.loads(response_text)
        
        # Update scores
        for candidate in candidates:
            for ranking in eval_data.get('rankings', []):
                if ranking['url'] in candidate['url']:  # More flexible matching
                    candidate['llm_score'] = float(ranking.get('score', 0))
                    candidate['reasoning'] = ranking.get('reasoning', 'No reasoning provided')
                    candidate['combined_score'] = candidate['llm_score'] * np.log(candidate['occurrences'] + 1)
                    break
            else:
                candidate['llm_score'] = 0
                candidate['reasoning'] = "Not evaluated"
                candidate['combined_score'] = 0
        
        return sorted(candidates, key=lambda x: x.get('combined_score', 0), reverse=True)
    except Exception as e:
        print(f"LLM ranking failed: {str(e)}")
        return candidates

def extract_code_from_repo(url: str) -> dict:
    """Extract code from repository URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        code_blocks = []
        
        # Find code elements with expanded selectors
        if 'github.com' in url.lower():
            code_elements = soup.find_all(['pre', 'div'], class_=['highlight', 'highlight-source', 'blob-code', 'js-file-line'])
        else:
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
    """
    Extract code from top k ranked repositories.
    """
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
    """
    Evaluate model accuracy by comparing found repositories with known repositories.
    """
    total_matches = 0
    total_repos = 0
    question_metrics = {}
    
    for title, data in results.items():
        # Get all found repos (including normalized URLs)
        found_repos = set()
        if 'classified_links' in data:
            # Add repos from classified links
            for repo in data['classified_links']['codebases']:
                # Normalize URL and add variations
                base_url = repo.lower().rstrip('/')
                found_repos.add(base_url)
                # Add common variations
                if 'github.com' in base_url:
                    found_repos.add(base_url.replace('github.com/', 'github.com'))
                    found_repos.add(base_url + '.git')

        # Normalize known repos
        known = set()
        for repo in known_repos[title]:
            base_url = repo.lower().rstrip('/')
            known.add(base_url)
            if 'github.com' in base_url:
                known.add(base_url.replace('github.com/', 'github.com'))
                known.add(base_url + '.git')

        # Find matches
        matches = found_repos.intersection(known)
        
        # Update counts
        matches_count = len(matches)
        known_count = len(known_repos[title])  # Use original count
        total_matches += matches_count
        total_repos += known_count
        
        # Calculate accuracy
        accuracy = matches_count / known_count if known_count > 0 else 0
        
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

    # Calculate overall accuracy
    overall_accuracy = total_matches / total_repos if total_repos > 0 else 0
    
    return {
        'overall_accuracy': overall_accuracy,
        'total_matches': total_matches,
        'total_repos': total_repos,
        'question_metrics': question_metrics
    }

if __name__ == "__main__":
    path = "C:\\Users\\65881\\Downloads\\questions.jsonl\\questions.jsonl"
    results, known_repos = process_questions(path, limit=5)
    
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

    print("\nEvaluating Results...")
    metrics = evaluate_model_accuracy(results, known_repos)
    
    print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.2%}")
    print(f"Total Matches: {metrics['total_matches']}")
    print(f"Total Known Repositories: {metrics['total_repos']}")    

#previous main block start
'''if __name__ == "__main__":
    path = "C:\\Users\\65881\\Downloads\\questions.jsonl\\questions.jsonl"
    results, known_repos,ans_title_repo = process_questions(path, limit=5)
    
    # Print final results
    print("\nFinal Results:")
    for title, classified_links in title_repo.items():
        print(f"\nQuestion: {title}")
        print(f"Number of found codebases: {len(classified_links['codebases'])}")
        print(f"Number of found articles: {len(classified_links['articles'])}")
        print(f"Number of found forums: {len(classified_links['forums'])}")
        
        # Compare with known repositories
        known_repos = ans_title_repo[title]
        print(f"Known repositories: {len(known_repos)}")
        
        # Print the actual links
        print("\nCodebase Links:")
        for link in classified_links['codebases']:
            print(f"- {link}")
            
        print("\nArticle Links:")
        for link in classified_links['articles']:
            print(f"- {link}")
            
        print("\nForum Links:")
        for link in classified_links['forums']:
            print(f"- {link}")

    # Print final results
    print("\nFinal Results:")
    for title, data in results.items():
        print(f"\nQuestion: {title}")
        classified_links = data['classified_links']
        processed_content = data['processed_content']
        
        print(f"Found links:")
        print(f"- Codebases: {len(classified_links['codebases'])}")
        print(f"- Articles: {len(classified_links['articles'])}")
        print(f"- Forums: {len(classified_links['forums'])}")
        
        if processed_content:
            articles = processed_content['content']['articles']
            forums = processed_content['content']['forums']
            
            print(f"\nProcessed content:")
            print(f"- Articles processed: {len(articles)}")
            print(f"- Forums processed: {len(forums)}")
            print(f"- Total vectors: {processed_content['vectors'].shape[0]}")
            
            # Print sample chunks
            print("\nSample chunks:")
            for content_type in ['articles', 'forums']:
                docs = processed_content['content'][content_type]
                if docs:
                    print(f"\nFrom {content_type}:")
                    doc = docs[0]
                    print(f"URL: {doc['url']}")
                    if doc['chunks']:
                        print("First chunk preview:")
                        print(doc['chunks'][0][:200] + "...")
        print("\nAnalyzing content similarity and extracting links...")
    
    for title, data in results.items():
        print(f"\nQuestion: {title}")
        processed_content = data['processed_content']
        
        if processed_content:
            analysis = analyze_similarity_and_extract_links(
                question=title,
                processed_content=processed_content,
                top_k=5
            )
            
            if analysis:
                print("\nTop similar chunks:")
                for i, chunk in enumerate(analysis['top_chunks'], 1):
                    print(f"\n{i}. From {chunk['content_type']} ({chunk['url']}):")
                    print(f"Similarity Score: {chunk['similarity_score']:.4f}")
                    print("Preview:", chunk['chunk_text'][:200] + "...")
                    if chunk['found_codebase_links']:
                        print("Found codebase links:")
                        for link in chunk['found_codebase_links']:
                            print(f"- {link}")
                
                print("\nAll unique codebase links found in top chunks:")
                for link in analysis['all_codebase_links']:
                    print(f"- {link}")
                
                # Compare with known repositories
                known = set(known_repos[title])
                found = set(analysis['all_codebase_links'])
                matches = known.intersection(found)
                
                print(f"\nRepository matching:")
                print(f"Found {len(matches)} out of {len(known)} known repositories")
                if matches:
                    print("Matched repositories:")
                    for repo in matches:
                        print(f"- {repo}")'''
#previous main block end


#oldest start
'''def classifier(results):
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

    return codebases'''

'''def extract_links(text):
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
print(pass_rate/limit)'''
#oldest end