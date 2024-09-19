from src.article import CodeArticle
from src.search import GoogleSearch
from src.codebase import Codebase, CodebaseType, GitHubCodebase
from src.url_classifier import url_classifier
from src.model_calls import call_query_simplifier, call_relevance, call_pro_con, call_scorer
from src.utils import json_to_table
import builtins
import json

def useless_func(*_):
    return

def pipeline(query, verbose = False): 
    print = builtins.print if verbose else useless_func

    print("Searching Google...")
    googlesearch = GoogleSearch(query)
    googlesearch.run_query()
    results = googlesearch.get_relevant_urls()
        
    print(f"Found {len(results)} Relevant Results!\n")
    
    codebases = []
    seen_urls = set()
    
    for url in results:
        if url in seen_urls:
            continue
        
        seen_urls.add(url)
        print(url)

        if url_classifier(url) == 'Codebase':
            object = Codebase(url)

            if object.is_code(object.repository_url):
                codebase_type = CodebaseType.format_type(object.type)

                # first check if the link does lead to a repository
                if object.check_is_repo():
                    print(f"Found {codebase_type} Repository {object.repository_url}")

                    codebase = {
                        'url': object.repository_url,
                        'info': object.combine_info(),
                    }

                    # relevance check for each codebase separately to circumvent token limit
                    if str(call_relevance([codebase], query)):
                        codebases.append(codebase)
        
        elif url_classifier(url) == 'Article':
            print(f"Opening Article, {url}...")
            object = CodeArticle(url)

            for cb in object.code_urls():
                # if url already is checked ignore it
                if cb.original_url not in seen_urls:
                    seen_urls.add(cb.original_url)
                    
                    codebase_type = CodebaseType.format_type(cb.type)

                    # first check if the link does lead to a repository
                    if cb.check_is_repo():
                        print(f"Found {codebase_type} Repository {cb.repository_url}")

                        codebase = {
                            'url': cb.repository_url,
                            'info': cb.combine_info(),
                        }

                        # relevance check for each codebase separately to circumvent token limit
                        if str(call_relevance([codebase], query)):
                            codebases.append(codebase)

        elif url_classifier(url) == 'Forum':
            pass
            # object = CodeForum(url)

        else:
            continue
        
        print("\n-------------------------------------------------------------------\n")
    
    print(f"\nFound {len(codebases)} Codebases!")

    if codebases:
        # now do pros and cons evaluation
        pro_con_response = call_pro_con(codebases, query)

        # now get the scorer
        scorer_response = call_scorer(codebases, query, pro_con_response)

        with open('output.txt', 'a') as file:
            file.write(f'Query: {query}')
            file.write('\n\n')
            file.write(scorer_response)
            file.write('\n\n')
            file.write(json_to_table(pro_con_response))
    else:
        with open('output.txt', 'a') as file:
            file.write(f'Query: {query}')
            file.write('\n\n')
            file.write('No relevant codebases found.\n\n')

# test portion cuz im lazy to write test file
def main():
    SAMPLE_QUERY = 'Can you code a VSCode Extension using React?'

    QUERY = 'How do I parse Javascript AST in Python with Tree-Sitter?'

    # new query for "codebase research problem" according to aloysius
    CODEBASE_QUERY = """
    i specifically want open source alternatives to this:

    Flowith is an innovative, canvas-based AI tool designed for content generation and deep work. It allows users to interactively create and organize various types of content, including long texts, code, and images, using a visually intuitive interface.
    """

    # pipeline(QUERY, True)
    queries = call_query_simplifier(CODEBASE_QUERY)
    queries = json.loads(queries)

    for query in queries['prompts']:
        pipeline(query, True)

main()
