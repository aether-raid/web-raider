from src.article import CodeArticle
from src.search import GoogleSearch
from src.codebase import Codebase, CodebaseType, GitHubCodebase
from src.url_classifier import url_classifier
from assets.relevance import call_relevance
import builtins
# import json    # just to dump the codebases thing to see how shit works

def useless_func(*_):
    return

def pipeline(query, verbose = False):
    print = builtins.print if verbose else useless_func

    print("Searching Google...")
    googlesearch = GoogleSearch(query)
    googlesearch.run_query()
    results = googlesearch.get_relevant_urls()
        
    print(f"Found {len(results)} Relevant Results!\n")
    
    codebases = {}
    
    for url in results:
        print(url)
        original_num = len(codebases)

        if url_classifier(url) == 'Codebase':
            object = Codebase(url)

            if object.is_code(object.original_url):
                codebase_type = CodebaseType.format_type(object.type)
                print(f"Found {codebase_type} Repository {object.original_url}!")
                codebases[object.original_url] = object.combine_info()
        
        elif url_classifier(url) == 'Article':
            print(f"Opening Article, {url}...")
            object = CodeArticle(url)

            for codebase in object.code_urls():
                codebase_type = CodebaseType.format_type(codebase.type)
                print(f"Found {codebase_type} Repository {codebase.original_url}!")
                codebases[codebase.original_url] = codebase.combine_info()

        elif url_classifier(url) == 'Forum':
            pass
            # object = CodeForum(url)

        else:
            pass
        
        if original_num == len(codebases):
            print("Found NO new codebases!")
        else:
            print(f"Found {len(codebases) - original_num} new codebases!")
        
        print("\n-------------------------------------------------------------------\n")
    
    print(f"\nFound {len(codebases)} Codebases!")

    response = call_relevance(codebases, QUERY)

    with open('output.txt', 'w') as file:
        file.write(response)

    file.close()

# test portion cuz im lazy to write test file
QUERY = 'Can you code a VSCode Extension using React?'

pipeline(QUERY, True)