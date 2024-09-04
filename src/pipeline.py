from article import CodeArticle
from search import GoogleSearch
from codebase import format_type


def useless_func(*_):
    return

def pipeline(query, verbose = False):
    if not verbose:
        print = useless_func

    print("Searching Google...")
    search_result = GoogleSearch(query)
    
    results = []
    
    for url in search_result.relevant_urls():
        results.append(url)
        
    print(f"Found {len(results)} Relevant Results!\n")
    
    codebases = {}
    
    for url in results:
        print(f"Opening Article, {url}...")
        article = CodeArticle(url)
        
        original_num = len(codebases)
        
        for codebase in article.code_urls():
            codebase_type = format_type(codebase.type)
            print(f"Found {codebase_type} Repository {codebase.repository_url}!")
            codebases[codebase.repository_url]  = codebase
        
        if original_num == len(codebases):
            print("Found NO new codebases!")
        else:
            print(f"Found {len(codebases) - original_num} new codebases!")
        
        print("\n-------------------------------------------------------------------\n")
    
    print(f"\nFound {len(codebases)} Codebases!")
    
    return [i for i in codebases]