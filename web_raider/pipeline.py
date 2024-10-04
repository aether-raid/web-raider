from .shortlist import codebase_shortlist
from .evaluate import codebase_evaluate
from .model_calls import call_query_simplifier
from .utils import tidy_results, get_unique_codebases
import json

def pipeline(query: str, verbose: bool = False) -> list[dict]:
    """
    Process the query to shortlist and evaluate codebases.

    Parameters
    ----------
    query : str
        The query string to search for codebases.
    verbose : bool, optional
        If True, prints detailed information during the process (default is False).

    Returns
    -------
    tuple(list[dict], list[dict])
        A tuple containing a list of dictionaries containing information about the evaluated codebases and 
        a list of dictionaries containing information about the evaluated code snippets.
    """
    codebases, code_snippets = codebase_shortlist(query, verbose)
    desired_info = codebase_evaluate(query, codebases, verbose)

    return desired_info, code_snippets

def pipeline_main(user_query: str) -> list[dict]:
    """
    Main function.

    Parameters
    ----------
    user_query : str
        The query string to search for codebases.

    Returns
    -------
    tuple(list[dict], list[dict])
        A tuple containing a list of dictionaries containing information about the final evaluated codebases and
        a list of dictionaries containing information about the final evaluated code snippets.
    """
    potential_codebases = []
    code_snippets = []
    seen_cs_urls = set()

    # pipeline(QUERY, True)
    queries = call_query_simplifier(user_query)
    queries = json.loads(queries)

    for query in queries['prompts']:
        cb, cs = pipeline(query, True)
        potential_codebases.extend(cb)
        for code_snip in cs:
            if code_snip['url'] not in seen_cs_urls:
                code_snippets.append(code_snip)
                seen_cs_urls.add(code_snip['url'])

    final_codebases = codebase_evaluate(user_query, get_unique_codebases(potential_codebases), True)
    final_results = tidy_results(final_codebases)

    print(final_codebases)
    print(final_results)

    data = {
        'codebases': final_results,
        'snippets': code_snippets
    }

    return json.dumps(data)