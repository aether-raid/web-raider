from src.shortlist import codebase_shortlist
from src.evaluate import codebase_evaluate
from src.model_calls import call_query_simplifier
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
    list[dict]
        A list of dictionaries containing information about the evaluated codebases.
    """
    codebases = codebase_shortlist(query, verbose)
    desired_info = codebase_evaluate(query, codebases, verbose)

    return desired_info

# test portion cuz im lazy to write test file
def main() -> list[dict]:
    """
    Main function to test the pipeline with sample queries.

    Returns
    -------
    list[dict]
        A list of dictionaries containing information about the final evaluated codebases.
    """
    SAMPLE_QUERY = 'Can you code a VSCode Extension using React?'

    QUERY = 'How do I parse Javascript AST in Python with Tree-Sitter?'

    # new query for "codebase research problem" according to aloysius
    CODEBASE_QUERY = """
    i specifically want open source alternatives to this:

    Flowith is an innovative, canvas-based AI tool designed for content generation and deep work. It allows users to interactively create and organize various types of content, including long texts, code, and images, using a visually intuitive interface.
    """

    potential_codebases = []

    # pipeline(QUERY, True)
    queries = call_query_simplifier(CODEBASE_QUERY)
    queries = json.loads(queries)

    for query in queries['prompts']:
        potential_codebases.extend(pipeline(query, True))

    final_codebases = codebase_evaluate(CODEBASE_QUERY, potential_codebases, True)

    return final_codebases

main()
