# import pytest
from search import GoogleSearch


PROMPT = "How to implement a React Codebase?"
MAX_CAP = 20


def test_search_init():
    GoogleSearch(PROMPT)

def test_search_number():
    result = GoogleSearch(PROMPT)
    
    num = 0
    for i in result.search_results:
        num += 1
    
    assert num == MAX_CAP

def test_relevant_search():
    result = GoogleSearch(PROMPT)
    
    blacklist = result.blacklist
    
    num = 0
    
    for url in result.relevant_urls():
        for item in blacklist:
            assert item not in url
        num += 1
    
    assert num <= MAX_CAP

