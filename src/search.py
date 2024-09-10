# src/search.py

from googlesearch import search, SearchResult
from typing import Generator, Union

from src.constants import CODE_BLACKLIST


class GoogleSearch:
    def __init__(self, query: str,
                 num_results: int = 20,
                 blacklist: list[str] = CODE_BLACKLIST):
        self.search_results = list(search(
                query, num_results=num_results,
                advanced=True
            ))
        self.cap = num_results
        self.blacklist = blacklist       
    
    def urls(self) -> Generator[str, None, None]:
        results = set()
        
        for result in self.search_results:
            if result.url not in results:
                yield result.url
                results.add(result.url)
        
        
    def relevant_urls(self) -> Generator[str, None, None]:
        for url in self.urls():
            if all(domain not in url for domain in self.blacklist):
                yield url