# src/search.py

from googlesearch import search, SearchResult
from typing import Generator, Union

from src.constants import CODE_BLACKLIST


class GoogleSearch:
    def __init__(self, query: str,
                 num_results: int = 20,
                 blacklist: list[str] = CODE_BLACKLIST):
        self.query = query
        self.cap = num_results
        self.blacklist = blacklist
    
    def run_query(self) -> None:
        self.search_results = list(search(
                self.query, num_results=self.cap,
                advanced=True
            ))

    def get_relevant_urls(self) -> list[str]:
        results = set()
        
        for result in self.search_results:
            if result.url not in results and all(domain not in result.url for domain in self.blacklist):
                results.add(result.url)

        return list(results)