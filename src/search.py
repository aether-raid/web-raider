# src/search.py

from googlesearch import search, SearchResult
from typing import Generator, Union

from constants import CODE_BLACKLIST


class GoogleSearch:
    def __init__(self, query: Union[str, list[SearchResult]],
                 num_results: int = 20,
                 blacklist: list[str] = CODE_BLACKLIST):
        self.search_results: list[SearchResult]
        if isinstance(query, str):
            self.search_results = list(search(
                query, num_results=num_results,
                advanced=True
            ))
        else:
            self.search_results = query.copy()[:num_results]
        
        self.cap = num_results
        self.blacklist = blacklist
    
    def copy(self):
        # because google is dumb and can't handle temperature = 0
        return GoogleSearch(self.search_results,
                            num_results=self.cap,
                            blacklist=self.blacklist)
        
    
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