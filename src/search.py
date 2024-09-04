# src/search.py

from googlesearch import search, SearchResult
from typing import Generator

from constants import CODE_BLACKLIST


class GoogleSearch:
    def __init__(self, query: str,
                 num_results: int = 20,
                 blacklist: list[str] = CODE_BLACKLIST):
        self.search_results = search(
            query, num_results=num_results,
            advanced=True
        )
        self.blacklist = blacklist
            
        self._search_archive: list[SearchResult] = []
    
    def urls(self) -> Generator[str, None, None]:
        results = set()
        
        for result in self._search_archive:
            # if all(domain not in result.url for domain in self.blacklist) and result.url not in results:
            yield result.url
            results.add(result.url)
        
        for result in self.search_results:
            self._search_archive.append(result)
            # if all(domain not in result.url for domain in self.blacklist) and result.url not in results:
            yield result.url
            results.add(result.url)
        
        
    def relevant_urls(self) -> Generator[str, None, None]:
        results = set()
        
        for result in self._search_archive:
            if all(domain not in result.url for domain in self.blacklist) and result.url not in results:
                yield result.url
                results.add(result.url)
        
        for result in self.search_results:
            self._search_archive.append(result)
            if all(domain not in result.url for domain in self.blacklist) and result.url not in results:
                yield result.url
                results.add(result.url)