from googlesearch import search

CODE_BLACKLIST = [
    "stackoverflow.com",
    "quora.com",
    "tutorialspoint.com",
    "w3schools.com",
    "programiz.com",
    "javatpoint.com",
    "geeksforgeeks.org"
]

class GoogleSearch:
    def __init__(self, query: str,
                 num_results: int = 20,
                 blacklist: list[str] = CODE_BLACKLIST):
        self.search_results = search(
            query, num_results=num_results,
            advanced=True
        )
        self.blacklist = blacklist
        
    def relevant_urls(self):
        results = set()
        for result in self.search_results:
            if all(domain not in result.url for domain in self.blacklist) and result.url not in results:
                yield result.url
                results.add(result.url)